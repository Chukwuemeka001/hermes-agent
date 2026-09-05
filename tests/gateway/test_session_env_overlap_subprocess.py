"""Gateway task -> executor -> local env bridge -> real child regression.

No AIAgent constructor is involved: cached turns cannot repair a missing
handler binding. This is a boundary integration test, not a provider turn.
"""
import asyncio
import json
import os
import subprocess
import sys
import threading
from contextvars import Context

import pytest

from gateway.platforms.base import Platform
import gateway.session_context as session_context
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from gateway.session_context import get_session_env
from tools.environments.local import _inject_session_context_env


SESSION_ID = "HERMES_SESSION_ID"
THREAD_ID = "HERMES_SESSION_THREAD_ID"


@pytest.fixture(autouse=True)
def _isolate_engagement(monkeypatch):
    # Context() isolates task values, not this process-global latch.
    # monkeypatch restores the entry value even after gateway calls set it.
    monkeypatch.setattr(session_context, "_session_context_engaged", False)


def _context(sid, thread):
    return SessionContext(
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="synthetic-chat",
            chat_type="dm",
            user_id="synthetic-user",
            thread_id=thread,
        ),
        connected_platforms=[],
        home_channels={},
        session_key=f"synthetic-session:{thread}",
        session_id=sid,
    )


def _child_identity(home):
    # Deliberately seed foreign snapshot values, never copy the host env.
    env = {
        "HOME": str(home),
        "HERMES_HOME": str(home),
        SESSION_ID: "foreign-session",
        THREAD_ID: "foreign-thread",
    }
    if "SystemRoot" in os.environ:
        env["SystemRoot"] = os.environ["SystemRoot"]
    _inject_session_context_env(env)
    child = subprocess.run(
        [sys.executable, "-I", "-c",
         "import json, os; print(json.dumps([os.getenv('HERMES_SESSION_ID'), "
         "os.getenv('HERMES_SESSION_THREAD_ID')]))"],
        env=env, cwd=home, capture_output=True, text=True, check=True, timeout=15,
    )
    return json.loads(child.stdout)


def test_overlapping_cached_turns_reach_real_children(monkeypatch, tmp_path):
    runner = object.__new__(GatewayRunner)
    monkeypatch.setenv(SESSION_ID, "foreign-session")
    monkeypatch.setenv(THREAD_ID, "foreign-thread")
    barrier = threading.Barrier(2, timeout=15)
    main_thread = threading.get_ident()

    def cached_work():
        assert threading.get_ident() != main_thread
        # Neither worker may read its bridge until both handlers are bound
        # and both executor copies are running. A sequential test deadlocks.
        barrier.wait()
        result = _child_identity(tmp_path)
        barrier.wait()
        return result

    async def turn(sid, thread):
        tokens = runner._set_session_env(_context(sid, thread))
        try:
            return await runner._run_in_executor_with_context(cached_work)
        finally:
            runner._clear_session_env(tokens)
            assert (get_session_env(SESSION_ID), get_session_env(THREAD_ID)) == ("", "")

    async def overlap():
        # Repeat on the same runner/executor, with no constructor-time binding.
        for _ in range(2):
            results = await asyncio.gather(
                turn("synthetic-a", "thread-a"),
                turn("synthetic-b", "thread-b"),
            )
            assert results == [["synthetic-a", "thread-a"],
                               ["synthetic-b", "thread-b"]]

    try:
        Context().run(asyncio.run, overlap())
        assert os.environ[SESSION_ID] == "foreign-session"
        assert os.environ[THREAD_ID] == "foreign-thread"
        # A clean context after gateway engagement strips foreign fallback.
        assert Context().run(_child_identity, tmp_path) == [None, None]
    finally:
        runner._get_executor().shutdown(wait=True)


def test_child_environment_retains_system_root(monkeypatch, tmp_path):
    # Windows requires this OS location when an explicit env is supplied.
    # Keep the real child process, observing only the env passed to it.
    root = os.environ.get("SystemRoot", str(tmp_path))
    monkeypatch.setenv("SystemRoot", root)
    real_run = subprocess.run
    observed = []

    def run_child(*args, **kwargs):
        observed.append(kwargs["env"].get("SystemRoot"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run_child)
    Context().run(_child_identity, tmp_path)
    assert observed == [root]


@pytest.mark.parametrize("sid", ["", None])
def test_blank_identity_and_exception_cleanup(sid, monkeypatch, tmp_path):
    runner = object.__new__(GatewayRunner)
    monkeypatch.setenv(SESSION_ID, "foreign-session")

    async def turn():
        tokens = runner._set_session_env(_context(sid, "blank-thread"))
        try:
            result = await runner._run_in_executor_with_context(
                _child_identity, tmp_path
            )
            assert result == ["", "blank-thread"]
            raise ValueError("synthetic turn failure")
        finally:
            runner._clear_session_env(tokens)
            # Cleanup intentionally blanks rather than restoring a foreign
            # process fallback (or a previously bound outer identity).
            assert get_session_env(SESSION_ID) == ""
            assert get_session_env(THREAD_ID) == ""
            assert _child_identity(tmp_path) == ["", ""]

    try:
        with pytest.raises(ValueError, match="synthetic turn failure"):
            Context().run(asyncio.run, turn())
        assert os.environ[SESSION_ID] == "foreign-session"
        assert Context().run(_child_identity, tmp_path) == [None, None]
    finally:
        runner._get_executor().shutdown(wait=True)
