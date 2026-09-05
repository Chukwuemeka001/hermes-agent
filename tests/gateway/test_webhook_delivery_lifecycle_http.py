"""Real HTTP delivery lifecycles; only the downstream platform is a fake.

No provider, agent, or live gateway is used. Events bound concurrency rather
than sleeps; the adapter-local clock advances without changing asyncio time.
"""

import asyncio
import hashlib
import hmac
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from aiohttp import ClientTimeout, ServerDisconnectedError, web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter


_SIGNING_KEY = "synthetic-webhook-lifecycle-test-key"


class Target:
    def __init__(self, *, failure=None, blocked=False):
        self.failure = failure
        self.blocked = blocked
        self.calls = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.task = None

    async def send(self, chat_id, content, metadata=None):
        self.calls.append((chat_id, content, metadata))
        if len(self.calls) == 1:
            self.task = asyncio.current_task()
            self.entered.set()
            if self.blocked:
                await self.release.wait()
            if isinstance(self.failure, Exception):
                raise self.failure
            if self.failure is not None:
                return self.failure
        return SendResult(success=True)


@asynccontextmanager
async def delivery_client(target):
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={
        "routes": {"alerts": {
            "secret": _SIGNING_KEY,
            "deliver_only": True,
            "deliver": "telegram",
            "deliver_extra": {"chat_id": "{chat}"},
            "prompt": "Alert: {message}",
        }},
    }))
    adapter.gateway_runner = SimpleNamespace(adapters={Platform.TELEGRAM: target})

    async def no_agent(_event):
        pytest.fail("deliver_only must not invoke the agent")

    adapter.handle_message = no_agent
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    async with TestClient(TestServer(app), timeout=ClientTimeout(total=3)) as client:
        try:
            yield adapter, client
        finally:
            target.release.set()
            if target.task is not None and not target.task.done():
                target.task.cancel()
                await asyncio.gather(target.task, return_exceptions=True)


async def post(client, delivery_id="same-id"):
    body = json.dumps({"chat": "chat-123", "message": "disk full"}).encode()
    signature = hmac.new(_SIGNING_KEY.encode(), body, hashlib.sha256).hexdigest()
    response = await client.post(
        "/webhooks/alerts",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Delivery": delivery_id,
            "X-Hub-Signature-256": "sha256=" + signature,
        },
    )
    return response.status, await response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    SendResult(success=False, error="downstream unavailable"),
    RuntimeError("downstream unavailable"),
], ids=["unsuccessful-result", "exception"])
async def test_http_failure_retry_success_then_duplicate(failure):
    target = Target(failure=failure)
    async with delivery_client(target) as (_, client):
        status, body = await post(client)
        assert status == 502
        assert body["error"] == "Delivery failed"
        assert "downstream unavailable" not in str(body)
        status, body = await post(client)
        assert status == 200
        assert body["status"] == "delivered"
        status, body = await post(client)
        assert status == 200
        assert body["status"] == "duplicate"
    assert target.calls == [("chat-123", "Alert: disk full", None)] * 2


@pytest.mark.asyncio
async def test_http_inflight_survives_ttl_and_success_gets_full_ttl(monkeypatch):
    target = Target(blocked=True)
    clock = SimpleNamespace(now=1000.0)
    # Replace only this module's clock, not the process-wide time module.
    monkeypatch.setattr("gateway.platforms.webhook.time", SimpleNamespace(time=lambda: clock.now))
    async with delivery_client(target) as (adapter, client):
        adapter._idempotency_ttl = 10
        first = asyncio.create_task(post(client))
        try:
            await asyncio.wait_for(target.entered.wait(), timeout=2)
            clock.now += 11
            status, body = await post(client)
            assert status == 200
            assert body["status"] == "duplicate"
            assert len(target.calls) == 1
            target.release.set()
            status, body = await first
            assert status == 200
            assert body["status"] == "delivered"
            clock.now += 9
            status, body = await post(client)
            assert status == 200
            assert body["status"] == "duplicate"
            clock.now += 2
            status, body = await post(client)
            assert status == 200
            assert body["status"] == "delivered"
        finally:
            target.release.set()
            await asyncio.gather(first, return_exceptions=True)
    assert target.calls == [("chat-123", "Alert: disk full", None)] * 2


@pytest.mark.asyncio
async def test_http_cancelled_handler_releases_claim_for_retry():
    target = Target(blocked=True)
    async with delivery_client(target) as (_, client):
        first = asyncio.create_task(post(client))
        try:
            await asyncio.wait_for(target.entered.wait(), timeout=2)
            # Cancel the actual aiohttp request task suspended inside send().
            # Cancelling a client task alone need not cancel a server handler.
            target.task.cancel()
            with pytest.raises(ServerDisconnectedError):
                await first
            status, body = await post(client)
            assert status == 200
            assert body["status"] == "delivered"
            status, body = await post(client)
            assert status == 200
            assert body["status"] == "duplicate"
        finally:
            target.release.set()
            await asyncio.gather(first, return_exceptions=True)
    assert target.calls == [("chat-123", "Alert: disk full", None)] * 2
