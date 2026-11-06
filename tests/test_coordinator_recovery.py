"""Recovery tests for the coordinator: the integration must always heal itself.

Each scenario here is a failure seen on real devices. See upstream issues #363,
#393 and #396.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from unittest.mock import MagicMock, patch

import pytest

from custom_components.philips_airpurifier_coap import coordinator as coordinator_module
from custom_components.philips_airpurifier_coap.coordinator import Coordinator

from .fake_coap import FakeClient, wait_until

HOST = "192.0.2.1"
STATUS = {"pwr": "1"}

LOST = object()
"""Scripted result for CoAPClient.create(): the sync reply never arrives."""


@pytest.fixture(autouse=True)
def short_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink every delay so that the tests run quickly."""
    monkeypatch.setattr(coordinator_module, "CONNECT_TIMEOUT", 0.02)
    monkeypatch.setattr(coordinator_module, "CONNECT_RETRY_DELAY", 0.001)


@pytest.fixture
def script_create() -> AsyncIterator[Callable[..., MagicMock]]:
    """Script what each CoAPClient.create() call does."""
    patchers = []

    def _script(*results: object) -> MagicMock:
        remaining = list(results)

        async def create(_host: str) -> object:
            result = remaining.pop(0) if remaining else FakeClient()
            if result is LOST:
                await asyncio.Event().wait()
            if isinstance(result, Exception):
                raise result
            return result

        patcher = patch.object(coordinator_module.CoAPClient, "create", side_effect=create)
        patchers.append(patcher)
        return patcher.start()

    yield _script

    for patcher in patchers:
        patcher.stop()


@pytest.fixture
async def make_coordinator() -> AsyncIterator[Callable[[FakeClient], Coordinator]]:
    """Build coordinators and shut them down after the test."""
    created: list[Coordinator] = []

    def _make(client: FakeClient) -> Coordinator:
        coordinator = Coordinator(MagicMock(), client, HOST, dict(STATUS))
        created.append(coordinator)
        return coordinator

    yield _make

    for coordinator in created:
        await coordinator.shutdown()


def warnings_from_coordinator(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return the warnings logged by the coordinator."""
    return [r for r in caplog.records if r.name == coordinator_module.__name__ and r.levelno == logging.WARNING]


async def test_lost_sync_is_retried_until_the_device_answers(script_create, make_coordinator) -> None:
    """A sync that gets no reply must not hang the reconnect (#396 section 1)."""

    new_client = FakeClient()
    create = script_create(LOST, LOST, new_client)
    coordinator = make_coordinator(FakeClient())

    await coordinator.reconnect()
    await wait_until(lambda: coordinator.client is new_client)

    assert create.call_count == 3
    await wait_until(lambda: new_client.observe_calls == 1)


async def test_failed_connect_is_retried(script_create, make_coordinator) -> None:
    """An error from create() is retried like a lost sync."""

    new_client = FakeClient()
    script_create(OSError("no route to host"), new_client)
    coordinator = make_coordinator(FakeClient())

    await coordinator.reconnect()

    await wait_until(lambda: coordinator.client is new_client)


async def test_reconnect_never_gives_up(script_create, make_coordinator) -> None:
    """However long the outage lasts, the reconnect keeps trying."""

    new_client = FakeClient()
    script_create(*[OSError("down")] * 50, new_client)
    coordinator = make_coordinator(FakeClient())

    await coordinator.reconnect()

    await wait_until(lambda: coordinator.client is new_client)


async def test_retry_delay_does_not_grow_during_an_outage(script_create, make_coordinator) -> None:
    """A long outage must not be followed by a long wait to notice its end."""

    new_client = FakeClient()
    script_create(*[OSError("down")] * 5, new_client)
    coordinator = make_coordinator(FakeClient())
    delays = []
    real_sleep = asyncio.sleep

    async def record_sleep(delay: float) -> None:
        delays.append(delay)
        await real_sleep(0)

    with patch.object(coordinator_module.asyncio, "sleep", record_sleep):
        await coordinator.reconnect()
        while coordinator.client is not new_client:
            await real_sleep(0.001)

    # The watchdog timer sleeps through the same function, for much longer.
    assert [delay for delay in delays if delay < 1] == [coordinator_module.CONNECT_RETRY_DELAY] * 5


async def test_watchdog_tick_does_not_restart_a_running_reconnect(script_create, make_coordinator) -> None:
    """Concurrent triggers must share one reconnect (bug 1: the tick cancelled it)."""

    create = script_create(LOST)
    coordinator = make_coordinator(FakeClient())

    await coordinator.reconnect()
    await wait_until(lambda: create.call_count == 1)
    task = coordinator._reconnect_task

    await coordinator.reconnect()
    await coordinator.reconnect()

    assert coordinator._reconnect_task is task
    assert not task.cancelled()


async def test_outage_logs_one_warning(script_create, make_coordinator, caplog) -> None:
    """Many failed attempts in one outage produce one warning."""

    new_client = FakeClient()
    script_create(LOST, LOST, LOST, new_client)
    coordinator = make_coordinator(FakeClient())

    with caplog.at_level(logging.WARNING, logger=coordinator_module.__name__):
        await coordinator.reconnect()
        await wait_until(lambda: coordinator.client is new_client)

    assert len(warnings_from_coordinator(caplog)) == 1


async def test_next_outage_warns_again(script_create, make_coordinator, caplog) -> None:
    """The warning is once per outage, not once per lifetime."""

    first = FakeClient()
    second = FakeClient()
    script_create(LOST, first, LOST, second)
    coordinator = make_coordinator(FakeClient())
    coordinator.async_add_listener(MagicMock())

    with caplog.at_level(logging.WARNING, logger=coordinator_module.__name__):
        await coordinator.reconnect()
        await wait_until(lambda: coordinator.client is first)
        first.push(STATUS)
        await wait_until(lambda: coordinator.status is STATUS)

        await coordinator.reconnect()
        await wait_until(lambda: coordinator.client is second)

    assert len(warnings_from_coordinator(caplog)) == 2


async def test_old_client_is_closed_exactly_once(script_create, make_coordinator) -> None:
    """Every retry must leave the old client alone once it is closed."""

    old_client = FakeClient()
    new_client = FakeClient()
    script_create(LOST, new_client)
    coordinator = make_coordinator(old_client)

    await coordinator.reconnect()
    await wait_until(lambda: coordinator.client is new_client)

    assert old_client.shutdown_calls == 1
