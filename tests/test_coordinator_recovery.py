"""Recovery tests for the coordinator: the integration must always heal itself.

Each scenario here is a failure seen on real devices. See upstream issues #363,
#393 and #396.
"""

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Callable
from itertools import pairwise
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aiocoap.error import LibraryShutdown, NetworkError
from homeassistant.exceptions import HomeAssistantError

from custom_components.philips_airpurifier_coap import coordinator as coordinator_module
from custom_components.philips_airpurifier_coap.coordinator import Coordinator

from .fake_coap import HANG, FakeClient, wait_until

HOST = "192.0.2.1"
STATUS = {"pwr": "1"}

LOST = object()
"""Scripted result for CoAPClient.create(): the sync reply never arrives."""


@pytest.fixture(autouse=True)
def short_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink every delay so that the tests run quickly."""
    monkeypatch.setattr(coordinator_module, "CONNECT_TIMEOUT", 0.02)
    monkeypatch.setattr(coordinator_module, "CONNECT_RETRY_DELAY", 0.001)
    monkeypatch.setattr(coordinator_module, "FIRST_STATUS_TIMEOUT", 0.01)
    monkeypatch.setattr(coordinator_module, "OBSERVE_RETRY_DELAY", 0.001)
    monkeypatch.setattr(coordinator_module, "COMMAND_TIMEOUT", 0.02)
    monkeypatch.setattr(coordinator_module, "RECONNECT_WAIT", 1)


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


async def test_command_during_reconnect_fails_clearly(script_create, make_coordinator) -> None:
    """A command with no client must raise HomeAssistantError, not AttributeError."""

    script_create(LOST)
    coordinator = make_coordinator(FakeClient())
    await coordinator.reconnect()
    await wait_until(lambda: coordinator.client is None)

    with pytest.raises(HomeAssistantError, match="reconnecting"):
        await coordinator.set_control_value("pwr", "1")
    with pytest.raises(HomeAssistantError, match="reconnecting"):
        await coordinator.set_control_values(data={"pwr": "1"})


async def test_client_is_never_a_closed_client(script_create, make_coordinator) -> None:
    """Bug 1: self.client stayed a closed client for the whole outage."""

    closed_clients = []
    old_client = FakeClient()
    script_create(LOST, LOST, FakeClient())
    coordinator = make_coordinator(old_client)

    await coordinator.reconnect()
    for _ in range(500):
        if coordinator.client is not None and coordinator.client.closed:
            closed_clients.append(coordinator.client)
        await asyncio.sleep(0)

    assert closed_clients == []


async def test_commands_reach_the_client(make_coordinator) -> None:
    """The wrappers send the same data the entities used to send directly."""

    client = FakeClient()
    coordinator = make_coordinator(client)

    await coordinator.set_control_value("pwr", "1")
    await coordinator.set_control_values(data={"mode": "S", "om": "1"})

    assert client.commands == [{"pwr": "1"}, {"mode": "S", "om": "1"}]


def test_no_entity_talks_to_the_client_directly() -> None:
    """Every command must go through the coordinator, or it has no protection."""

    component = Path(coordinator_module.__file__).parent
    direct = [
        path.name
        for path in component.glob("*.py")
        if path.name != "coordinator.py" and re.search(r"\.client\.set_control_value", path.read_text())
    ]

    assert direct == []


async def test_no_first_status_asks_again_on_the_same_client(script_create, make_coordinator) -> None:
    """A lost observe GET is retried without replacing the client or its socket.

    A new socket means a new source port, and a device that is still pushing to
    the old port may not answer the new one for minutes. So the retry must not
    add to the number of port changes.
    """

    create = script_create()
    client = FakeClient()
    coordinator = make_coordinator(client)

    coordinator.async_add_listener(MagicMock())
    await wait_until(lambda: client.observe_calls >= 3)

    assert coordinator.client is client
    assert client.shutdown_calls == 0
    create.assert_not_called()


async def test_status_after_a_retry_is_delivered(make_coordinator) -> None:
    """An observation that answers on a later attempt works normally."""

    client = FakeClient()
    coordinator = make_coordinator(client)
    listener = MagicMock()

    coordinator.async_add_listener(listener)
    await wait_until(lambda: client.observe_calls >= 2)
    fresh = {"pwr": "0"}
    client.push(fresh)

    await wait_until(lambda: coordinator.status is fresh)
    listener.assert_called()


async def test_quiet_device_is_not_timed_out_after_its_first_status(make_coordinator) -> None:
    """Only the first response has a deadline: later silence is for the watchdog."""

    client = FakeClient()
    coordinator = make_coordinator(client)
    coordinator.async_add_listener(MagicMock())
    client.push(STATUS)
    await wait_until(lambda: coordinator.status is STATUS)

    await asyncio.sleep(coordinator_module.FIRST_STATUS_TIMEOUT * 5)

    assert client.observe_calls == 1


async def test_successful_sync_does_not_end_the_outage(script_create, make_coordinator, caplog) -> None:
    """A sync proves the device answers, not that the observation is alive."""

    new_client = FakeClient()
    script_create(LOST, new_client)
    coordinator = make_coordinator(FakeClient())
    coordinator.async_add_listener(MagicMock())

    with caplog.at_level(logging.INFO, logger=coordinator_module.__name__):
        await coordinator.reconnect()
        await wait_until(lambda: coordinator.client is new_client)
        await wait_until(lambda: new_client.observe_calls >= 2)

        assert "Receiving status from" not in caplog.text

        new_client.push(STATUS)
        await wait_until(lambda: coordinator.status is STATUS)

    assert "Receiving status from" in caplog.text
    assert len(warnings_from_coordinator(caplog)) == 1


async def test_new_client_gets_a_full_watchdog_window(script_create, make_coordinator) -> None:
    """A reconnect just before the deadline must not be thrown away by the old deadline."""

    new_client = FakeClient()
    create = script_create(new_client)
    coordinator = make_coordinator(FakeClient())
    coordinator.async_add_listener(MagicMock())
    coordinator._timer_disconnected.setTimeout(0.6)

    await asyncio.sleep(0.4)
    await coordinator.reconnect()
    await wait_until(lambda: coordinator.client is new_client)
    await asyncio.sleep(0.4)

    assert create.call_count == 1
    assert coordinator.client is new_client


async def test_silent_device_warns_once(make_coordinator, caplog) -> None:
    """Asking again and again in one outage produces one warning."""

    client = FakeClient()
    coordinator = make_coordinator(client)

    with caplog.at_level(logging.WARNING, logger=coordinator_module.__name__):
        coordinator.async_add_listener(MagicMock())
        await wait_until(lambda: client.observe_calls >= 4)

    assert len(warnings_from_coordinator(caplog)) == 1


async def test_retries_do_not_hold_off_the_watchdog(script_create, make_coordinator) -> None:
    """Sync OK, observe never answers: the watchdog must still replace the client.

    If restarting the observation reset the watchdog, a device that never
    answers would never be reconnected to.
    """

    replacement = FakeClient()
    script_create(replacement)
    client = FakeClient()
    coordinator = make_coordinator(client)
    coordinator.async_add_listener(MagicMock())
    coordinator._timer_disconnected.setTimeout(0.1)

    await wait_until(lambda: coordinator.client is replacement)

    assert client.observe_calls > 1
    assert client.closed


async def test_outage_where_only_sync_works_recovers_on_a_fixed_cycle(script_create, make_coordinator) -> None:
    """The failure of the fork: syncs answer, observes do not, for several rounds.

    The device stays silent on three replacement clients, then answers. Recovery
    must be automatic, and the time between replacements must not grow.
    """

    silent = [FakeClient() for _ in range(3)]
    answering = FakeClient()
    create = script_create(*silent, answering)
    coordinator = make_coordinator(FakeClient())
    coordinator.async_add_listener(MagicMock())
    coordinator._timer_disconnected.setTimeout(0.05)
    replaced_at = []
    loop = asyncio.get_running_loop()

    async with asyncio.timeout(5):
        while coordinator.client is not answering:
            if len(replaced_at) < create.call_count:
                replaced_at.append(loop.time())
            await asyncio.sleep(0.001)

    fresh = {"pwr": "1", "pm25": 3}
    answering.push(fresh)
    await wait_until(lambda: coordinator.status is fresh)

    gaps = [later - earlier for earlier, later in pairwise(replaced_at)]
    assert create.call_count == 4
    assert max(gaps) < min(gaps) * 3


async def test_commands_work_while_waiting_for_the_first_status(make_coordinator) -> None:
    """Issue #388: devices can stay silent for 2 h while still accepting commands."""

    client = FakeClient()
    coordinator = make_coordinator(client)
    coordinator.async_add_listener(MagicMock())
    await wait_until(lambda: client.observe_calls >= 2)

    await coordinator.set_control_value("pwr", "1")

    assert client.commands == [{"pwr": "1"}]


async def test_first_listener_during_a_reconnect_waits_for_it(script_create, make_coordinator) -> None:
    """With no client yet, the reconnect starts the observation when it is done."""

    new_client = FakeClient()
    script_create(LOST, new_client)
    coordinator = make_coordinator(FakeClient())
    await coordinator.reconnect()
    await wait_until(lambda: coordinator.client is None)

    coordinator.async_add_listener(MagicMock())

    await wait_until(lambda: new_client.observe_calls >= 1)


async def test_command_with_no_reply_reconnects_and_retries(script_create, make_coordinator) -> None:
    """A lost command must not hang until a later reconnect kills it (bug 4, #363)."""

    client = FakeClient()
    client.command_results.append(HANG)
    new_client = FakeClient()
    create = script_create(new_client)
    coordinator = make_coordinator(client)

    await coordinator.set_control_value("pwr", "1")

    assert new_client.commands == [{"pwr": "1"}]
    assert create.call_count == 1


@pytest.mark.parametrize(
    "failure",
    [LibraryShutdown(), NetworkError("gone"), ValueError("bad key"), False],
    ids=["shutdown", "network", "bad-key", "rejected"],
)
async def test_failed_command_reconnects_and_retries(script_create, make_coordinator, failure) -> None:
    """Every way a command can fail is retried, including a False return.

    aioairctrl returns False when the device keeps replying "failed", and
    raises ValueError on a corrupt sync key (aioairctrl#14).
    """

    client = FakeClient()
    client.command_results.append(failure)
    new_client = FakeClient()
    script_create(new_client)
    coordinator = make_coordinator(client)

    await coordinator.set_control_value("pwr", "1")

    assert new_client.commands == [{"pwr": "1"}]


async def test_command_is_retried_only_once(script_create, make_coordinator) -> None:
    """A second failure is reported, and does not loop."""

    client = FakeClient()
    client.command_results.append(HANG)
    new_client = FakeClient()
    new_client.command_results.append(HANG)
    script_create(new_client)
    coordinator = make_coordinator(client)

    with pytest.raises(HomeAssistantError, match=re.escape(HOST)):
        await coordinator.set_control_value("pwr", "1")

    assert len(client.commands) == 1
    assert len(new_client.commands) == 1


async def test_rejected_retry_is_reported(script_create, make_coordinator) -> None:
    """A command the device keeps refusing must not look like it succeeded."""

    client = FakeClient()
    client.command_results.append(False)
    new_client = FakeClient()
    new_client.command_results.append(False)
    script_create(new_client)
    coordinator = make_coordinator(client)

    with pytest.raises(HomeAssistantError):
        await coordinator.set_control_value("pwr", "1")


async def test_concurrent_failures_share_one_reconnect(script_create, make_coordinator) -> None:
    """A burst of commands from Node-RED must not start a reconnect each."""

    client = FakeClient()
    client.command_results.extend([HANG, HANG, HANG])
    new_client = FakeClient()
    create = script_create(new_client)
    coordinator = make_coordinator(client)

    await asyncio.gather(
        coordinator.set_control_value("pwr", "1"),
        coordinator.set_control_value("mode", "S"),
        coordinator.set_control_value("uil", "0"),
    )

    assert create.call_count == 1
    assert len(new_client.commands) == 3


async def test_failure_on_an_already_replaced_client_does_not_reconnect_again(script_create, make_coordinator) -> None:
    """A command that was in flight when the client was replaced just retries."""

    client = FakeClient()
    release = asyncio.Event()

    async def blocked_until_replaced(data, retry_count=5, resync=True):
        await release.wait()
        raise LibraryShutdown

    client.set_control_values = blocked_until_replaced
    new_client = FakeClient()
    create = script_create(new_client)
    coordinator = make_coordinator(client)

    with patch.object(coordinator_module, "COMMAND_TIMEOUT", 5):
        command = asyncio.create_task(coordinator.set_control_value("pwr", "1"))
        await asyncio.sleep(0)
        await coordinator.reconnect()
        await wait_until(lambda: coordinator.client is new_client)
        release.set()
        await command

    assert create.call_count == 1
    assert new_client.commands == [{"pwr": "1"}]


async def test_command_gives_up_but_the_reconnect_carries_on(script_create, make_coordinator) -> None:
    """A device still down fails the command, and must not stop the recovery."""

    client = FakeClient()
    client.command_results.append(HANG)
    new_client = FakeClient()
    create = script_create(*[LOST] * 20, new_client)
    coordinator = make_coordinator(client)

    with (
        patch.object(coordinator_module, "RECONNECT_WAIT", 0.03),
        pytest.raises(HomeAssistantError, match="reconnecting"),
    ):
        await coordinator.set_control_value("pwr", "1")

    assert create.call_count < 20
    await wait_until(lambda: coordinator.client is new_client)


async def test_unload_during_a_failed_command_does_not_hang(script_create, make_coordinator) -> None:
    """A command waiting for a reconnect is released when the entry unloads."""

    client = FakeClient()
    client.command_results.append(HANG)
    script_create(*[LOST] * 1000)
    coordinator = make_coordinator(client)

    with patch.object(coordinator_module, "RECONNECT_WAIT", 30):
        command = asyncio.create_task(coordinator.set_control_value("pwr", "1"))
        await wait_until(lambda: coordinator.client is None)
        await coordinator.shutdown()

        with pytest.raises(HomeAssistantError):
            await command


async def test_cancelled_command_is_not_retried(script_create, make_coordinator) -> None:
    """Cancelling a service call must stop it, not trigger a reconnect."""

    client = FakeClient()
    client.command_results.append(HANG)
    create = script_create()
    coordinator = make_coordinator(client)

    with patch.object(coordinator_module, "COMMAND_TIMEOUT", 5):
        command = asyncio.create_task(coordinator.set_control_value("pwr", "1"))
        await wait_until(lambda: client.commands)
        command.cancel()
        with pytest.raises(asyncio.CancelledError):
            await command

    create.assert_not_called()
