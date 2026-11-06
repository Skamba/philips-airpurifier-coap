"""Module containing the Coordinator class to manage data requests from the Philips API."""

import asyncio
import contextlib
import logging
import time
from asyncio.tasks import Task
from collections.abc import Callable
from typing import Any

from aioairctrl import CoAPClient
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError

from .timer import Timer

_LOGGER = logging.getLogger(__name__)

MISSED_PACKAGE_COUNT = 3

# Limit on each teardown step, so that unload and reconnect can never hang on a
# client or task that does not finish. aiocoap's own shutdown takes up to 3 s.
TEARDOWN_TIMEOUT = 5

# The sync is a NON request that is never retransmitted, so a lost reply has to
# be detected here. Replies normally take under 2.5 s, even when the host needs
# all three ARP broadcasts (#396), and a lost one never arrives late.
CONNECT_TIMEOUT = 8

# Deliberately a flat delay and not a backoff: a long outage must not be followed
# by a long wait to notice that it is over. Each retry is one small datagram, and
# also a fresh ARP lookup.
CONNECT_RETRY_DELAY = 5

# How long to wait for the first status of an observation before asking again.
# This must stay well above the 80 s that an AC4236 on firmware @79 took not to
# answer a new observe GET, because asking too soon only abandons a reply that
# was still on its way. A lost GET costs this long; the watchdog still replaces
# the client at 180 s, as it always did.
FIRST_STATUS_TIMEOUT = 90

# Pause before observing again, so that a persistent failure cannot spin.
OBSERVE_RETRY_DELAY = 5

# Commands are NON requests too, and are normally answered in about 0.25 s. This
# leaves room for the three ARP broadcasts of a failed lookup, and for aioairctrl
# to re-sync and resend after a "failed" reply.
COMMAND_TIMEOUT = 8

# How long a failed command waits for the reconnect before giving up: two sync
# attempts. The reconnect itself carries on regardless.
RECONNECT_WAIT = 2 * CONNECT_TIMEOUT + CONNECT_RETRY_DELAY


class Coordinator:
    """Class to coordinate the data requests from the Philips API."""

    def __init__(self, hass: HomeAssistant, client: CoAPClient, host: str, status: dict[str, Any] | None) -> None:
        """Initialize the Coordinator.

        :param hass: HomeAssistant instance.
        :param client: CoAPClient instance.
        :param host: Host address of the device.
        :param status: Initial status of the device.
        """

        self.hass = hass
        self.client = client
        self.host = host
        self.status = status

        self._listeners: list[CALLBACK_TYPE] = []
        self._task: Task | None = None

        self._reconnect_task: Task | None = None
        self._timeout: int = 60
        self._closed = False
        self._outage = False
        self._last_status: float | None = None

        self._timer_disconnected = Timer(
            timeout=self._timeout * MISSED_PACKAGE_COUNT,
            callback=self.reconnect,
            autostart=True,
        )
        self._timer_disconnected.setAutoRestart(True)

    async def shutdown(self):
        """Shutdown the API connection. Safe to repeat, and never raises."""

        self._closed = True

        if self._timer_disconnected is not None:
            # Force the cancel: an unguarded Timer.cancel() raises
            # CallbackRunningException if it lands while the timeout callback
            # is running - which is exactly the case when we unload during a
            # reconnect - and the auto-restarting timer would otherwise keep
            # running after the entry is gone.
            self._timer_disconnected.setAutoRestart(False)
            self._timer_disconnected.cancel(msg="SHUTDOWN", force=True)

        # Stop the observer before closing the client it is running on (PR #387).
        await self._stop_tasks(self._task, self._reconnect_task)
        self._task = None
        self._reconnect_task = None

        await self._close_client()

    async def _stop_tasks(self, *tasks: Task | None) -> None:
        """Cancel tasks and wait for them to finish, within a limit."""

        running = [task for task in tasks if task is not None]
        for task in running:
            task.cancel()
        if running:
            await asyncio.wait(running, timeout=TEARDOWN_TIMEOUT)

    async def _close_client(self) -> None:
        """Forget the client, then close it.

        Dropping the reference first means self.client can never point at a
        closed client, whatever happens next. Closing an aiocoap context a
        second time raises AttributeError, and anything raised here would leave
        the config entry in FAILED_UNLOAD until Home Assistant is restarted.
        """

        client, self.client = self.client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(client.shutdown(), TEARDOWN_TIMEOUT)

    async def reconnect(self):
        """Reconnect to the API connection, unless that is already happening."""

        if self._closed or (self._reconnect_task is not None and not self._reconnect_task.done()):
            return

        self._reconnect_task = asyncio.create_task(self._reconnect())

    async def _reconnect(self):
        """Replace the client, retrying until the device answers a sync.

        Only one of these runs at a time, so a step that never finished would
        end all recovery. Every await in here has a time limit.
        """

        await self._stop_tasks(self._task)
        self._task = None
        await self._close_client()

        while True:
            try:
                self.client = await asyncio.wait_for(CoAPClient.create(self.host), CONNECT_TIMEOUT)
                break
            except Exception as ex:
                self._warn_once("Lost connection to %s, retrying until it answers (%r)", self.host, ex)
            await asyncio.sleep(CONNECT_RETRY_DELAY)

        self._start_observing()

    def _warn_once(self, msg: str, *args: Any) -> None:
        """Log a warning, once per outage. A real status ends the outage."""

        if not self._outage:
            self._outage = True
            _LOGGER.warning(msg, *args)

    async def set_control_value(self, key: str, value: Any) -> None:
        """Send one control value to the device."""

        await self.set_control_values(data={key: value})

    async def set_control_values(self, data: dict[str, Any]) -> None:
        """Send control values to the device, reconnecting and retrying once on failure.

        Control writes set absolute values, so sending one twice is harmless.
        """

        client = self._connected_client()
        try:
            await self._send_or_resend(client, data)
        except Exception as ex:
            _LOGGER.warning("Command to %s failed, reconnecting and retrying once (%r)", self.host, ex)
        else:
            return

        if self.client is client:
            await self.reconnect()
        if self._reconnect_task is not None:
            # asyncio.wait() leaves the task running on a timeout. Other commands
            # may be waiting for it, and it has to continue either way.
            await asyncio.wait([self._reconnect_task], timeout=RECONNECT_WAIT)

        try:
            await self._send(self._connected_client(), data)
        except HomeAssistantError:
            raise
        except Exception as ex:
            msg = f"Command to {self.host} failed: {ex!r}"
            raise HomeAssistantError(msg) from ex

    async def _send_or_resend(self, client: CoAPClient, data: dict[str, Any]) -> None:
        """Send control values, and once more if the reply is lost on a healthy connection.

        A reconnect has a price: a new client has a new source port, and the
        device may keep pushing to the old one for minutes. One lost datagram is
        not worth that while status is still arriving (#363).
        """

        try:
            await self._send(client, data)
        except TimeoutError:
            if self.client is not client or not self._status_is_recent():
                raise
            _LOGGER.info("Command to %s got no reply, sending it again", self.host)
            await self._send(client, data)

    def _status_is_recent(self) -> bool:
        """Return whether a status arrived within the Max-Age of the device."""

        return self._last_status is not None and time.monotonic() - self._last_status < self._timeout

    async def _send(self, client: CoAPClient, data: dict[str, Any]) -> None:
        """Send control values on a client, within a time limit."""

        async with asyncio.timeout(COMMAND_TIMEOUT):
            # aioairctrl returns False when the device keeps refusing a command.
            if not await client.set_control_values(data=data):
                msg = f"Command to {self.host} was rejected by the device"
                raise HomeAssistantError(msg)

    def _connected_client(self) -> CoAPClient:
        """Return the current client, or explain why there is none."""

        if self.client is None:
            msg = f"Cannot reach {self.host}: reconnecting, try again shortly"
            raise HomeAssistantError(msg)
        return self.client

    async def async_first_refresh(self) -> None:
        """Refresh the data for the first time."""

        try:
            self.status, timeout = await self.client.get_status()
            self._timeout = timeout
            if self._timer_disconnected is not None:
                self._timer_disconnected.setTimeout(timeout * MISSED_PACKAGE_COUNT)
            _LOGGER.debug("finished first refresh for host %s", self.host)

        except Exception as ex:
            _LOGGER.exception("Config not ready, first refresh failed for host %s", self.host)
            raise ConfigEntryNotReady from ex

    @callback
    def async_add_listener(self, update_callback: CALLBACK_TYPE) -> Callable[[], None]:
        """Listen for data updates."""

        start_observing = not self._listeners
        self._listeners.append(update_callback)

        if start_observing:
            self._start_observing()

        @callback
        def remove_listener() -> None:
            """Remove update listener."""
            self.async_remove_listener(update_callback)

        return remove_listener

    @callback
    def async_remove_listener(self, update_callback) -> None:
        """Remove data update."""

        self._listeners.remove(update_callback)

        if not self._listeners and self._task:
            # Keep the reference: shutdown() has to wait for this task to finish.
            self._task.cancel()

    async def _async_observe_status(self) -> None:
        """Keep the status of the device under observation, until cancelled."""

        while True:
            try:
                await self._observe_once()
                self._warn_once("Observation of %s ended, observing again", self.host)
            except TimeoutError:
                self._warn_once("No status from %s yet, asking again. It may just be idle", self.host)
            except Exception as ex:
                # For example a corrupt notification, which kills the observation
                # in aioairctrl (aioairctrl#14). The client itself is still fine.
                self._warn_once("Observation of %s failed, observing again (%r)", self.host, ex)
            await asyncio.sleep(OBSERVE_RETRY_DELAY)

    async def _observe_once(self) -> None:
        """Run one observation, on the current client.

        A sync proves that the device answers, not that it will notify us, so
        only a real status counts as having a working connection again. This
        deliberately neither resets the watchdog nor replaces the client: a new
        client has a new source port, and the device may keep pushing to the old
        one for minutes. The watchdog replaces the client if this never works.
        """

        async with asyncio.timeout(FIRST_STATUS_TIMEOUT) as first_status:
            async with contextlib.aclosing(self.client.observe_status()) as statuses:
                async for status in statuses:
                    first_status.reschedule(None)
                    self.status = status
                    self._last_status = time.monotonic()
                    self._timer_disconnected.reset()
                    if self._outage:
                        self._outage = False
                        _LOGGER.info("Receiving status from %s again", self.host)
                    for update_callback in self._listeners:
                        update_callback()

    def _start_observing(self) -> None:
        """Schedule state observation."""

        if self.client is None:
            # A reconnect is running, and it starts the observation when it is done.
            return

        if self._task:
            self._task.cancel()
            self._task = None

        self._task = asyncio.create_task(self._async_observe_status())
        self._timer_disconnected.reset()
