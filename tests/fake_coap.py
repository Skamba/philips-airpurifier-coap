"""Fake CoAP client for coordinator tests.

Models the aiocoap 0.4.17 behaviour that matters here: a request on a closed
context raises LibraryShutdown, closing a context twice raises AttributeError,
and a reply that was lost simply never arrives.
"""

import asyncio
from collections import deque
from typing import Any

from aiocoap.error import LibraryShutdown

HANG = object()
"""Queue this for a command, or leave the status queue empty, for a lost reply."""


class FakeClient:
    """Stand-in for aioairctrl's CoAP client."""

    def __init__(self) -> None:
        """Start with an open context and nothing queued."""
        self.closed = False
        self.shutdown_calls = 0
        self.observe_calls = 0
        self.commands: list[dict[str, Any]] = []
        self.command_results: deque[Any] = deque()
        self._notifications: asyncio.Queue[Any] = asyncio.Queue()

    def push(self, item: Any) -> None:
        """Deliver a status to the observer, or raise an Exception inside it."""
        self._notifications.put_nowait(item)

    def end_observation(self) -> None:
        """Make the running observation finish without an error."""
        self._notifications.put_nowait(StopAsyncIteration)

    async def shutdown(self) -> None:
        """Close the context, failing like aiocoap on a second call."""
        self.shutdown_calls += 1
        if self.closed:
            message = "'NoneType' object has no attribute 'values'"
            raise AttributeError(message)
        self.closed = True
        self._notifications.put_nowait(LibraryShutdown())

    async def observe_status(self):
        """Yield queued statuses until the observation ends or fails."""
        self.observe_calls += 1
        if self.closed:
            raise LibraryShutdown
        while True:
            item = await self._notifications.get()
            if item is StopAsyncIteration:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    async def set_control_values(self, data: dict[str, Any], retry_count: int = 5, resync: bool = True) -> bool:
        """Record a command and play back the next queued result."""
        if self.closed:
            raise LibraryShutdown
        self.commands.append(data)
        result = self.command_results.popleft() if self.command_results else True
        if result is HANG:
            await asyncio.Event().wait()
        if isinstance(result, Exception):
            raise result
        return result

    async def set_control_value(self, key: str, value: Any, retry_count: int = 5, resync: bool = True) -> bool:
        """Send one value, as aioairctrl does."""
        return await self.set_control_values({key: value}, retry_count, resync)


async def wait_until(predicate, timeout: float = 5) -> None:
    """Wait for a condition without assuming how long it takes."""
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.001)
