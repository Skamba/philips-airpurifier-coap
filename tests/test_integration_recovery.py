"""Recovery tests through a real Home Assistant, on the AC4236 entity classes.

These cover what the coordinator tests cannot: config entry state on reload, and
the entity IDs and service calls that Node-RED depends on.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from unittest.mock import patch

import pytest
from aioairctrl import CoAPClient
from homeassistant.components.fan import ATTR_PERCENTAGE, SERVICE_SET_PERCENTAGE
from homeassistant.components.fan import DOMAIN as FAN_DOMAIN
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import (
    ATTR_ENTITY_ID,
    CONF_HOST,
    CONF_NAME,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_UNAVAILABLE,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.philips_airpurifier_coap.const import (
    CONF_DEVICE_ID,
    CONF_MODEL,
    CONF_STATUS,
    DOMAIN,
    PhilipsApi,
)
from custom_components.philips_airpurifier_coap.coordinator import Coordinator

from .fake_coap import FakeClient, wait_until

FAN = "fan.living_room"
PM25 = "sensor.living_room_pm2_5"
PREFERRED_INDEX = "select.living_room_preferred_index"

STATUS = {
    PhilipsApi.NAME: "Living Room",
    PhilipsApi.MODEL_ID: "AC4236/14",
    PhilipsApi.DEVICE_ID: "0123456789abcdef",
    PhilipsApi.WIFI_VERSION: "AWS_Philips_AIR@79",
    PhilipsApi.SOFTWARE_VERSION: "Ms4418",
    PhilipsApi.POWER: "1",
    PhilipsApi.MODE: "M",
    PhilipsApi.SPEED: "1",
    PhilipsApi.PM25: 7,
    PhilipsApi.CHILD_LOCK: False,
    PhilipsApi.DISPLAY_BACKLIGHT: "1",
    PhilipsApi.LIGHT_BRIGHTNESS: 100,
    "ddp": "1",
}

EXPECTED_ENTITIES = [
    ("AC4236-0123456789abcdef", "fan.living_room"),
    ("AC4236-0123456789abcdef-aqil", "light.living_room_light_brightness"),
    ("AC4236-0123456789abcdef-cl", "switch.living_room_child_lock"),
    ("AC4236-0123456789abcdef-ddp#1", "select.living_room_preferred_index"),
    ("AC4236-0123456789abcdef-pm25", "sensor.living_room_pm2_5"),
    ("AC4236-0123456789abcdef-uil", "light.living_room_display_backlight"),
]

LOST = object()
"""Scripted result for CoAPClient.create(): the sync reply never arrives."""


@pytest.fixture
def script_create() -> AsyncIterator[Callable[..., None]]:
    """Script what each CoAPClient.create() call does."""
    remaining: list[object] = []

    async def create(_host: str) -> object:
        result = remaining.pop(0) if remaining else FakeClient()
        if result is LOST:
            await asyncio.Event().wait()
        return result

    with patch.object(CoAPClient, "create", side_effect=create):
        yield remaining.extend


@pytest.fixture
async def entry(hass: HomeAssistant, script_create) -> AsyncIterator[MockConfigEntry]:
    """Set up one AC4236, as it would be after being added through the UI."""
    hass.config.components.update({"http", "frontend"})
    client = FakeClient()
    script_create([client])
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id=STATUS[PhilipsApi.DEVICE_ID],
        data={
            CONF_HOST: "192.0.2.1",
            CONF_MODEL: "AC4236",
            CONF_NAME: "Living Room",
            CONF_DEVICE_ID: STATUS[PhilipsApi.DEVICE_ID],
            CONF_STATUS: dict(STATUS),
        },
    )
    config_entry.add_to_hass(hass)

    with (
        patch("custom_components.philips_airpurifier_coap.async_setup", return_value=True),
        patch("custom_components.philips_airpurifier_coap.async_get_mac_address_from_host", return_value=None),
    ):
        assert await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
        yield config_entry
        if config_entry.state is ConfigEntryState.LOADED:
            assert await hass.config_entries.async_unload(config_entry.entry_id)


def coordinator_of(hass: HomeAssistant, entry: MockConfigEntry) -> Coordinator:
    """Return the coordinator of a config entry."""
    return hass.data[DOMAIN][entry.entry_id].coordinator


async def start_outage(hass: HomeAssistant, entry: MockConfigEntry) -> FakeClient:
    """Put the integration in a reconnect that is not getting any answers."""
    coordinator = coordinator_of(hass, entry)
    old_client = coordinator.client
    await coordinator.reconnect()
    await wait_until(lambda: coordinator.client is None)
    return old_client


async def test_node_red_entities_exist(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """The entity IDs that the Node-RED flows use must be there."""

    assert hass.states.get(FAN).state == "on"
    assert hass.states.get(PM25).state == "7"
    assert hass.states.get(PREFERRED_INDEX) is not None


@pytest.mark.parametrize(
    ("service", "data", "expected"),
    [
        (SERVICE_TURN_OFF, {}, {PhilipsApi.POWER: "0"}),
        (SERVICE_TURN_ON, {}, {PhilipsApi.POWER: "1"}),
        (
            SERVICE_SET_PERCENTAGE,
            {ATTR_PERCENTAGE: 100},
            {PhilipsApi.POWER: "1", PhilipsApi.MODE: "T", PhilipsApi.SPEED: "t"},
        ),
    ],
)
async def test_node_red_services_reach_the_device(
    hass: HomeAssistant, entry: MockConfigEntry, service: str, data: dict, expected: dict
) -> None:
    """The services that Node-RED calls must send the same data as before."""

    client = coordinator_of(hass, entry).client

    await hass.services.async_call(FAN_DOMAIN, service, {ATTR_ENTITY_ID: FAN, **data}, blocking=True)

    assert client.commands == [expected]


async def test_service_during_reconnect_raises_a_clear_error(
    hass: HomeAssistant, entry: MockConfigEntry, script_create
) -> None:
    """Node-RED must get an error it can act on, and not a crash or a hang."""

    script_create([LOST])
    await start_outage(hass, entry)

    with pytest.raises(HomeAssistantError, match="reconnecting"):
        await hass.services.async_call(FAN_DOMAIN, SERVICE_TURN_OFF, {ATTR_ENTITY_ID: FAN}, blocking=True)


async def test_failed_service_call_does_not_change_the_state(
    hass: HomeAssistant, entry: MockConfigEntry, script_create
) -> None:
    """A command that failed must not show up as if it worked."""

    script_create([LOST])
    await start_outage(hass, entry)

    with pytest.raises(HomeAssistantError):
        await hass.services.async_call(FAN_DOMAIN, SERVICE_TURN_OFF, {ATTR_ENTITY_ID: FAN}, blocking=True)
    await hass.async_block_till_done()

    assert hass.states.get(FAN).state == "on"


async def test_entities_stay_available_during_an_outage(
    hass: HomeAssistant, entry: MockConfigEntry, script_create
) -> None:
    """Availability must stay as it is in v0.37 (the fork went unavailable here)."""

    script_create([LOST])
    await start_outage(hass, entry)
    await hass.async_block_till_done()

    assert hass.states.get(FAN).state != STATE_UNAVAILABLE
    assert hass.states.get(PM25).state != STATE_UNAVAILABLE


async def test_status_after_a_reconnect_reaches_the_entities(
    hass: HomeAssistant, entry: MockConfigEntry, script_create
) -> None:
    """The whole path: outage, reconnect, new observation, new state."""

    coordinator = coordinator_of(hass, entry)
    new_client = FakeClient()
    script_create([new_client])

    await coordinator.reconnect()
    await wait_until(lambda: coordinator.client is new_client and new_client.observe_calls == 1)
    new_client.push({**STATUS, PhilipsApi.PM25: 42})
    await wait_until(lambda: coordinator.status[PhilipsApi.PM25] == 42)
    await hass.async_block_till_done()

    assert hass.states.get(PM25).state == "42"


async def test_reload_during_an_outage_recovers(hass: HomeAssistant, entry: MockConfigEntry, script_create) -> None:
    """Bug 3: this used to end in FAILED_UNLOAD, which only a restart clears."""

    script_create([LOST])
    old_client = await start_outage(hass, entry)
    new_client = FakeClient()
    script_create([new_client])

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    assert coordinator_of(hass, entry).client is new_client
    assert old_client.shutdown_calls == 1
    await wait_until(lambda: new_client.observe_calls == 1)


async def test_unload_with_an_already_closed_client_succeeds(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Bug 3, directly: a client closed behind our back must not fail the unload."""

    await coordinator_of(hass, entry).client.shutdown()

    assert await hass.config_entries.async_unload(entry.entry_id)

    assert entry.state is ConfigEntryState.NOT_LOADED


async def test_unique_ids_and_entity_ids_match_v037(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Changing a unique ID would create new entities and break the Node-RED flows.

    EXPECTED_ENTITIES was captured by running this test against an unmodified
    v0.37 checkout.
    """

    entries = er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)

    assert sorted((e.unique_id, e.entity_id) for e in entries) == EXPECTED_ENTITIES
