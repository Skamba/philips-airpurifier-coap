"""Tests for the Philips config flow."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.const import CONF_HOST
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers.service_info.dhcp import DhcpServiceInfo

from custom_components.philips_airpurifier_coap.config_flow import PhilipsAirPurifierConfigFlow

HOST = "192.0.2.1"


@pytest.mark.parametrize(
    ("data", "options"),
    [({CONF_HOST: HOST}, {}), ({CONF_HOST: "192.0.2.9"}, {CONF_HOST: HOST})],
    ids=["host-in-data", "host-in-options"],
)
async def test_dhcp_does_not_probe_a_configured_host(data: dict, options: dict) -> None:
    """A probe would take the observer away from the running coordinator (PR #389)."""

    flow = PhilipsAirPurifierConfigFlow()

    with (
        patch.object(flow, "_async_current_entries", return_value=[MagicMock(data=data, options=options)]),
        patch(
            "custom_components.philips_airpurifier_coap.config_flow.CoAPClient.create",
            new_callable=AsyncMock,
        ) as create_client,
    ):
        result = await flow.async_step_dhcp(DhcpServiceInfo(ip=HOST, hostname="purifier", macaddress="001122334455"))

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"
    create_client.assert_not_awaited()
