"""Integration tests for NST Pro Studio ASGI server endpoints."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.server import api_presets, api_status, index_view


def test_server_status_endpoint() -> None:
    """Verify that api_status returns 200 with online hardware info."""
    response = asyncio.run(api_status(None))  # type: ignore[arg-type]
    assert response.status_code == 200
    data = json.loads(response.body.decode("utf-8"))
    assert data.get("status") == "online"
    assert "device_name" in data
    assert "cuda_available" in data


def test_server_presets_endpoint() -> None:
    """Verify that api_presets returns all exhibition preset cartridges."""
    response = asyncio.run(api_presets(None))  # type: ignore[arg-type]
    assert response.status_code == 200
    data = json.loads(response.body.decode("utf-8"))
    assert "presets" in data
    presets = data["presets"]
    assert len(presets) >= 3
    assert "picasso_ballerina" in presets
    assert "monet_bridge" in presets
    assert "starry_ballerina" in presets
    assert presets["picasso_ballerina"]["artist"] == "Pablo Picasso"


def test_server_index_view_file_exists() -> None:
    """Verify that index_view serves valid index.html."""
    response = asyncio.run(index_view(None))  # type: ignore[arg-type]
    assert response.status_code == 200
    assert Path(response.path).exists()
    assert Path(response.path).name == "index.html"
