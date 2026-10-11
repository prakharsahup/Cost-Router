"""HTTP-level tests for the load-testing API server's input validation."""

from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from costrouter.api_server import CostRouterRequestHandler


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), CostRouterRequestHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def _request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    data = json.loads(resp.read() or b"{}")
    conn.close()
    return resp.status, data


def test_route_success(server):
    status, data = _request(server, "POST", "/api/route", json.dumps({"prompt": "Who wrote Hamlet?"}))
    assert status == 200
    assert data["model_chosen"] in ("cheap", "expensive")


@pytest.mark.parametrize("body", [
    json.dumps({"prompt": 123}),
    json.dumps({"prompt": ""}),
    json.dumps(["not", "an", "object"]),
    "{not json",
])
def test_bad_input_is_400(server, body):
    status, data = _request(server, "POST", "/api/route", body)
    assert status == 400
    assert "error" in data


def test_oversized_body_is_413(server):
    status, _ = _request(server, "POST", "/api/route", "x", {"Content-Length": str(10**9)})
    assert status == 413


def test_stats_uses_public_budget_api(server):
    status, data = _request(server, "GET", "/api/stats")
    assert status == 200
    assert data["total_spend"] == pytest.approx(data["initial_budget"] - data["remaining_budget"])
