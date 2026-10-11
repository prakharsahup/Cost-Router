"""Lightweight HTTP REST API for CostRouter.

Provides endpoints for JMeter load testing:
- GET  /health     : Health check
- GET  /api/stats  : Budget totals
- POST /api/route  : Evaluates prompt difficulty, routes to cheap/expensive model,
                     updates budget ledger, and returns full routing metadata.

Run with ``python -m costrouter.api_server [port]``.  Models are mocks — this
server exists to load-test the routing/budget machinery, not real LLM latency.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict

from costrouter.types import (
    BudgetMode,
    ModelResponse,
    ModelTier,
    RouterConfig,
)
from costrouter.budget import BudgetTracker
from costrouter.difficulty import FailureRateStore, HeuristicDifficultyEstimator
from costrouter.router import RouterNode


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CostRouterAPI")

MAX_BODY_BYTES = 1_000_000


class MockCheapModel:
    """Fast, low-cost model simulation."""
    def invoke(self, prompt: str, **kwargs) -> ModelResponse:
        time.sleep(0.002)
        return ModelResponse(
            text=f"[cheap] Response to: {prompt[:40]}...",
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=25,
            cost_usd=0.00005,
        )


class MockExpensiveModel:
    """Capable, higher-cost model simulation."""
    def invoke(self, prompt: str, **kwargs) -> ModelResponse:
        time.sleep(0.006)
        return ModelResponse(
            text=f"[expensive] Detailed analysis of: {prompt[:40]}...",
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=150,
            cost_usd=0.0025,
        )


RATE_CARD = {
    ModelTier.CHEAP: {"prompt": 0.000001, "completion": 0.000002},
    ModelTier.EXPENSIVE: {"prompt": 0.00001, "completion": 0.00003},
}

GLOBAL_BUDGET = BudgetTracker(
    initial_budget_usd=1000.0,
    rate_card=RATE_CARD,
    mode=BudgetMode.SOFT,
)
STORE = FailureRateStore(prior=0.5, min_samples=5)
ESTIMATOR = HeuristicDifficultyEstimator(failure_store=STORE)
ROUTER_CONFIG = RouterConfig(threshold=0.5, retry_on_error=True)

GLOBAL_ROUTER = RouterNode(
    cheap_model=MockCheapModel(),
    expensive_model=MockExpensiveModel(),
    estimator=ESTIMATOR,
    budget=GLOBAL_BUDGET,
    config=ROUTER_CONFIG,
)


class CostRouterRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for CostRouter."""

    def _send_json(self, status_code: int, data: Dict[str, Any]):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/health", "/", "/health/"):
            response = {
                "status": "healthy",
                "service": "CostRouter",
                "version": "1.0.0",
                "remaining_budget": GLOBAL_BUDGET.remaining_budget(),
            }
            self._send_json(200, response)
        elif self.path == "/api/stats":
            rem = GLOBAL_BUDGET.remaining_budget()
            response = {
                "initial_budget": GLOBAL_BUDGET.initial_budget,
                "remaining_budget": rem,
                "total_spend": GLOBAL_BUDGET.initial_budget - rem,
                "threshold": ROUTER_CONFIG.threshold,
            }
            self._send_json(200, response)
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        if self.path in ("/api/route", "/api/route/"):
            try:
                content_length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                self._send_json(400, {"error": "Invalid Content-Length"})
                return
            if content_length < 0 or content_length > MAX_BODY_BYTES:
                self._send_json(413, {"error": f"Body must be 0..{MAX_BODY_BYTES} bytes"})
                return
            post_body = self.rfile.read(content_length)

            try:
                data = json.loads(post_body.decode("utf-8")) if post_body else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(400, {"error": "Invalid JSON"})
                return
            if not isinstance(data, dict):
                self._send_json(400, {"error": "JSON body must be an object"})
                return

            prompt = data.get("prompt", "Default test question")
            step_type = data.get("step_type", "default")
            step_id = data.get("step_id", f"req_{int(time.time()*1000)}")
            for name, value in (("prompt", prompt), ("step_type", step_type), ("step_id", step_id)):
                if not isinstance(value, str) or not value:
                    self._send_json(400, {"error": f"'{name}' must be a non-empty string"})
                    return

            t_start = time.perf_counter()
            state = {
                "step_id": step_id,
                "prompt": prompt,
                "step_type": step_type,
            }

            try:
                result = GLOBAL_ROUTER(state)
                t_total_ms = (time.perf_counter() - t_start) * 1000
                decision = result["last_decision"]

                response = {
                    "status": "success",
                    "step_id": step_id,
                    "model_chosen": decision.model_chosen.value,
                    "difficulty_score": round(decision.difficulty_score, 4),
                    "budget_forced": decision.budget_forced,
                    "retried": decision.retried,
                    "cost_usd": decision.actual_cost_usd,
                    "budget_remaining": round(decision.budget_remaining_after, 6),
                    "latency_ms": round(t_total_ms, 2),
                    "response": result["response"],
                }
                self._send_json(200, response)
            except Exception as e:
                logger.exception("Error handling route")
                self._send_json(500, {"error": str(e)})
        else:
            self._send_json(404, {"error": "Not Found"})

    def log_message(self, format, *args):
        # Suppress spamming console during load testing
        return


def run_server(port: int = 8000):
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, CostRouterRequestHandler)
    httpd.request_queue_size = 128
    logger.info(f"CostRouter API Server (threaded) running on http://127.0.0.1:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping CostRouter API Server...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run_server(port)
