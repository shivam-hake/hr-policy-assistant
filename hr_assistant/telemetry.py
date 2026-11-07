"""Telemetry: every question and every LLM call, with tokens, latency and cost.

A LangChain callback handler is attached to each graph run. It sees every
model call inside the graph (classifier, answer, retries) and LangGraph tells
it which node made the call. Records go to a "sink" function:
- in the app: TelemetryStore.add_llm_call (SQLite)
- in the evaluation: a list, so cost per question lands in the eval report

Telemetry must never break answering: recording errors are logged and ignored.
"""
import logging
import sqlite3
import statistics
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from hr_assistant.pricing import cost_usd

log = logging.getLogger(__name__)
Sink = Callable[..., None]

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id         TEXT PRIMARY KEY,
    question   TEXT NOT NULL,
    status     TEXT NOT NULL,      -- answered | not_covered | declined | error
    category   TEXT,
    latency_ms INTEGER NOT NULL,
    error      TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id       TEXT NOT NULL,
    step             TEXT,
    model            TEXT,
    input_tokens     INTEGER NOT NULL DEFAULT 0,
    output_tokens    INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms       INTEGER NOT NULL,
    cost_usd         REAL,
    status           TEXT NOT NULL,
    error            TEXT,
    created_at       TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TelemetryCallback(BaseCallbackHandler):
    def __init__(self, sink: Sink, request_id: str):
        self.sink = sink
        self.request_id = request_id
        self._runs: dict[UUID, dict[str, Any]] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id: UUID, metadata=None, **kwargs) -> None:
        params = kwargs.get("invocation_params") or {}
        self._runs[run_id] = {"start": time.perf_counter(), "step": (metadata or {}).get("langgraph_node"),
                              "model": params.get("model") or params.get("model_name")}

    def on_llm_end(self, response, *, run_id: UUID, **kwargs) -> None:
        run = self._runs.pop(run_id, None)
        if run is None:
            return
        try:
            msg = response.generations[0][0].message
            usage = getattr(msg, "usage_metadata", None) or {}
            model = (getattr(msg, "response_metadata", None) or {}).get("model_name") or run["model"]
            i, o = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            self.sink(request_id=self.request_id, step=run["step"], model=model, input_tokens=i,
                      output_tokens=o, reasoning_tokens=(usage.get("output_token_details") or {}).get("reasoning", 0),
                      latency_ms=int((time.perf_counter() - run["start"]) * 1000),
                      cost_usd=cost_usd(model, i, o), status="ok")
        except Exception:
            log.exception("Failed to record LLM call")

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs) -> None:
        run = self._runs.pop(run_id, None)
        if run is None:
            return
        try:
            self.sink(request_id=self.request_id, step=run["step"], model=run["model"],
                      latency_ms=int((time.perf_counter() - run["start"]) * 1000), status="error",
                      error=f"{type(error).__name__}: {str(error)[:300]}")
        except Exception:
            log.exception("Failed to record LLM error")


class ListSink:
    """Collects calls in memory (used by the evaluation)."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **row) -> None:
        self.calls.append(row)

    def for_request(self, request_id: str) -> list[dict]:
        return [c for c in self.calls if c["request_id"] == request_id]


class TelemetryStore:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    def _exec(self, sql: str, params: tuple) -> None:
        with self._lock, self._conn:
            self._conn.execute(sql, params)

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def add_llm_call(self, **r) -> None:
        self._exec(
            "INSERT INTO llm_calls (request_id, step, model, input_tokens, output_tokens, reasoning_tokens, "
            "latency_ms, cost_usd, status, error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (r["request_id"], r.get("step"), r.get("model"), r.get("input_tokens", 0), r.get("output_tokens", 0),
             r.get("reasoning_tokens", 0), r["latency_ms"], r.get("cost_usd"), r["status"], r.get("error"), _now()),
        )

    def add_request(self, request_id: str, question: str, status: str, category: str | None,
                    latency_ms: int, error: str | None = None) -> None:
        self._exec(
            "INSERT INTO requests (id, question, status, category, latency_ms, error, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (request_id, question, status, category, latency_ms, error, _now()),
        )

    def requests(self, limit: int = 50) -> list[dict]:
        return self._rows("SELECT * FROM requests ORDER BY created_at DESC LIMIT ?", (limit,))

    def llm_calls(self) -> list[dict]:
        return self._rows("SELECT * FROM llm_calls ORDER BY id")

    def summary(self) -> dict:
        reqs = self._rows("SELECT * FROM requests")
        calls = self.llm_calls()
        costs = [c["cost_usd"] for c in calls if c["cost_usd"] is not None]
        lat = [r["latency_ms"] for r in reqs]
        by_status: dict[str, int] = defaultdict(int)
        for r in reqs:
            by_status[r["status"]] += 1
        return {
            "requests": len(reqs),
            "by_status": dict(by_status),
            "avg_latency_ms": int(statistics.mean(lat)) if lat else 0,
            "p95_latency_ms": p95(lat),
            "llm_calls": len(calls),
            "total_cost_usd": round(sum(costs), 6) if costs else None,
            "avg_cost_per_request_usd": round(sum(costs) / len(reqs), 6) if costs and reqs else None,
            "steps": step_stats(calls),
        }


def p95(values: list[int]) -> int:
    if len(values) < 2:
        return values[0] if values else 0
    return int(statistics.quantiles(values, n=20)[-1])


def step_stats(calls: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for c in calls:
        groups[c.get("step") or "unknown"].append(c)
    rows = []
    for step, cs in groups.items():
        lat = [c["latency_ms"] for c in cs]
        costs = [c["cost_usd"] for c in cs if c.get("cost_usd") is not None]
        rows.append({
            "step": step, "calls": len(cs), "errors": sum(c["status"] == "error" for c in cs),
            "avg_latency_ms": int(statistics.mean(lat)), "p95_latency_ms": p95(lat),
            "avg_input_tokens": int(statistics.mean(c.get("input_tokens", 0) for c in cs)),
            "avg_output_tokens": int(statistics.mean(c.get("output_tokens", 0) for c in cs)),
            "avg_reasoning_tokens": int(statistics.mean(c.get("reasoning_tokens", 0) for c in cs)),
            "total_cost_usd": round(sum(costs), 6) if costs else None,
        })
    return sorted(rows, key=lambda r: r["avg_latency_ms"], reverse=True)
