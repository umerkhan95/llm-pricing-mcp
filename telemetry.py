"""A log of what actually happened: prompts sent, replies received, tools run.

The INFO logs say a tool ran. They do not say what went into it, what came back,
which user query caused it, or how long it took. This writes one JSON object per
event to ``telemetry/events.jsonl`` so that can be answered afterwards.

No extra dependencies — it is the standard library and a file. Read it back with::

    python -m telemetry              # summary of the last run
    python -m telemetry --full       # every event, prompts and replies included

Nothing is ever written to stdout. The MCP servers speak JSON-RPC over stdio, so
printing there would corrupt the protocol stream.
"""

import functools
import inspect
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

LOG_DIR = Path(os.getenv("TELEMETRY_DIR", "telemetry"))
LOG_FILE = LOG_DIR / "events.jsonl"

# Tool results run to hundreds of KB. Keep enough to see what happened.
MAX_CHARS = int(os.getenv("TELEMETRY_MAX_CHARS", "2000"))

_lock = threading.Lock()
_service = "unknown"
_query = None


def setup(service_name: str) -> None:
    """Name the process that is writing events."""
    global _service
    _service = service_name


def set_query(query: str | None) -> None:
    """Tag every following event with the user query that caused it."""
    global _query
    _query = query


def clip(value: Any) -> Any:
    """Shorten a value for logging, keeping the shape recognisable."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= MAX_CHARS:
        return text
    return f"{text[:MAX_CHARS]}… [+{len(text) - MAX_CHARS} chars]"


def log(event: str, **fields: Any) -> None:
    """Append one event."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "service": _service,
        "event": event,
        "query": _query,
    }
    record.update(fields)
    with _lock:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def traced(event: str, record: tuple = ()) -> Callable:
    """Log a call's arguments, duration and outcome.

    Args:
        event: what to call it in the log, e.g. 'tool.scrape_websites'
        record: argument names to capture. Values are clipped, so passing a
            large payload is safe.

    Works on sync and async functions alike.
    """

    def decorate(fn: Callable) -> Callable:
        signature = inspect.signature(fn)

        def start(args, kwargs) -> dict:
            bound = signature.bind_partial(*args, **kwargs)
            return {k: clip(bound.arguments[k]) for k in record if k in bound.arguments}

        def done(fields: dict, result: Any, started: float, error: BaseException | None):
            fields["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
            if error is not None:
                fields["ok"] = False
                fields["error"] = f"{type(error).__name__}: {error}"
            else:
                fields["ok"] = True
                fields["result"] = clip(result)
            return fields

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                fields, started, error, result = start(args, kwargs), time.perf_counter(), None, None
                try:
                    result = await fn(*args, **kwargs)
                    return result
                except BaseException as e:
                    error = e
                    raise
                finally:
                    log(event, **done(fields, result, started, error))
            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            fields, started, error, result = start(args, kwargs), time.perf_counter(), None, None
            try:
                result = fn(*args, **kwargs)
                return result
            except BaseException as e:
                error = e
                raise
            finally:
                log(event, **done(fields, result, started, error))
        return wrapper

    return decorate


def log_llm_call(model: str, messages: list, tools: list | None, system: str | None,
                 response: Any, duration_ms: float) -> None:
    """Record one LLM round trip: what was sent, what came back."""
    last = messages[-1] if messages else {}
    blocks = [
        {"type": b.type,
         "text": clip(getattr(b, "text", None)) if b.type == "text" else None,
         "tool": getattr(b, "name", None),
         "input": clip(getattr(b, "input", None)) if b.type == "tool_use" else None}
        for b in getattr(response, "content", [])
    ]
    usage = getattr(response, "usage", None)
    log("llm.call",
        model=model,
        system_chars=len(system or ""),
        tools_offered=len(tools or []),
        message_count=len(messages),
        last_message_role=last.get("role"),
        last_message=clip(last.get("content")),
        stop_reason=getattr(response, "stop_reason", None),
        reply_blocks=blocks,
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        duration_ms=round(duration_ms, 1))


def _report(full: bool = False) -> None:
    """Print what the log says. Called via `python -m telemetry`."""
    if not LOG_FILE.exists():
        print(f"no events yet at {LOG_FILE}")
        return

    events = [json.loads(line) for line in LOG_FILE.read_text(encoding="utf-8").splitlines() if line]
    print(f"{len(events)} events in {LOG_FILE}\n")

    if full:
        for e in events:
            print(json.dumps(e, indent=2, ensure_ascii=False))
            print("-" * 70)
        return

    by_query: dict = {}
    for e in events:
        by_query.setdefault(e.get("query") or "(startup)", []).append(e)

    for query, group in by_query.items():
        tools = [e for e in group if e["event"].startswith("tool.")]
        llm = [e for e in group if e["event"] == "llm.call"]
        tokens_in = sum(e.get("input_tokens") or 0 for e in llm)
        tokens_out = sum(e.get("output_tokens") or 0 for e in llm)
        ms = sum(e.get("duration_ms") or 0 for e in group)
        print(f"  {query[:72]}")
        print(f"      {len(llm)} llm calls, {len(tools)} tool calls, "
              f"{tokens_in:,} in / {tokens_out:,} out tokens, {ms/1000:.1f}s")
        for t in tools:
            mark = "ok " if t.get("ok") else "ERR"
            print(f"        {mark} {t['event'][5:]:22} {t.get('duration_ms', 0):>8.0f}ms")
        print()


if __name__ == "__main__":
    import sys
    _report(full="--full" in sys.argv)
