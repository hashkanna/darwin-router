"""Immutable holdout replay. Snapshots the index, runs the hand-written
holdout through the live router's /route endpoint (decision only — eval
traffic can never enter the index), appends an accuracy point to
data/replay.jsonl. Cost/latency for the same point come from the live log.

Run once:      python -m router.replay
Every 30 min:  python -m router.replay --loop
"""

import json
import statistics
import sys
import time
from datetime import datetime, timezone

import httpx

from router import config
from router.index import get_index

BASE = "http://127.0.0.1:8787"
INTERVAL_S = 1800


def load_holdout() -> list[dict]:
    with open(config.HOLDOUT_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]


def live_traffic_stats() -> tuple[float | None, float | None]:
    """(avg cost/request, p50 latency ms) over all live (non-eval) traffic."""
    if not config.LOG_PATH.exists():
        return None, None
    with open(config.LOG_PATH) as f:
        live = [json.loads(l) for l in f if l.strip()]
    live = [e for e in live if not e.get("eval")]
    if not live:
        return None, None
    return (
        sum(e["cost"] for e in live) / len(live),
        statistics.median(e["latency_ms"] for e in live),
    )


def run_once() -> dict:
    holdout = load_holdout()
    snapshot = get_index().snapshot() if config.EXEMPLARS_PATH.exists() else None

    correct, by_route = 0, {}
    with httpx.Client(timeout=60) as client:
        for item in holdout:
            r = client.post(f"{BASE}/route", json={"text": item["text"]},
                            headers={"x-darwin-eval": "1"})
            r.raise_for_status()
            pred = r.json()["route"]
            hit = pred == item["label"]
            correct += hit
            key = item["label"]
            by_route.setdefault(key, {"n": 0, "correct": 0})
            by_route[key]["n"] += 1
            by_route[key]["correct"] += hit

    avg_cost, p50_latency = live_traffic_stats()
    with httpx.Client(timeout=10) as client:
        h = client.get(f"{BASE}/healthz").json()
    point = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "accuracy": round(correct / len(holdout), 4),
        "n": len(holdout),
        "by_route": by_route,
        "exemplars": h["exemplars"],
        "avg_cost_per_request": avg_cost,
        "p50_latency_ms": p50_latency,
        "snapshot": snapshot,
    }
    with open(config.REPLAY_LOG_PATH, "a") as f:
        f.write(json.dumps(point) + "\n")
    print(f"[replay] accuracy={point['accuracy']} ({correct}/{len(holdout)}) "
          f"exemplars={h['exemplars']} avg_cost={avg_cost} p50={p50_latency}ms")
    return point


def main():
    loop = "--loop" in sys.argv
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[replay] error: {e}", file=sys.stderr)
            if not loop:
                sys.exit(1)
        if not loop:
            break
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
