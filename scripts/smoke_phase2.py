"""Phase 2 smoke: send a low-margin query, wait for the async judge, verify a
judge event landed (and ideally an exemplar insert). Server must be running.
Run: python scripts/smoke_phase2.py"""

import json
import sys
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8787"
JUDGE_LOG = Path(__file__).resolve().parent.parent / "data" / "judge.jsonl"

QUERY = "Prove that the sum of two odd integers is always even, then write a Python check for it."


def judge_count() -> int:
    if not JUDGE_LOG.exists():
        return 0
    return sum(1 for l in open(JUDGE_LOG) if l.strip())


def main():
    before = judge_count()
    r = httpx.post(f"{BASE}/v1/chat/completions",
                   json={"model": "darwin", "messages": [{"role": "user", "content": QUERY}]},
                   timeout=180)
    r.raise_for_status()
    d = r.json()["x_darwin"]
    print(f"routed={d['route']} margin={d['margin']} model={d['model']}")

    print("waiting for async judge", end="", flush=True)
    for _ in range(36):  # up to 180s
        if judge_count() > before:
            break
        print(".", end="", flush=True)
        time.sleep(5)
    print()
    if judge_count() <= before:
        print("FAIL: no judge event appeared")
        sys.exit(1)
    event = json.loads(open(JUDGE_LOG).readlines()[-1])
    print("judge event:", json.dumps(event, indent=2))
    print("ok" if event.get("why") and not str(event["why"]).startswith("error") else "FAIL (judge errored)")
    sys.exit(0 if event.get("why") and not str(event["why"]).startswith("error") else 1)


if __name__ == "__main__":
    main()
