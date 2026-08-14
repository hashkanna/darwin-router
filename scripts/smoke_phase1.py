"""Phase 1 smoke test. Server must be running on :8787.
The imagegen case needs no API key (pre-route + stub); the other two need
DASHSCOPE_API_KEY and a bootstrapped index. Run: python scripts/smoke_phase1.py"""

import sys

import httpx

BASE = "http://127.0.0.1:8787"

CASES = [
    ("hey, how's it going?", "simple"),
    ("Prove that the sum of two odd integers is even, then write a Python check.", "reasoning"),
    ("please draw an image of a red panda on a bicycle", "imagegen"),
]


def main():
    failures = 0
    for query, expected in CASES:
        r = httpx.post(
            f"{BASE}/v1/chat/completions",
            json={"model": "darwin", "messages": [{"role": "user", "content": query}]},
            timeout=120,
        )
        if r.status_code != 200:
            print(f"FAIL [{expected}] HTTP {r.status_code}: {r.text[:200]}")
            failures += 1
            continue
        d = r.json().get("x_darwin", {})
        ok = d.get("route") == expected
        print(f"{'ok  ' if ok else 'FAIL'} [{expected}] -> route={d.get('route')} "
              f"margin={d.get('margin')} model={d.get('model')} {d.get('latency_ms')}ms")
        failures += 0 if ok else 1
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
