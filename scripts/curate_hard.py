"""Operator curation: the judge cannot reliably grade symbol-manipulation and
stateful-simulation answers, so it occasionally inserts 'simple' exemplars for
queries the 4B genuinely fails. Purge judge-sourced 'simple' exemplars that
near-duplicate the traffic HARD pool (verified 4B-breaking, independent of the
holdout) and anchor those families with manual 'reasoning' exemplars, which the
judge can never delete. Uses ONLY training traffic — the holdout is untouched.

Dry run:  PYTHONPATH=. python scripts/curate_hard.py
Apply:    PYTHONPATH=. python scripts/curate_hard.py --apply
"""

import json
import sys
from datetime import datetime, timezone

import numpy as np

from router import config
from router.embed import get_client
from scripts.traffic import HARD

THRESHOLD = 0.80


def main():
    apply = "--apply" in sys.argv
    rows = [json.loads(l) for l in open(config.EXEMPLARS_PATH) if l.strip()]
    hard_vecs = np.array(get_client().embed(list(HARD)), dtype=np.float32)
    hard_vecs /= np.linalg.norm(hard_vecs, axis=1, keepdims=True)

    keep, dropped = [], []
    for r in rows:
        if r["source"] == "judge" and r["label"] == "simple":
            v = np.array(r["embedding"], dtype=np.float32)
            v /= np.linalg.norm(v)
            sim = float((hard_vecs @ v).max())
            if sim >= THRESHOLD:
                dropped.append((sim, r["text"][:80]))
                continue
        keep.append(r)

    print(f"judge-simple exemplars in hard territory (>= {THRESHOLD}): {len(dropped)}")
    for sim, t in sorted(dropped, reverse=True):
        print(f"  {sim:.3f}  {t}")

    existing = {r["text"] for r in keep}
    added = 0
    if apply:
        ts = datetime.now(timezone.utc).isoformat()
        for q, v in zip(HARD, hard_vecs):
            if q not in existing:
                keep.append({"text": q, "label": "reasoning", "source": "manual",
                             "ts": ts, "embedding": v.tolist()})
                added += 1
        tmp = config.EXEMPLARS_PATH.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")
        tmp.replace(config.EXEMPLARS_PATH)
        print(f"applied: dropped {len(dropped)}, added {added} manual reasoning anchors, "
              f"total {len(keep)}")
    else:
        print("(dry run — pass --apply to write)")


if __name__ == "__main__":
    main()
