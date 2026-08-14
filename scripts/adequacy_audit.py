"""One-shot audit: for every holdout item, run BOTH text lanes and apply the
exact judge protocol from router/judge.py. Reports where the hand labels
disagree with the adequacy objective. Never touches the index or the eval
files — pure analysis. Run: python scripts/adequacy_audit.py <out.jsonl>"""

import asyncio
import json
import random
import sys

from router import config
from router.judge import (JUDGE_SYSTEM, JUDGE_USER, RESP_TRUNC, _chat,
                          _content, _judge_backend, _parse_verdict)

CONCURRENCY = 4


async def audit_one(item: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        q = item["text"]
        messages = [{"role": "user", "content": q}]
        cheap_task = _chat(config.SIE_BASE_URL, config.SIE_API_KEY,
                           config.SIE_CHAT_MODEL, messages)
        exp_task = _chat(config.DASHSCOPE_BASE_URL, config.DASHSCOPE_API_KEY,
                         config.MODEL_FOR_ROUTE["reasoning"], messages)
        cheap_r, exp_r = await asyncio.gather(cheap_task, exp_task)
        cheap = _content(cheap_r)[:RESP_TRUNC]
        expensive = _content(exp_r)[:RESP_TRUNC]

        cheap_is_a = random.random() < 0.5
        resp_a, resp_b = (cheap, expensive) if cheap_is_a else (expensive, cheap)
        j_base, j_key, j_model = _judge_backend()
        v = _parse_verdict(_content(await _chat(j_base, j_key, j_model, [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(q=q[:1000], resp_a=resp_a, resp_b=resp_b)},
        ])))
        cheap_score = int(v["a"] if cheap_is_a else v["b"])
        exp_score = int(v["b"] if cheap_is_a else v["a"])

        if exp_score - cheap_score >= 2:
            adequacy, clear = "reasoning", True
        elif cheap_score >= 4 or exp_score - cheap_score <= 1:
            adequacy, clear = "simple", cheap_score >= 4
        else:
            adequacy, clear = None, False
        return {
            "text": q,
            "hand_label": item["label"],
            "adequacy_label": adequacy,
            "clear": clear,
            "cheap_score": cheap_score,
            "exp_score": exp_score,
            "why": str(v.get("why", ""))[:80],
        }


async def main():
    out_path = sys.argv[1]
    in_path = sys.argv[2] if len(sys.argv) > 2 else config.HOLDOUT_PATH
    with open(in_path) as f:
        holdout = [json.loads(l) for l in f if l.strip()]
    sem = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(*(audit_one(i, sem) for i in holdout))

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    agree = sum(1 for r in results if r["adequacy_label"] == r["hand_label"])
    flips = [r for r in results if r["adequacy_label"] != r["hand_label"]]
    print(f"agreement: {agree}/{len(results)}")
    print(f"\nitems where adequacy disagrees with hand label:")
    for r in flips:
        mark = "CLEAR" if r["clear"] else "weak "
        print(f"  [{mark}] hand={r['hand_label']:9s} adequacy={str(r['adequacy_label']):9s} "
              f"({r['cheap_score']} vs {r['exp_score']}) :: {r['text'][:70]}")


if __name__ == "__main__":
    asyncio.run(main())
