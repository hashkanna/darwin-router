"""The Darwin loop. Exploration gate (low margin or random sample) fires an
async judge: shadow-run the alternative text route, show BOTH real responses
to the judge anonymised as A/B, and on a clear verdict write the query back
into the index as a labelled exemplar (with poisoning guards). Never raises
into the request path."""

import json
import random
from datetime import datetime, timezone

import httpx

from router import config
from router.index import get_index

MARGIN_GATE = 0.7
SAMPLE_RATE = 0.10
DEDUP_COS = 0.95
CAP_PER_ROUTE = 40
RESP_TRUNC = 6000  # chars of each response shown to the judge; too low biases
                   # the judge toward the cheap model (long answers get cut)

JUDGE_SYSTEM = (
    "You compare two model responses to the same query. One is from a cheap "
    "fast model, one from a frontier model. You do not know which is which."
)
JUDGE_USER = (
    "Query: {q}\nResponse A: {resp_a}\nResponse B: {resp_b}\n"
    "First, solve the query yourself in brief. Then check each response "
    "against your solution for concrete errors (wrong numbers, invalid steps, "
    "missed constraints, unstated failure cases). A response with any material "
    "error scores at most 2. Only then score each 1-5 on correctness and "
    "completeness. Reply strict JSON on the last line: "
    '{{"a": int, "b": int, "verdict": "a"|"b"|"tie", "why": "<15 words"}}'
)

http_client = httpx.AsyncClient(timeout=120)


def should_judge(route: str, margin: float | None, is_eval: bool) -> bool:
    if is_eval or route not in config.TEXT_ROUTES or margin is None:
        return False
    return margin < MARGIN_GATE or random.random() < SAMPLE_RATE


def _judge_backend() -> tuple[str, str, str]:
    if config.DASHSCOPE_API_KEY:
        return config.DASHSCOPE_BASE_URL, config.DASHSCOPE_API_KEY, "qwen-max"
    return config.SIE_BASE_URL, config.SIE_API_KEY, config.SIE_SMART_MODEL


def _content(resp: dict) -> str:
    try:
        return resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


async def _chat(base_url: str, api_key: str, model: str, messages: list) -> dict:
    r = await http_client.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": messages, "stream": False},
    )
    r.raise_for_status()
    return r.json()


def _parse_verdict(raw: str) -> dict:
    # verification-first prompt puts reasoning before the JSON: take last {...}
    start, end = raw.rfind("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON in verdict: {raw[-120:]}")
    return json.loads(raw[start : end + 1])


def _log(entry: dict):
    with open(config.JUDGE_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


async def run_judge(body: dict, text: str, embedding: list[float], routed: str,
                    margin: float, primary_resp: dict):
    """Fire-and-forget from the proxy. Swallows all errors."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query_preview": text[:120],
        "routed": routed,
        "margin": margin,
        "correct": None,
        "inserted": False,
        "why": None,
    }
    try:
        shadow = "reasoning" if routed == "simple" else "simple"
        s_model = config.MODEL_FOR_ROUTE[shadow]
        s_base, s_key = (
            (config.SIE_BASE_URL, config.SIE_API_KEY)
            if config.BACKEND_FOR_ROUTE[shadow] == "sie"
            else (config.DASHSCOPE_BASE_URL, config.DASHSCOPE_API_KEY)
        )
        shadow_resp = await _chat(s_base, s_key, s_model, body["messages"])

        by_route = {routed: _content(primary_resp), shadow: _content(shadow_resp)}
        cheap, expensive = by_route["simple"][:RESP_TRUNC], by_route["reasoning"][:RESP_TRUNC]
        if not cheap or not expensive:
            entry["why"] = "empty response, skipped"
            return

        cheap_is_a = random.random() < 0.5
        resp_a, resp_b = (cheap, expensive) if cheap_is_a else (expensive, cheap)
        j_base, j_key, j_model = _judge_backend()
        verdict_raw = _content(await _chat(j_base, j_key, j_model, [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(q=text[:1000], resp_a=resp_a, resp_b=resp_b)},
        ]))
        v = _parse_verdict(verdict_raw)
        cheap_score = int(v["a"] if cheap_is_a else v["b"])
        exp_score = int(v["b"] if cheap_is_a else v["a"])
        entry["cheap_score"], entry["exp_score"] = cheap_score, exp_score
        entry["judge_why"] = str(v.get("why", ""))[:100]

        # Adequacy rule: cheap good enough -> simple. Clear frontier win -> reasoning.
        if exp_score - cheap_score >= 2:
            correct, clear = "reasoning", True
        elif cheap_score >= 4 or exp_score - cheap_score <= 1:
            correct, clear = "simple", cheap_score >= 4
        else:
            correct, clear = None, False
        entry["correct"] = correct
        if not clear:
            entry["why"] = "no clear verdict"
            return

        # Poisoning guards
        index = get_index()
        if index.count(label=correct, source="judge") >= CAP_PER_ROUTE:
            entry["why"] = "cap reached"
            return
        # verdict flip on a repeated query: overturning an existing exemplar
        # demands DECISIVE evidence (gap >= 3), else the old verdict stands —
        # a lucky cheap-model run must not erase a decisive failure
        if index.max_conflicting_sim(embedding, correct) >= DEDUP_COS:
            gap = (exp_score - cheap_score) if correct == "reasoning" else (cheap_score - exp_score)
            if gap < 3:
                entry["why"] = "flip rejected (verdict not decisive)"
                return
            entry["flipped"] = index.remove_conflicting(embedding, correct, DEDUP_COS)
        if index.max_sim(embedding, correct) >= DEDUP_COS:
            entry["why"] = "near-duplicate"
            return
        index.add(text, correct, "judge", embedding)
        entry["inserted"] = True
        entry["why"] = "misroute corrected" if correct != routed else "boundary sharpened"
    except Exception as e:
        entry["why"] = f"error: {type(e).__name__}: {str(e)[:120]}"
    finally:
        _log(entry)
