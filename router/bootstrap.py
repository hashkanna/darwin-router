"""Cold start: qwen-max generates ~20 labelled queries per text route, embed
them, write data/exemplars.jsonl. Deliberately small — a weaker start makes
the learning curve visibly steeper. Run: python -m router.bootstrap"""

import json
import sys

import httpx

from router import config
from router.embed import get_client
from router.index import get_index

N_PER_ROUTE = 20

ROUTE_BRIEF = {
    "simple": (
        "greetings, casual chit-chat, quick factual lookups, formatting requests, "
        "short extraction or rewrite tasks — things a small fast model handles well"
    ),
    "reasoning": (
        "math problems, multi-step logic, writing or debugging code, planning and "
        "analysis — things that need a frontier model"
    ),
}

GEN_PROMPT = """Generate {n} realistic user queries a chat assistant might receive, all belonging to the category "{route}": {brief}.
Vary length (3 to 40 words), tone, formality, and phrasing. No numbering.
Reply with STRICT JSON only: {{"queries": ["...", "..."]}}"""


def _gen_backend() -> tuple[str, str, str]:
    if config.DASHSCOPE_API_KEY:
        return config.DASHSCOPE_BASE_URL, config.DASHSCOPE_API_KEY, "qwen-max"
    return config.SIE_BASE_URL, config.SIE_API_KEY, config.SIE_SMART_MODEL


def _chat(prompt: str) -> str:
    base_url, api_key, model = _gen_backend()
    r = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _parse_queries(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        raw = raw.removeprefix("json").strip()
    qs = json.loads(raw)["queries"]
    return [q.strip() for q in qs if isinstance(q, str) and q.strip()]


def main():
    if config.EXEMPLARS_PATH.exists():
        print(f"{config.EXEMPLARS_PATH} already exists — delete it to re-bootstrap.")
        sys.exit(1)
    if not (config.DASHSCOPE_API_KEY or config.SIE_API_KEY):
        print("Neither DASHSCOPE_API_KEY nor SIE_API_KEY set (.env)")
        sys.exit(1)

    embedder = get_client()
    index = get_index()
    gen_model = _gen_backend()[2]
    for route in config.TEXT_ROUTES:
        print(f"[{route}] generating ~{N_PER_ROUTE} queries via {gen_model}...")
        queries = _parse_queries(
            _chat(GEN_PROMPT.format(n=N_PER_ROUTE, route=route, brief=ROUTE_BRIEF[route]))
        )
        print(f"[{route}] got {len(queries)}, embedding via {embedder.backend}...")
        vecs = embedder.embed(queries)
        for q, v in zip(queries, vecs):
            index.add(q, route, "bootstrap", v)
        print(f"[{route}] done.")
    print(f"Bootstrap complete: {len(index.meta)} exemplars -> {config.EXEMPLARS_PATH}")


if __name__ == "__main__":
    main()
