"""Darwin Router proxy. OpenAI-shaped POST /v1/chat/completions:
pre-route (image / gen verbs) -> else embed + kNN -> dispatch to backend.
Every request appended to data/log.jsonl. Run:
    uv run uvicorn router.proxy:app --port 8787
"""

import asyncio
import json
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from router import config, costs, judge
from router.embed import get_client
from router.index import get_index

IMAGEGEN_RE = re.compile(
    r"\b(draw|generate|create|make|paint|sketch)\b.{0,40}\b(image|picture|photo|logo|illustration|drawing|painting)\b",
    re.IGNORECASE | re.DOTALL,
)
VIDEOGEN_RE = re.compile(
    r"\b(make|generate|create|produce|render)\b.{0,40}\b(video|clip|animation|movie)\b",
    re.IGNORECASE | re.DOTALL,
)

http_client = httpx.AsyncClient(timeout=120)


@asynccontextmanager
async def lifespan(app: FastAPI):
    n = get_index().load(embed_fn=get_client().embed)
    print(f"[darwin] index loaded: {n} exemplars (embeddings via {get_client().backend})")
    yield
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)


def extract_query(body: dict) -> tuple[str, bool]:
    """Last user message -> (text, has_image). Content may be str or parts."""
    text, has_image = "", False
    for msg in body.get("messages", []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        else:
            parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            text = "\n".join(p for p in parts if p)
            if any(p.get("type") in ("image_url", "input_image", "image") for p in content):
                has_image = True
    return text, has_image


def pre_route(text: str, has_image: bool) -> str | None:
    if has_image:
        return "vision"
    if IMAGEGEN_RE.search(text):
        return "imagegen"
    if VIDEOGEN_RE.search(text):
        return "videogen"
    return None


def _wrap(content: str) -> dict:
    return {
        "id": "darwin-visual",
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": content,
        }}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }


async def dispatch(route: str, body: dict, text: str) -> tuple[dict, str, str]:
    """Forward to the target backend. Returns (response_json, model, backend)."""
    model = config.MODEL_FOR_ROUTE[route]
    backend = config.BACKEND_FOR_ROUTE[route]
    if route == "imagegen":
        # Sync native API (this key rejects async on the image endpoint)
        r = await http_client.post(
            config.IMAGEGEN_ENDPOINT,
            headers={"Authorization": f"Bearer {config.DASHSCOPE_API_KEY}"},
            json={"model": model,
                  "input": {"messages": [{"role": "user", "content": [{"text": text}]}]},
                  "parameters": {"size": "1024*1024", "n": 1}},
        )
        r.raise_for_status()
        parts = r.json()["output"]["choices"][0]["message"]["content"]
        url = next((c["image"] for c in parts if isinstance(c, dict) and c.get("image")), None)
        return _wrap(f"Here is your image: {url}"), model, backend
    if route == "videogen":
        # Async task submit; per CLAUDE.md we return the acknowledgement, not poll
        r = await http_client.post(
            config.VIDEOGEN_ENDPOINT,
            headers={"Authorization": f"Bearer {config.DASHSCOPE_API_KEY}",
                     "X-DashScope-Async": "enable"},
            json={"model": model, "input": {"prompt": text},
                  "parameters": {"size": "832*480", "duration": 5}},
        )
        r.raise_for_status()
        task_id = r.json()["output"]["task_id"]
        return _wrap(f"Video generation task submitted ({model}), task_id={task_id}. "
                     f"Poll {config.DASHSCOPE_NATIVE_BASE}/tasks/{task_id}"), model, backend
    if backend == "sie":
        base_url, api_key = config.SIE_BASE_URL, config.SIE_API_KEY
    else:
        base_url, api_key = config.DASHSCOPE_BASE_URL, config.DASHSCOPE_API_KEY
    out_body = {**body, "model": model, "stream": False}
    r = await http_client.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=out_body,
    )
    r.raise_for_status()
    return r.json(), model, backend


def log_line(entry: dict):
    with open(config.LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    t0 = time.time()
    body = await request.json()
    is_eval = request.headers.get("x-darwin-eval") == "1"
    text, has_image = extract_query(body)

    route = pre_route(text, has_image)
    margin = None
    embedding = None
    if route is None:
        embedding = (await asyncio.to_thread(get_client().embed, [text]))[0]
        route, margin = get_index().route(embedding)

    try:
        resp, model, backend = await dispatch(route, body, text)
    except httpx.HTTPStatusError as e:
        return JSONResponse(status_code=502, content={
            "error": {"message": f"backend error for route {route}: {e.response.text[:300]}"}
        })

    usage = resp.get("usage") or {}
    tokens_in = usage.get("prompt_tokens", 0)
    tokens_out = usage.get("completion_tokens", 0)
    latency_ms = int((time.time() - t0) * 1000)
    if route in ("imagegen", "videogen"):
        cost = cf_cost = costs.flat_cost(model)  # no routing alternative
    else:
        cost = costs.estimate(model, tokens_in, tokens_out)
        cf_cost = costs.counterfactual(tokens_in, tokens_out)

    darwin = {
        "route": route,
        "margin": margin,
        "model": model,
        "latency_ms": latency_ms,
        "cost_estimate": cost,
        "counterfactual_cost": cf_cost,
    }
    log_line({
        "ts": datetime.now(timezone.utc).isoformat(),
        "query_preview": text[:120],
        "route": route,
        "margin": margin,
        "backend": backend,
        "model": model,
        "latency_ms": latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost": cost,
        "counterfactual_cost": cf_cost,
        "eval": is_eval,
        "judged": None,
        "judge_verdict": None,
    })
    if judge.should_judge(route, margin, is_eval):
        asyncio.create_task(judge.run_judge(body, text, embedding, route, margin, resp))

    return JSONResponse(
        content={**resp, "x_darwin": darwin},
        headers={"x-darwin-route": route, "x-darwin-margin": str(margin)},
    )


@app.post("/route")
async def route_only(request: Request):
    """Routing decision only — no dispatch, no logging, no judge. Used by
    replay.py so eval traffic can never leak into the index."""
    body = await request.json()
    emb = (await asyncio.to_thread(get_client().embed, [body["text"]]))[0]
    label, margin = get_index().route(emb)
    return {"route": label, "margin": margin}


@app.get("/stats")
async def stats():
    """Aggregates data/log.jsonl for the dashboard (Phase 3 polls this)."""
    entries = []
    if config.LOG_PATH.exists():
        with open(config.LOG_PATH) as f:
            entries = [json.loads(l) for l in f if l.strip()]
    live = [e for e in entries if not e.get("eval")]
    by_route: dict[str, int] = {}
    for e in live:
        by_route[e["route"]] = by_route.get(e["route"], 0) + 1
    cost_curve = []
    cum = cum_cf = 0.0
    for e in live:
        cum += e["cost"]
        cum_cf += e["counterfactual_cost"]
        cost_curve.append({"cost": round(cum, 6), "counterfactual": round(cum_cf, 6)})

    judge_events = []
    if config.JUDGE_LOG_PATH.exists():
        with open(config.JUDGE_LOG_PATH) as f:
            judge_events = [json.loads(l) for l in f if l.strip()]
    replay_points = []
    if config.REPLAY_LOG_PATH.exists():
        with open(config.REPLAY_LOG_PATH) as f:
            replay_points = [json.loads(l) for l in f if l.strip()]

    idx = get_index()
    return {
        "requests": len(live),
        "by_route": by_route,
        "total_cost": round(sum(e["cost"] for e in live), 6),
        "total_counterfactual_cost": round(sum(e["counterfactual_cost"] for e in live), 6),
        "exemplars": {
            "total": len(idx.meta),
            "judge_added": idx.count(source="judge"),
        },
        "cost_curve": cost_curve,
        "judge": {
            "events": len(judge_events),
            "inserted": sum(1 for e in judge_events if e.get("inserted")),
            "recent": judge_events[-10:],
        },
        "replay": replay_points,
        "recent": [
            {k: e[k] for k in ("ts", "query_preview", "route", "margin", "latency_ms", "cost")}
            for e in live[-20:]
        ],
    }


@app.get("/healthz")
async def healthz():
    return {"ok": True, "exemplars": len(get_index().meta)}


app.mount("/", StaticFiles(directory=config.ROOT / "dashboard", html=True), name="dashboard")
