# Darwin Router — Superlinked x Qwen Hackathon (14 Aug 2026)

A self-improving semantic router. Routing decisions are kNN lookups over a vector
index of labelled exemplars (embeddings from SIE). Misroutes detected by an
LLM-as-judge get written back into the index, so the router improves online with
no training loop. Demo = accuracy curve climbing over the day.

## Hard constraints
- Submissions close 18:00 SHARP. Working end-to-end path by 12:30, polish after.
- Python 3.12. FastAPI. No auth, no DB — JSONL files for all persistence.
- Judging rewards using BOTH backends: SIE (local) and Alibaba Cloud Model Studio.

## Backends (RESOLVED from onboarding doc, 14 Aug morning)
- SIE (managed cloud): OpenAI-compatible at `https://api.superlinked.com/v1`
  (EU: `https://eu.api.superlinked.com/v1`). Key from console.superlinked.com
  (sk-sie-..., shown once); apply for $500 hackathon credit grant, mention Qwen
  hackathon. Models resident: `Qwen/Qwen3-Embedding-4B` (2560-dim, 32K ctx),
  `Qwen/Qwen3-Reranker-0.6B/4B` (+ `/v1/rerank`), `Qwen/Qwen3.5-4B` (fast chat),
  `Qwen/Qwen3.6-27B` (smart chat, text+image). 402 INSUFFICIENT_CREDITS → ask
  organizers for top-up. → "simple" route runs ON SIE (Qwen3.5-4B).
- Model Studio chat: workspace MaaS host (NOT public intl endpoint):
  OpenAI-compatible `https://ws-217y1bpliyzcf5nl.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1`.
  API key shared out-of-band at event (Singapore region, default workspace).
- Model Studio visual (per Alibaba catalog guide): native API on
  `https://dashscope-intl.aliyuncs.com/api/v1`. Image = SYNC
  `/services/aigc/multimodal-generation/generation`, model `qwen-image-3.0-pro`
  (confirmed via workspace /models; also `wan2.7-image`, `wan2.7-image-pro`).
  Video = ASYNC `/services/aigc/video-generation/video-synthesis` + poll
  `/tasks/{task_id}`, header `X-DashScope-Async: enable`, models
  `happyhorse-1.1-t2v` / `-i2v` / `-r2v` (NOT wan; Wan2.7-image exists, ID TBD
  in console).
  Env: `DASHSCOPE_API_KEY`, `SIE_API_KEY` (URLs/models have correct defaults in
  `router/config.py`; see `.env.example`).

## Routes
| route      | target                  | examples of traffic                       |
|------------|-------------------------|-------------------------------------------|
| simple     | qwen-flash (ModelStudio) or small Qwen on SIE if available | greetings, lookups, formatting, short extraction |
| reasoning  | qwen-max                | math, multi-step, code, planning          |
| vision     | qwen-vl                 | any request with an image attached (pre-route, no kNN needed) |
| imagegen   | qwen-image-pro-3.0 (Model Studio, sync native API) | "draw/generate/create an image of..." |
| videogen   | happyhorse-1.1-t2v (Model Studio, async native API — NOT wan) | "make a video of..." |

Pre-route rules (no embedding call): payload has image → vision; explicit
generate-image/video verbs → imagegen/videogen. Everything else → kNN.

## Components (build in this order)

### Phase 1 — proxy + static router (target: done by 12:30)
- `router/embed.py` — thin client for SIE `/v1/embeddings`. Batch, retry once.
- `router/index.py` — in-memory exemplar store: numpy matrix + metadata list.
  Cosine kNN (k=7, majority vote, record margin = top-class share). Persist to
  `data/exemplars.jsonl`, reload on start. NO external vector DB.
- `router/bootstrap.py` — cold start: ask qwen-max to generate ~20 labelled
  queries per text route (simple/reasoning), varied length + phrasing, output
  strict JSON. Embed and load into index. One command: `python -m router.bootstrap`.
  Deliberately small: a weaker start makes the learning curve visibly steeper.
- `router/proxy.py` — FastAPI. POST `/v1/chat/completions` (OpenAI-shaped).
  Pre-route → else embed → kNN → dispatch to target backend → return response
  with extra `x-darwin` fields (route, margin, latency_ms, cost_estimate,
  counterfactual_cost). Append every request to `data/log.jsonl`.
- `router/costs.py` — hardcoded $/1M-token table per model. Fine if approximate;
  label as estimates in UI.

### Phase 2 — the Darwin loop (target: done by 15:30)
- `router/judge.py` — exploration gate: margin < 0.7 OR random 10% of text
  requests. For gated requests, async fire-and-forget: SHADOW-RUN the
  alternative text route, then send query + BOTH responses (anonymised as A/B,
  randomise order) to qwen-max with JUDGE_PROMPT below. A judge cannot assess a
  counterfactual it never saw — always compare two real outputs.
  Poisoning guards (all must pass before an exemplar is inserted):
  1. clear verdict, not "roughly equal"
  2. not a near-duplicate: cosine < 0.95 vs existing exemplars of that route
  3. cap: max 40 judge-added exemplars per route
  4. bootstrap exemplars are never deleted or overwritten
  Then append `{text, label, source:"judge", ts}` and upsert into index live.
- `router/replay.py` — hold out 40 labelled queries (write them by hand at
  lunch, mixed difficulty; do NOT let bootstrap generate them). IMMUTABLE:
  replay requests carry header `x-darwin-eval: 1`, which bypasses the judge
  loop entirely — eval traffic must NEVER be inserted into the index, or the
  curve is memorisation, not learning. Replays run against the router's
  index-at-time-T; snapshot index every 30 min (`data/snapshots/`).
  Report per snapshot: routing accuracy, avg cost/request, route-share %,
  p50 latency — the pitch chart is "quality up, cost flat", not accuracy alone.

### Phase 3 — demo surface (target: done by 16:30)
- `dashboard/index.html` — single static page, Chart.js from CDN, polls
  `/stats` every 2s. Three panels: live request feed (query, route, margin),
  cumulative cost vs all-qwen-max counterfactual, accuracy-over-time line from
  replay snapshots. Dark theme. No build step, no React.
- `/stats` endpoint aggregates `data/log.jsonl` + replay results.

## JUDGE_PROMPT (v1 — pairwise; iterate if verdicts are noisy)
System: You compare two model responses to the same query. One is from a cheap
fast model, one from a frontier model. You do not know which is which.
User: Query: {q}\nResponse A: {resp_a}\nResponse B: {resp_b}\n
Score each 1-5 on correctness and completeness. Reply strict JSON:
{"a": int, "b": int, "verdict": "a"|"b"|"tie", "why": "<15 words"}
Caller-side rule: if the cheap model's response scores >= 4 OR within 1 point
of the expensive one, the correct route was "simple" — adequacy wins, not
maximum quality. Insert exemplar only when |a-b| >= 2 (clear verdict).

## Data schemas
- exemplar: {"text": str, "label": route, "source": "bootstrap"|"judge"|"manual", "ts": iso}
- log line: {"ts", "query_preview" (first 120 chars), "route", "margin", "backend",
  "model", "latency_ms", "tokens_in", "tokens_out", "cost", "counterfactual_cost",
  "judged": bool|null, "judge_verdict": null|route}

## STUB / SKIP (do not gold-plate)
- SKIP: streaming, auth, retries beyond 1, tool calling, agents, tests beyond
  one smoke test per phase, docker, deployment.
- STUB: cost table (approximate), imagegen/videogen dispatch can return the
  task-submitted acknowledgement rather than polling to completion — EXCEPT
  build one working happyhorse-1.1-t2v call for the visual-track submission.
- If SIE embeddings are slow/unavailable: fall back to Model Studio's embedding
  endpoint; keep the SIE client interface so we can swap back.

## OPEN QUESTIONS — mostly resolved (see Backends). Still open:
1. Get DASHSCOPE_API_KEY (shared out-of-band — ask organizers / Discord
   @Filip M, https://discord.gg/gsyXkdbKT).
2. Human step: sign up superlinked.com/cloud, create SIE key, apply for credit.
3. Rate limits on the workspace key (esp. qwen-max — judge loop depends on it;
   if tight, judge sample rate drops to 10%). Not stated in docs; test at kickoff.
4. Confirm model IDs available on the workspace host (qwen-max / qwen-vl-max /
   wan naming can differ on MaaS workspaces) — `GET /models` after key arrives.

## Submission requirements (form: https://forms.gle/GHtfejKYW6dtsLBf6, due 18:00 HARD)
- Short description + PUBLIC repo + 2-minute video
- Alibaba visual track (MANDATORY): Qwen image/video-generated visualization
  explaining our solution → budget 30 min before 17:00; wan call doubles as this.
- Social media track: public post w/ Qwen visuals + sponsor tags (impressions
  judged at ~18:00, so post EARLY afternoon, not at deadline).
- Judging criterion (main): "Best application built around SIE (optional
  Alibaba Cloud offload rewarded)" — recommended demo is literally "offloading
  with a routing dashboard" = Darwin. Frame it that way: SIE is the core
  (embeddings + cheap lane + rerank), Alibaba is the offload tier.

## Pitch skeleton (write slides at 17:00, not before)
1. Most routers need training runs to adapt; models and traffic change faster
   than retraining cycles. (Do NOT claim online routing is unexplored — it
   isn't. The claim is simplicity, not novelty.)
2. Darwin is TRAINING-FREE: router memory is a vector index, so adaptation is
   literally an index write. Cold-started from nothing this morning.
3. Live chart: quality up from 11:30 to now, avg cost/request flat, all on an
   immutable holdout the router never learned from.
4. Built on SIE embeddings + rerank and Model Studio tiers; wan made our video.

## Deferred (only if ahead of schedule)
- Third text tier (qwen-plus): add only if judge data shows a clear middle
  band. Two tiers with clean labels beat three with noisy ones on day one.
