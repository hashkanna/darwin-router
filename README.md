# Darwin Router

**🏆 First prize — Superlinked × Qwen Hackathon, 14 Aug 2026**

A **self-improving semantic router**, built in one day.

![Darwin Router poster](assets/poster_pro_a.png)

Most LLM routers need training runs to adapt, but models and traffic change
faster than retraining cycles. Darwin is **training-free**: routing decisions
are kNN lookups over a vector index of labelled exemplars, so adaptation is
literally an index write. Misroutes detected by an LLM-as-judge are written
back into the index — the router improves online, live, with no training loop.

▶️ **[2-minute demo video](assets/demo_video.mp4)** — narrated tour of the live
dashboard, recorded on hackathon day.

## Results (one day, cold start to prize)

Measured on an immutable, hand-written 40-query holdout that replays against
the live index and can never enter it:

| | 09:54 (cold start) | 17:45 (end of day) |
|---|---|---|
| Holdout routing accuracy | 75.0% | **85.0%** |
| Avg cost per request | $0.0022 | **$0.0011** |
| Exemplars in index | 57 | 91 |

Accuracy climbed 10 points while per-request cost **halved** — and total spend
ran ~65% below the send-everything-to-qwen-max counterfactual. Every gained
point came from the Darwin loop writing judged exemplars into the index.

## How it works

```mermaid
flowchart LR
    Q[query] --> P{pre-route rules}
    P -->|image attached| V[qwen-vl-max]
    P -->|"draw/make an image"| I[qwen-image / z-image-turbo]
    P -->|"make a video"| W[happyhorse t2v]
    P -->|text| E[SIE embeddings\nQwen3-Embedding-4B]
    E --> K[cosine kNN over\nexemplar index]
    K -->|simple| S[Qwen3.5-4B on SIE]
    K -->|reasoning| R[qwen-max on Alibaba Cloud]
    K -.->|margin < 0.7 or 10% sample| J[judge loop]
    J -->|shadow-run other route,\nqwen-max compares both| X[insert labelled exemplar\ninto index]
    X -.-> K
```

- **SIE (Superlinked Inference Engine)** is the core: every text request is
  embedded with `Qwen/Qwen3-Embedding-4B`, and the cheap lane is served by
  `Qwen/Qwen3.5-4B` on SIE.
- **Alibaba Cloud Model Studio** is the offload tier: `qwen-max` for reasoning,
  `qwen-vl-max` for vision, `z-image-turbo`/`qwen-image-3.0-pro` for images,
  `happyhorse-1.1-t2v` for video.
- **The Darwin loop**: uncertain routings (low kNN margin, or a 10% random
  sample) trigger an async shadow-run of the alternative route. qwen-max judges
  both real responses blind (A/B, randomized order). On a clear verdict the
  query is inserted into the index as a new exemplar.
- **Honest evaluation**: the holdout replays every 10 minutes against a
  decision-only endpoint — eval traffic can never enter the index, so the
  accuracy curve is learning, not memorisation.
- The judge applies an **adequacy rule**: if the cheap model's answer is good
  enough, the correct route is the cheap one — quality per dollar, not maximum
  quality.

## What we learned (the interesting bits)

1. **The learnable boundary is adequacy, not difficulty.** A 2026-era 4B model
   aces textbook-difficulty tasks — induction proofs, SQL, probability. What
   actually breaks it: pattern-matching traps (twisted river-crossing classics),
   stateful simulation with interruptions, exact symbol manipulation, and
   Python-semantics traps (MRO, mutable defaults, late binding). Embedding
   space captures topic more than trickiness, so this boundary is genuinely
   non-obvious — and worth learning.
2. **An LLM judge can't grade answers it can't produce.** qwen-max reliably
   judged reasoning and writing, but on symbol-manipulation tasks it scored
   confidently-wrong cheap answers as adequate. Poisoning guards are load-
   bearing: near-duplicate dedup (cosine ≥ 0.95), per-route insert caps,
   immutable bootstrap exemplars, verdict-flip hysteresis (overturning an
   exemplar needs a decisive score gap), and an operator curation script for
   judge noise the guards miss.
3. **Index writes beat training runs on hackathon timescales.** Cold start to
   85% in a single day, every adaptation inspectable as a JSONL line.

## Run it

```bash
uv sync
cp .env.example .env       # add SIE_API_KEY and DASHSCOPE_API_KEY
uv run python -m router.bootstrap        # cold-start ~40 exemplars
uv run uvicorn router.proxy:app --port 8787
uv run python -m router.replay --loop &  # 10-min holdout snapshots
open http://127.0.0.1:8787/              # live dashboard
```

Send it OpenAI-shaped traffic:

```bash
curl -s localhost:8787/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"darwin","messages":[{"role":"user","content":"hey there!"}]}'
```

Responses carry an `x_darwin` block: route, kNN margin, model, latency, cost
estimate, and the all-qwen-max counterfactual cost.

## Repo map

| path | what |
|---|---|
| `router/proxy.py` | FastAPI proxy: pre-route → embed → kNN → dispatch → log |
| `router/index.py` | in-memory exemplar store, cosine kNN, JSONL persistence |
| `router/judge.py` | the Darwin loop: gate, shadow-run, blind judge, guarded insert |
| `router/replay.py` | immutable holdout replay + index snapshots |
| `router/bootstrap.py` | LLM-generated cold-start exemplars |
| `scripts/traffic.py` | demo traffic generator (independent of the holdout) |
| `scripts/curate_hard.py` | operator curation for judge noise on hard families |
| `dashboard/index.html` | live dashboard (Chart.js, polls `/stats` every 2s) |
| `data/holdout.jsonl` | hand-written eval set — never enters the index |
| `assets/` | demo video, Qwen-generated poster, HappyHorse concept video |

No database, no vector store, no training infra: numpy + JSONL files.

*Costs shown are hardcoded estimates ($/1M tokens). Built with Claude Code.*
