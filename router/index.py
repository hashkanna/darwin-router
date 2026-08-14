"""In-memory exemplar store: numpy matrix + metadata list. Cosine kNN (k=7,
majority vote, margin = top-class share). Persists to data/exemplars.jsonl
(embedding stored per line so restart needs no re-embed). No vector DB."""

import json
import shutil
import threading
from datetime import datetime, timezone

import numpy as np

from router import config

K = 7


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


class ExemplarIndex:
    def __init__(self):
        self.vecs: np.ndarray | None = None  # (n, d), L2-normalised
        self.meta: list[dict] = []
        self.lock = threading.Lock()

    # -- persistence ---------------------------------------------------------

    def load(self, embed_fn=None) -> int:
        """Load exemplars.jsonl. Lines missing an embedding get re-embedded
        via embed_fn (batch)."""
        if not config.EXEMPLARS_PATH.exists():
            return 0
        rows = []
        with open(config.EXEMPLARS_PATH) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        missing = [r for r in rows if "embedding" not in r]
        if missing:
            if embed_fn is None:
                raise RuntimeError(f"{len(missing)} exemplars lack embeddings and no embed_fn given")
            vecs = embed_fn([r["text"] for r in missing])
            for r, v in zip(missing, vecs):
                r["embedding"] = v
            self._rewrite(rows)
        with self.lock:
            self.meta = [{k: r[k] for k in ("text", "label", "source", "ts")} for r in rows]
            self.vecs = _norm(np.array([r["embedding"] for r in rows], dtype=np.float32))
        return len(rows)

    def _rewrite(self, rows: list[dict]):
        tmp = config.EXEMPLARS_PATH.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        tmp.replace(config.EXEMPLARS_PATH)

    def add(self, text: str, label: str, source: str, embedding: list[float], persist: bool = True):
        row = {
            "text": text,
            "label": label,
            "source": source,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        v = _norm(np.array(embedding, dtype=np.float32).reshape(1, -1))
        with self.lock:
            self.meta.append(row)
            self.vecs = v if self.vecs is None else np.vstack([self.vecs, v])
            if persist:
                with open(config.EXEMPLARS_PATH, "a") as f:
                    f.write(json.dumps({**row, "embedding": embedding}) + "\n")

    # -- queries -------------------------------------------------------------

    def route(self, embedding: list[float]) -> tuple[str, float]:
        """Returns (label, margin). Margin = share of top class among k
        neighbours. Empty index -> ('reasoning', 0.0) as the safe default."""
        with self.lock:
            if self.vecs is None or len(self.meta) == 0:
                return "reasoning", 0.0
            q = _norm(np.array(embedding, dtype=np.float32))
            sims = self.vecs @ q
            k = min(K, len(self.meta))
            top = np.argpartition(-sims, k - 1)[:k]
            votes: dict[str, int] = {}
            for i in top:
                votes[self.meta[i]["label"]] = votes.get(self.meta[i]["label"], 0) + 1
            label = max(votes, key=votes.get)
            return label, votes[label] / k

    def max_sim(self, embedding: list[float], label: str) -> float:
        """Highest cosine vs existing exemplars of a route (judge dedup guard)."""
        with self.lock:
            if self.vecs is None:
                return 0.0
            mask = np.array([m["label"] == label for m in self.meta])
            if not mask.any():
                return 0.0
            q = _norm(np.array(embedding, dtype=np.float32))
            return float((self.vecs[mask] @ q).max())

    def remove_conflicting(self, embedding: list[float], keep_label: str,
                           threshold: float) -> int:
        """Drop judge-sourced exemplars that near-duplicate this embedding but
        carry a DIFFERENT label (verdict flip on a repeated query): latest
        clear verdict wins. Bootstrap exemplars are never touched."""
        with self.lock:
            if self.vecs is None:
                return 0
            q = _norm(np.array(embedding, dtype=np.float32))
            sims = self.vecs @ q
            drop = {
                i for i, (m, s) in enumerate(zip(self.meta, sims))
                if s >= threshold and m["label"] != keep_label and m["source"] == "judge"
            }
            if not drop:
                return 0
            keep = [i for i in range(len(self.meta)) if i not in drop]
            self.meta = [self.meta[i] for i in keep]
            self.vecs = self.vecs[keep] if keep else None
            # persist: normalized vectors are fine, cosine is norm-invariant
            self._rewrite([
                {**m, "embedding": self.vecs[i].tolist()}
                for i, m in enumerate(self.meta)
            ])
            return len(drop)

    def count(self, label: str | None = None, source: str | None = None) -> int:
        with self.lock:
            return sum(
                1
                for m in self.meta
                if (label is None or m["label"] == label)
                and (source is None or m["source"] == source)
            )

    def snapshot(self) -> str:
        config.SNAPSHOT_DIR.mkdir(exist_ok=True)
        name = datetime.now(timezone.utc).strftime("%H%M%S") + ".jsonl"
        with self.lock:
            shutil.copy(config.EXEMPLARS_PATH, config.SNAPSHOT_DIR / name)
        return name


_index: ExemplarIndex | None = None


def get_index() -> ExemplarIndex:
    global _index
    if _index is None:
        _index = ExemplarIndex()
    return _index
