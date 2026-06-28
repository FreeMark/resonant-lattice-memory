"""eval_embed.py — embedding sources for the harness, with a disk cache so presets reuse vectors.

Two sources, one cache:
  * deterministic_embed — a seeded pseudo-embedding (unit vector) for fast, Ollama-free tests/CI.
  * ollama_embed        — real ``nomic-embed-text`` via the running Ollama; the signal that
                          actually matters for tuning recall quality.
  * CachedEmbedder      — memoizes by (model, text) to an on-disk JSON cache so a corpus is
                          embedded ONCE and every preset run reuses it (determinism + speed; the
                          embeddings are identical across presets, so only the DYNAMICS vary).

The replay sizes the store's vector_dim from whatever the embedder returns, so the two can never
drift out of sync. ``embed_fn(text) -> list[float] | None`` is the only contract.
"""

import hashlib
import json
import os

import numpy as np

DEFAULT_ENDPOINT = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-text"


def deterministic_embed(text, dim=256):
    """Seeded unit pseudo-embedding — same text always maps to the same vector, no Ollama needed.
    Good enough to exercise the dynamics deterministically in tests; NOT semantically meaningful
    (use ollama_embed for real recall-quality tuning)."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    v = np.random.default_rng(seed).standard_normal(int(dim))
    return (v / (np.linalg.norm(v) or 1.0)).tolist()


def ollama_embed(text, model=DEFAULT_MODEL, endpoint=DEFAULT_ENDPOINT, timeout=60.0):
    """Real embedding from a running Ollama (nomic-embed-text → 768-d). Returns None on failure."""
    import urllib.request
    payload = {"model": model, "prompt": text}
    req = urllib.request.Request(
        f"{endpoint}/api/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8")).get("embedding")
    except Exception:
        return None


class CachedEmbedder:
    """Memoizing wrapper around an ``embed_fn``. Keyed by (model, text); optionally persisted to a
    JSON file so re-runs and parallel preset replays reuse the SAME embeddings (so only the tuned
    dynamics differ between presets, never the vectors)."""

    def __init__(self, embed_fn, cache_path=None, model=DEFAULT_MODEL):
        self._embed = embed_fn
        self._model = model
        self._cache_path = cache_path
        self._cache = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as fh:
                    self._cache = json.load(fh)
            except Exception:
                self._cache = {}

    def __call__(self, text):
        key = f"{self._model}\x00{text}"
        v = self._cache.get(key)
        if v is None:
            v = self._embed(text)
            if v is not None:
                self._cache[key] = v
        return v

    def warm(self, texts):
        """Pre-embed an iterable of texts (one pass) so later (parallel) replays only read cache."""
        for t in texts:
            if t:
                self(t)
        return self

    def save(self):
        if self._cache_path:
            tmp = self._cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._cache, fh)
            os.replace(tmp, self._cache_path)
        return self
