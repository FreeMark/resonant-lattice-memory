"""eval_embedders.py — compare embedding models on the reference corpus (model-selection stage).

Holds the dynamics FIXED (low decay so facts persist — recall failures reflect the EMBEDDER, not
forgetting) and varies only the embedder. Reports top-1 ranking + recall + poison leak in two modes:
  * pure   — keyword_weight 0: the embedder's RAW semantic ranking quality.
  * hybrid — keyword_weight 0.35 + relevance_margin 0.10: the realistic P2a configuration.
The harness auto-sizes vector_dim to whatever the embedder returns. Needs Ollama with the embedders
pulled. Usage: python eval_embedders.py [model ...]
"""

import eval_corpus
import eval_metrics
import eval_replay
from eval_embed import CachedEmbedder, ollama_embed

# Facts persist (decay 0.1, high initial + recall bump) so the score isolates embedding quality.
_PURE = {"keyword_weight": 0.0, "relevance_margin": None, "decay_per_cycle": 0.1,
         "initial_resonance": 6, "recall_bump": 2.0, "block_size": 8}
_HYBRID = {**_PURE, "keyword_weight": 0.35, "relevance_margin": 0.10}


def bench(models, corpus_path="eval_corpus_reference.json", endpoint="http://localhost:11434"):
    corpus = eval_corpus.load_corpus(corpus_path)
    texts = eval_corpus.all_texts(corpus)
    rows = []
    for m in models:
        emb = CachedEmbedder(lambda t, _m=m: ollama_embed(t, model=_m, endpoint=endpoint), model=m)
        probe = emb("dimension probe")
        if not probe:
            rows.append({"model": m, "error": "no embedding (model loaded? try /api/embed)"})
            continue
        emb.warm(texts)
        out = {"model": m, "dim": len(probe)}
        for tag, cfg in (("pure", _PURE), ("hybrid", _HYBRID)):
            s = eval_metrics.summarize(eval_replay.replay(corpus, config=cfg, embed_fn=emb))
            out[f"top1_{tag}"] = s["relevance_ordering"]["top1_accuracy"]
            out[f"recall_{tag}"] = s["right_time_recall"]["recall"]
            out[f"poison_{tag}"] = s["poison"]["leak_rate"]
        rows.append(out)
    return rows


def format_table(rows):
    h = (f"{'embedder':<24}{'dim':>6}{'top1_pure':>10}{'top1_hyb':>9}"
         f"{'rec_hyb':>8}{'pois_hyb':>9}")
    out = [h, "-" * len(h)]
    for r in rows:
        if r.get("error"):
            out.append(f"{r['model']:<24}  ERROR: {r['error']}")
            continue
        out.append(f"{r['model']:<24}{r['dim']:>6}{r['top1_pure']:>10.2f}{r['top1_hybrid']:>9.2f}"
                   f"{r['recall_hybrid']:>8.2f}{r['poison_hybrid']:>9.2f}")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    default = ["nomic-embed-text", "embeddinggemma:300m", "mxbai-embed-large:335m",
               "qwen3-embedding:0.6b", "qwen3-embedding:4b"]
    print(format_table(bench(sys.argv[1:] or default)))
