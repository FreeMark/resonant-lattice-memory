"""eval_run.py — CLI for the Phase-1 eval / tuning harness.

Examples:
  python eval_run.py                                  # example corpus, default presets, pseudo-embeds
  python eval_run.py --ollama                         # real nomic-embed-text via local Ollama
  python eval_run.py --ollama --cache emb.json        # persist/reuse embeddings across runs
  python eval_run.py --corpus mine.jsonl --presets presets.json --ollama --json

A presets file is a JSON list of ``{"name": str, "config": {...}}`` — the export from the slider
page (A12). The harness embeds the corpus once (cached), replays it under each preset in parallel,
and prints a best-first comparison (high right-time recall, low poison leak).
"""

import argparse
import json


def main(argv=None):
    ap = argparse.ArgumentParser(description="Resonant Lattice eval/tuning harness")
    ap.add_argument("--corpus", help="corpus JSON/JSONL (default: built-in example)")
    ap.add_argument("--presets", help="presets JSON [{name,config}] (default: built-in sweep)")
    ap.add_argument("--ollama", action="store_true", help="use real nomic-embed-text via Ollama")
    ap.add_argument("--endpoint", default="http://localhost:11434")
    ap.add_argument("--model", default="nomic-embed-text")
    ap.add_argument("--cache", help="embedding cache JSON path (reused across runs)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json", action="store_true", help="emit full JSON results")
    args = ap.parse_args(argv)

    import eval_corpus
    import eval_presets
    from eval_embed import CachedEmbedder, deterministic_embed, ollama_embed

    corpus = eval_corpus.load_corpus(args.corpus) if args.corpus else eval_corpus.example_corpus()
    presets = None
    if args.presets:
        with open(args.presets, "r", encoding="utf-8") as fh:
            presets = json.load(fh)

    if args.ollama:
        endpoint, model = args.endpoint, args.model
        embed = CachedEmbedder(lambda t: ollama_embed(t, model=model, endpoint=endpoint),
                               cache_path=args.cache, model=model)
    else:
        embed = CachedEmbedder(deterministic_embed, cache_path=args.cache, model="deterministic")

    rows = eval_presets.run_presets(corpus, presets, embed_fn=embed, max_workers=args.workers)
    embed.save()
    print(json.dumps(rows, indent=2) if args.json else eval_presets.format_table(rows))
    return rows


if __name__ == "__main__":
    main()
