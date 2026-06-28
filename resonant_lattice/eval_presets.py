"""eval_presets.py — run a corpus under N config presets and compare.

Embeddings are computed ONCE into a shared cache, so presets differ ONLY in their dynamics (never
in the vectors); the replays then run in parallel (each gets its own temp store). A preset is
``{"name": str, "config": {...}}`` — exactly the JSON the slider page (A12) exports. Results are
ranked by right-time recall, then by (low) poison leak — the A1/A6 priority order.
"""

import concurrent.futures as _cf

import eval_corpus
import eval_metrics
import eval_replay


def default_presets():
    """A starter sweep across the knobs most likely to move right-time recall (A1): decay rate,
    initial resonance vs the promotion bar, and recall reinforcement. Swap in slider-page exports."""
    return [
        {"name": "baseline", "config": {}},
        {"name": "slow-decay", "config": {"decay_per_cycle": 0.25, "initial_resonance": 5}},
        {"name": "recall-required", "config": {"initial_resonance": 4, "promotion_resonance_threshold": 4,
                                               "recall_bump": 2.0, "decay_per_cycle": 0.5}},
        {"name": "sticky", "config": {"decay_per_cycle": 0.1, "initial_resonance": 6,
                                      "recall_bump": 2.0}},
        {"name": "aggressive-decay", "config": {"decay_per_cycle": 3.0, "initial_resonance": 3,
                                                "recall_bump": 0.0, "session_gap_cycles": 8,
                                                "dream_every_n": 1}},
        {"name": "precise", "config": {"recall_floor": 0.30, "block_size": 6,
                                       "decay_per_cycle": 0.25, "initial_resonance": 5,
                                       "recall_bump": 2.0,
                                       "keyword_weight": 0.35, "relevance_margin": 0.10}},
        {"name": "tiered", "config": {"recall_floor": 0.30, "block_size": 6,
                                      "decay_per_cycle": 0.25, "initial_resonance": 5,
                                      "recall_bump": 2.0, "keyword_weight": 0.35,
                                      "relevance_margin": 0.10, "dormant_floor": 3.0,
                                      "strong_cue": 0.55}},
    ]


def run_presets(corpus, presets=None, embed_fn=None, max_workers=4):
    """Return ``[{name, config, metrics}]`` for each preset, ranked best-first (high right-time
    recall, then low poison leak). Embeddings are warmed once into a shared cache before the
    parallel replays, so the comparison isolates the tuned dynamics."""
    eval_corpus.validate_corpus(corpus)
    presets = presets or default_presets()
    from eval_embed import CachedEmbedder, deterministic_embed
    if embed_fn is None:
        embed_fn = deterministic_embed
    cached = embed_fn if isinstance(embed_fn, CachedEmbedder) else CachedEmbedder(embed_fn)
    # Warm once (corpus texts + the replay's dim-probe) so parallel replays only READ the cache.
    cached.warm(eval_corpus.all_texts(corpus) + ["dimension probe"])

    def _one(preset):
        res = eval_replay.replay(corpus, config=preset.get("config"), embed_fn=cached)
        return {"name": preset.get("name", "?"), "config": preset.get("config", {}),
                "metrics": eval_metrics.summarize(res)}

    if max_workers and max_workers > 1 and len(presets) > 1:
        with _cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            rows = list(ex.map(_one, presets))
    else:
        rows = [_one(p) for p in presets]
    rows.sort(key=lambda r: (-r["metrics"]["relevance_ordering"]["top1_accuracy"],
                             -r["metrics"]["right_time_recall"]["recall"],
                             r["metrics"]["poison"]["leak_rate"]))
    return rows


def format_table(rows):
    """Compact best-first comparison table: recall (A1), top-1 ranking (contextual relevance),
    poison leak (A6 guardrail), tool-halluc baseline."""
    out = [f"{'preset':<18}{'recall':>8}{'top1':>7}{'satisf':>8}{'poisonLk':>10}{'toolHal':>9}",
           "-" * 60]
    for r in rows:
        m = r["metrics"]
        out.append(f"{r['name']:<18}"
                   f"{m['right_time_recall']['recall']:>8.2f}"
                   f"{m['relevance_ordering']['top1_accuracy']:>7.2f}"
                   f"{m['right_time_recall']['turn_satisfaction_rate']:>8.2f}"
                   f"{m['poison']['leak_rate']:>10.2f}"
                   f"{m['tool']['rate']:>9.2f}")
    return "\n".join(out)
