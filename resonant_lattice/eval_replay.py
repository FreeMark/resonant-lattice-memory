"""eval_replay.py — drive the REAL store/retriever dynamics over a corpus under one config preset.

Faithful: it builds an actual ``LatticeStore`` + ``LatticeRetriever`` and exercises the real
FactsMixin / DreamCycleMixin / recall code — the dynamics being tuned (resonance, decay, promotion,
conflict) all live in the store, so this measures the genuine behaviour, not a reimplementation.
Deterministic: facts are injected from the corpus and embeddings come from an injectable cached
``embed_fn``, so two presets differ ONLY in their dynamics. Provider-free: Hermes isn't installable,
so the dream-cycle STEP ORDER is replicated over the real store methods.

Per turn (mirroring the Hermes loop — A10): (1) PREFETCH for the user prompt and score it BEFORE
this turn's facts exist; (2) reinforce the recalled facts (affects future turns); (3) ingest the
turn's facts; (4) advance the memory cycle and, on cadence, run the dream-cycle maintenance. A
session boundary advances extra idle cycles so decay/consolidation actually happen "between
sessions" — the A1 'three weeks later' shape.
"""

import os
import tempfile

# Store knobs the slider page (A12) tunes → LatticeStore kwargs (vector_dim is sized from the
# embedder, never from config, so the two can't drift).
_STORE_KEYS = {
    "initial_resonance", "decay_per_cycle", "short_tier_cycles", "mid_tier_cycles",
    "promotion_threshold", "similarity_threshold", "reinforce_threshold",
    "conflict_sim_low", "conflict_sim_high", "novelty_enabled", "novelty_boost", "hrr_dim",
}
# Note: promotion_threshold is the internal LatticeStore kwarg name.
# Public config key is promotion_resonance_threshold (mapped by provider).
# Replay-level knobs (not store constructor args) + their defaults.
_REPLAY_DEFAULTS = {
    "recall_bump": 1.0,         # resonance bump applied to recalled facts (reinforce_on_recall)
    "block_size": 8,            # resonant-block size = prefetch top-k
    "dream_every_n": 3,         # run a dream cycle every N memory cycles
    "session_gap_cycles": 3,    # idle dream cycles between sessions (simulate time passing)
    "conflict_decay_floor": 0.0,
    "keyword_weight": 0.25,     # P2a: significant-token boost on contextual relevance
    "relevance_margin": None,   # P2a: adaptive precision gate (None = keep full ranked block)
    "dormant_floor": None,      # P2b: resonance below which a fact is dormant (None = off)
    "strong_cue": 0.6,          # P2b: relevance that plucks a dormant fact back
    "forget_after_dormant_cycles": 0,  # P2b-store: 0=delete at 0 (legacy); >0=demote then deep-delete
    "conflict_limbo": False,    # P2: hold contested facts in limbo (protect decay/prune, skip bleed)
    "surprise_decay_discount": 0.0,  # A11: high-peak facts (surprising one-offs) fade slower
}


def split_config(config):
    """Split a flat preset ``config`` dict into (store_kwargs, recall_floor, replay_params)."""
    config = dict(config or {})
    store_kwargs = {k: config[k] for k in _STORE_KEYS if k in config}
    recall_floor = float(config.get("recall_floor", 0.30))
    rp = dict(_REPLAY_DEFAULTS)
    for k in rp:
        if k in config:
            rp[k] = config[k]
    return store_kwargs, recall_floor, rp


def _make_harness_retriever(store, embed_fn, recall_floor):
    """A LatticeRetriever whose ONLY change is that _get_embedding routes to the harness embed_fn
    (so recall is deterministic / cacheable and needs no live Ollama for tests)."""
    from retrieval import LatticeRetriever

    class _HarnessRetriever(LatticeRetriever):
        def _get_embedding(self, text):
            return embed_fn(text)

    return _HarnessRetriever(store, "http://harness.invalid", store.embed_model,
                             min_similarity=recall_floor)


def replay(corpus, config=None, embed_fn=None, db_path=None):
    """Replay ``corpus`` under ``config`` and return the list of per-turn result dicts
    ({expected, prefetched, poison, tool_calls}) that eval_metrics scores."""
    from store import LatticeStore
    if embed_fn is None:
        from eval_embed import deterministic_embed
        embed_fn = deterministic_embed
    store_kwargs, recall_floor, rp = split_config(config)

    # Size the store's embedding dim from the embedder so they can never mismatch.
    probe = embed_fn("dimension probe")
    if not probe:
        raise RuntimeError("embed_fn returned no vector — is Ollama running for ollama_embed?")
    store_kwargs["vector_dim"] = len(probe)

    tmpdir = tempfile.mkdtemp()
    store = LatticeStore(db_path=(db_path or os.path.join(tmpdir, "eval.db")), **store_kwargs)
    retr = _make_harness_retriever(store, embed_fn, recall_floor)
    try:
        import holographic as _hg
        _HRR = True
    except Exception:
        _hg, _HRR = None, False

    key_to_fid, fid_to_key = {}, {}
    block = int(rp["block_size"])
    bump = float(rp["recall_bump"])
    dream_every = int(rp["dream_every_n"])
    gap = int(rp["session_gap_cycles"])
    floor = float(rp["conflict_decay_floor"])
    kw = float(rp["keyword_weight"])
    margin = rp["relevance_margin"]
    margin = None if margin is None else float(margin)
    dorm = rp["dormant_floor"]
    dorm = None if dorm is None else float(dorm)
    strong = float(rp["strong_cue"])
    forget_after = int(rp["forget_after_dormant_cycles"])
    limbo = bool(rp["conflict_limbo"])
    peak_disc = float(rp["surprise_decay_discount"])
    mc = [0]  # boxed so the nested _dream can advance the logical clock

    def _dream():
        store.apply_cycle_decay(protect_conflicts=limbo, peak_discount=peak_disc)
        if not limbo:
            store.apply_conflict_decay(floor)
        store.promote_facts()
        store.prune_weak_facts(forget_after, protect_conflicts=limbo)
        store.free_conflict_winners()
        try:
            store.resolve_hrr_conflicts()
        except Exception:
            pass
        store.set_cycle_counts(memory_cycle=mc[0])

    results = []
    try:
        for session in corpus:
            for turn in session:
                # 1. PREFETCH (before this turn's facts exist) — the recall we score.
                user = turn.get("user") or ""
                prefetched_fids = ([h["id"] for h in retr.search(
                    user, limit=block, keyword_weight=kw, relevance_margin=margin,
                    dormant_floor=dorm, strong_cue=strong)]
                    if user else [])
                results.append({
                    "expected": list(turn.get("expect_recall") or []),
                    "expect_top": list(turn.get("expect_top") or []),
                    "prefetched": [fid_to_key[f] for f in prefetched_fids if f in fid_to_key],
                    "poison": list(turn.get("poison") or []),
                    "tool_calls": list(turn.get("tool_calls") or []),
                })
                # 2. RECALL REINFORCEMENT — referencing memories raises resonance (future turns).
                if prefetched_fids and bump:
                    try:
                        store.reinforce_on_recall(prefetched_fids, bump)
                    except Exception:
                        pass
                # 3. INGEST this turn's facts (deterministic; embedding via embed_fn).
                for f in (turn.get("facts") or []):
                    emb = embed_fn(f["content"])
                    if not emb:
                        continue
                    ents = f.get("entities") or []
                    hv = _hg.encode_fact(f["content"], ents, dim=store.hrr_dim) if _HRR else None
                    _action, fid = store.add_or_reinforce_fact(
                        f["content"], emb, f.get("category", "general"), "eval",
                        hrr_vector=hv, entities=ents)
                    if fid and fid > 0:
                        key_to_fid[f["key"]] = fid
                        fid_to_key[fid] = f["key"]
                # 4. CYCLE ADVANCE + cadence dream.
                mc[0] += 1
                store.set_cycle_counts(memory_cycle=mc[0])
                if turn.get("dream") or (dream_every > 0 and mc[0] % dream_every == 0):
                    _dream()
            # SESSION BOUNDARY: idle consolidation so decay actually bites between sessions.
            for _ in range(gap):
                mc[0] += 1
                _dream()
    finally:
        store.close()
    return results


def replay_to_metrics(corpus, config=None, embed_fn=None):
    """Convenience: replay + summarize into the three-metric dict (eval_metrics.summarize)."""
    import eval_metrics
    return eval_metrics.summarize(replay(corpus, config=config, embed_fn=embed_fn))
