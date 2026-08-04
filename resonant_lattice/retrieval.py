import json
import logging
import time
import urllib.request
import re
from typing import List, Dict, Optional

from store_common import ensure_plugin_on_path
ensure_plugin_on_path()

from store import LatticeStore, serialize_vector

logger = logging.getLogger(__name__)


# ── Lexical hybrid-recall helpers (P2a precision / relevance-tier) ───────────────────────────
# SIGNIFICANT tokens = the rare/specific terms that disambiguate CONTEXT but often embed poorly:
# numbers+units ("3-inch", "0.2mm", "5432", "3.3v") and distinctive words. Overlap on these BOOSTS
# a fact's contextual relevance over a semantic-only neighbour (batt_3in over batt_7in for a
# "3-inch" query), lifting top-1 ranking. Cheap, deterministic, no extra deps.
_STOPWORDS = frozenset((
    "the a an and or but if then else for to of in on at by with from into over under is are "
    "was were be been being do does did have has had this that these those it its my your his "
    "her our their what which who whom how why when where should would could can will just want "
    "need help me i you we they about more most some any new old run set use using used get got "
    "make made like also there here as so not no yes per do you"
).split())


def _significant_tokens(text: str) -> set:
    out = set()
    for t in re.findall(r"[a-z0-9][a-z0-9.\-]*", (text or "").lower()):
        if any(ch.isdigit() for ch in t):          # numbers / specs: 3-inch, 0.2mm, 5432, 3.3v
            out.add(t)
        elif len(t) >= 4 and t not in _STOPWORDS:   # distinctive words
            out.add(t)
    return out


def _cosine(u, v) -> float:
    """Cosine over two stored embeddings. Both come from semantic_vec, so they are the
    same model and dimension; a length mismatch means a degraded index and scores 0
    rather than raising, because a redundancy filter must never break recall."""
    if not u or not v or len(u) != len(v):
        return 0.0
    du = sum(x * x for x in u) ** 0.5
    dv = sum(x * x for x in v) ** 0.5
    if du == 0.0 or dv == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(u, v)) / (du * dv)


def _relevance_tier(score: float) -> str:
    """Coarse, agent-facing contextual-relevance label (A22) - distinct from resonance/strength."""
    if score >= 0.60:
        return "high"
    if score >= 0.40:
        return "related"
    return "weak"


class LatticeRetriever:
    """Hybrid semantic + keyword retrieval with neuroplastic metadata."""

    # Phase 2: maximum additive cosine-distance penalty applied to a fully-stale
    # fact when ranking. Deliberately small - this only breaks near-ties so a
    # fresh near-match can edge out a long-unconfirmed strong one; it is a gentle
    # nudge, never a hard filter (stale facts are never dropped from recall).
    FRESHNESS_MAX_NUDGE = 0.08

    def __init__(self, store: LatticeStore, ollama_endpoint: str, embed_model: str,
                 min_similarity: float = 0.30, freshness_halflife: float = 0.0,
                 embed_timeout: float = 30.0, embed_keep_alive: str = "10m",
                 rank_fusion_k: float = 0.0):
        self.store = store
        self.ollama_endpoint = ollama_endpoint
        self.embed_model = embed_model
        # Embedding HTTP timeout. Default 30s (was a hardcoded 5s) so a COLD networked
        # embedder - e.g. a small GPU that idle-unloaded the model - has time to load on
        # the first call (observed ~6s cold vs ~0.6s warm); a 5s ceiling silently dropped
        # facts at consolidation time. embed_keep_alive is sent to Ollama so the model stays
        # resident between turns (far fewer cold loads); "" disables the hint.
        self.embed_timeout = float(embed_timeout)
        self.embed_keep_alive = embed_keep_alive
        # Recall floor: vector hits below this cosine similarity are discarded.
        # Deliberately much looser than store.similarity_threshold (a dedup
        # gate) - this only blocks clearly-unrelated k-NN filler, which would
        # otherwise be injected into context AND recall-reinforced every turn.
        self.min_similarity = float(min_similarity)
        # Phase 2 freshness nudge: cycles for a fact's freshness to halve. 0
        # disables the nudge entirely (pure similarity ranking). Cycle-driven -
        # read from last_confirmed_cycle vs the meta memory_cycle, never wall-clock.
        self.freshness_halflife = float(freshness_halflife)
        # Reciprocal rank fusion constant. 0 = OFF, which is the ordering every profile
        # has today. When set (60 is the paper's value and the usual default), results are
        # ordered by fused RANK across the vector and keyword channels instead of by the
        # additive relevance score - because that score cannot rank a keyword-only hit at
        # all: it is floored at min_similarity and can gain at most keyword_weight, a
        # ceiling below what ordinary vector hits already score. `_score` itself is
        # untouched, so tiers, relevance_margin and the redundancy gate are unaffected.
        self.rank_fusion_k = float(rank_fusion_k)

    @staticmethod
    def _freshness_penalty(staleness: float, halflife: float,
                           max_nudge: float = FRESHNESS_MAX_NUDGE) -> float:
        """Gentle additive cosine-distance penalty for a stale fact (Phase 2).

        Returns 0 when the fact is fresh (staleness <= 0) or the nudge is off
        (halflife <= 0); otherwise rises smoothly toward max_nudge as staleness
        grows: freshness = 0.5**(staleness/halflife); penalty = max_nudge*(1-freshness).
        Bounded in [0, max_nudge) - presentation/ranking only, never a filter.
        """
        if halflife <= 0 or staleness <= 0:
            return 0.0
        freshness = 0.5 ** (staleness / halflife)
        return max_nudge * (1.0 - freshness)

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Fetch embedding from Ollama. Called lock-free.
        Basic retry with exponential backoff (Phase 7 resilience).
        """
        last_err = None
        for attempt in range(3):  # 3 attempts
            try:
                payload = {"model": self.embed_model, "prompt": text}
                if self.embed_keep_alive:
                    payload["keep_alive"] = self.embed_keep_alive
                req = urllib.request.Request(
                    f"{self.ollama_endpoint}/api/embeddings",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=self.embed_timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    return result.get("embedding")
            except Exception as e:
                last_err = e
                if attempt < 2:
                    sleep = 0.5 * (2 ** attempt)
                    logger.debug("Embedding fetch attempt %d failed (%s); retrying in %.1fs", attempt+1, e, sleep)
                    time.sleep(sleep)
        logger.debug("Embedding fetch failed after retries: %s", last_err)
        return None

    def _drop_redundant(self, scored: List[Dict], ceiling: float, limit: int) -> List[Dict]:
        """Greedy redundancy filter over an already-ranked list.

        Reads each candidate's STORED embedding (semantic_vec) rather than re-embedding,
        so this adds SQL lookups and dot products, never a model call. A candidate whose
        cosine to any already-admitted result is >= ceiling is held back; anything whose
        embedding is unavailable is admitted unchanged, so a degraded vec index can never
        cause a fact to silently vanish from recall.
        """
        ids = [h["id"] for h in scored[:max(limit * 4, limit)]]
        if len(ids) < 2:
            return scored
        vecs = {}
        try:
            # store._lock is an RLock and search() already holds it, so the store's own
            # tested read-back is safe to re-enter here - no second unpack to keep in
            # sync with serialize_vector's float32 packing.
            for fid in ids:
                v = self.store.get_fact_embedding(fid)
                if v:
                    vecs[fid] = v
        except Exception as e:
            logger.debug("Redundancy gate: embeddings unavailable (%s); ranking unchanged", e)
            return scored
        kept, held, chosen = [], [], []
        for h in scored:
            v = vecs.get(h["id"])
            if v is None or all(_cosine(v, c) < ceiling for c in chosen):
                kept.append(h)
                if v is not None:
                    chosen.append(v)
            else:
                held.append(h)
            if len(kept) >= limit:
                break
        # Held-back rows are APPENDED, never discarded: this reorders recall, it does
        # not censor it, so a caller asking for k still gets k when the corpus has k.
        return kept + held

    def search(self, query: str, limit: int = 8, *,
               keyword_weight: float = 0.25, relevance_margin: "float | None" = None,
               dormant_floor: "float | None" = None, strong_cue: float = 0.6,
               redundancy_ceiling: "float | None" = None) -> List[Dict]:
        """Hybrid search: vector + FTS5 keyword, unified re-rank by CONTEXTUAL RELEVANCE.

        Both passes always run (keyword catches exact technical names that embed poorly). Candidates
        are scored on one axis - relevance = cosine (minus a gentle freshness nudge) + a
        ``keyword_weight`` boost for matching the query's SIGNIFICANT tokens - and ranked by it,
        with resonance as the tiebreak (two-axis: "useful now" = relevant AND established).
        ``relevance_margin`` (opt-in) drops candidates whose score is more than ``margin`` below the
        top - the adaptive precision gate (A6). Each result carries ``relevance`` + ``relevance_tier``
        (A22). Default args preserve the full ranked list (no gate) for callers that expect it.
        """
        if not query:
            return []

        # ── Fetch embedding lock-free (network I/O) ───────────────────
        # In degraded mode (embedding-dim mismatch) the vec table can't be
        # queried with current-dim vectors, so skip embedding entirely and let
        # the FTS keyword path serve results.
        query_vec = None if self.store.degraded else self._get_embedding(query)
        vec_bytes = serialize_vector(query_vec) if query_vec else None

        # ── All SQL work inside the lock ──────────────────────────────
        with self.store._lock:
            results_dict: Dict[int, Dict] = {}
            # Each channel's OWN ranking, kept for reciprocal-rank fusion. Recorded even
            # when fusion is off: it costs two dict writes and makes the axis observable.
            vec_rank: Dict[int, int] = {}
            fts_rank: Dict[int, int] = {}
            # 0 = off, preserving the exact ordering every existing profile has today.
            # 60 is the constant from the original RRF paper and the usual default; it is
            # deliberately large so that differences deep in a list stay small.
            rank_fusion_k = float(getattr(self, "rank_fusion_k", 0) or 0)

            # 1. Primary: Vector Search
            if vec_bytes:
                vec_sql = """
                    SELECT f.id, f.content, f.category, f.tier,
                           f.resonance_count, f.conflict_group_id, f.conflict_since_cycle,
                           f.source_session, f.source_ref, f.last_confirmed_cycle,
                           f.learned_at_cycle, f.max_resonance_seen, f.pinned, v.distance
                    FROM semantic_vec v
                    JOIN semantic_facts f ON f.id = v.id
                    WHERE v.embedding MATCH ? AND k = ?
                      AND f.tier != 'superseded'
                    ORDER BY v.distance
                """
                try:
                    vec_rows = self.store._conn.execute(vec_sql, (vec_bytes, limit)).fetchall()
                    max_dist = 1.0 - self.min_similarity
                    for row in vec_rows:
                        if row["distance"] is not None and row["distance"] > max_dist:
                            continue  # below recall floor - k-NN filler, not a match
                        results_dict[row["id"]] = dict(row)
                        vec_rank.setdefault(row["id"], len(vec_rank) + 1)
                except Exception as e:
                    logger.error("Vector search failed: %s", e)

            # 2. FTS5 keyword pass - always runs (hybrid union, not a
            #    fallback): catches exact technical names that embed poorly.
            fts_sql = """
                SELECT f.id, f.content, f.category, f.tier,
                       f.resonance_count, f.conflict_group_id, f.conflict_since_cycle,
                       f.source_session, f.source_ref, f.last_confirmed_cycle,
                       f.learned_at_cycle, f.max_resonance_seen, f.pinned, fts.rank
                FROM semantic_fts fts
                JOIN semantic_facts f ON f.id = fts.rowid
                WHERE semantic_fts MATCH ?
                  AND f.tier != 'superseded'
                ORDER BY fts.rank LIMIT ?
            """
            try:
                # Strip FTS5 control chars but keep hyphens and dots for technical names
                clean_query = re.sub(r'["*^()+\[\]{}]', '', query).strip()

                if clean_query:
                    # Attempt 1: phrase search
                    fts_rows = self.store._conn.execute(
                        fts_sql, (f'"{clean_query}"', limit)
                    ).fetchall()

                    # Attempt 2: AND-token search
                    if not fts_rows:
                        tokens = [t for t in clean_query.split() if len(t) > 1]
                        if tokens:
                            # Quote every token: '-', '.', ':' are FTS5 syntax
                            # (NOT/column-filter), so unquoted technical names
                            # ("granite-4.1-30b", "localhost:8174") are syntax
                            # errors. Quoting makes them literal terms.
                            and_query = " AND ".join(f'"{t}"' for t in tokens)
                            fts_rows = self.store._conn.execute(
                                fts_sql, (and_query, limit)
                            ).fetchall()

                    # Attempt 3: OR over SIGNIFICANT tokens.
                    #
                    # Without this the keyword half of the hybrid is dead for anything
                    # shaped like a question. Attempt 1 needs the whole sentence to appear
                    # verbatim; attempt 2 needs EVERY token present, stopwords included,
                    # and FTS5's default tokenizer does not stem - so "encoded" will not
                    # match a fact that says "encoding".
                    #
                    # MEASURED on a 34,474-fact corpus, nine natural-language questions:
                    # attempts 1 and 2 returned ZERO rows for every single one. Retrieval
                    # was therefore dense-only in practice, exactly where the docstring
                    # above promises a union. Dropping stopwords and OR-ing what is left
                    # put the wanted fact in the FTS top-25 for 8 of the 9 - including the
                    # one question the vector pass misses outright, where the answer sits
                    # at cosine rank 197 of 34,474 but ranks 20th by keyword.
                    #
                    # Safe by construction: it only runs when the stricter attempts found
                    # NOTHING, so no query that works today changes, and FTS-only hits
                    # enter the union at the recall floor - they cannot outrank a strong
                    # vector hit, only rescue a query that had no keyword candidates.
                    # Reuses _significant_tokens - the SAME helper the keyword boost below
                    # scores on - so the terms that FIND a candidate are the terms that
                    # then rank it, rather than two different notions of "significant".
                    if not fts_rows:
                        significant = sorted(_significant_tokens(clean_query))
                        if significant:
                            or_query = " OR ".join(f'"{t}"' for t in significant)
                            fts_rows = self.store._conn.execute(
                                fts_sql, (or_query, limit)
                            ).fetchall()

                    for row in fts_rows:
                        fid = row["id"]
                        fts_rank.setdefault(fid, len(fts_rank) + 1)
                        if fid not in results_dict:
                            results_dict[fid] = dict(row)

            except Exception as e:
                logger.debug("FTS5 search issue: %s", e)

            # 3. Unified hybrid re-rank on ONE contextual-relevance axis.
            #    relevance = cosine (1 - distance, minus a gentle freshness nudge for vector hits;
            #    FTS-only keyword hits start at the recall floor) + a keyword boost on the query's
            #    significant-token overlap. Resonance breaks ties. Optional margin gate (precision).
            q_sig = _significant_tokens(query)
            cur_cycle = self.store._current_memory_cycle() if self.freshness_halflife > 0 else 0
            scored: List[Dict] = []
            for r in results_dict.values():
                dist = r.get("distance")
                if dist is not None:
                    base = 1.0 - float(dist)
                    if self.freshness_halflife > 0:
                        lc = r.get("last_confirmed_cycle")
                        stale = (cur_cycle - lc) if lc is not None else 0
                        base -= self._freshness_penalty(stale, self.freshness_halflife)
                else:
                    base = float(self.min_similarity)   # FTS-only (keyword) baseline
                # Authoritative-fact preference: a PINNED fact (identity-level,
                # never-forget) gets a small bounded relevance bump so a curated
                # policy ranks above merely query-matching content (e.g. a
                # query-optimized poison). Bounded - it reorders among candidates
                # that already cleared the floor; it never surfaces an irrelevant
                # pin. The [PRIORITY] marker in the recall block remains the hard
                # trust signal regardless of rank (A/B-validated: it makes the
                # agent obey the pinned rule 15/15 vs 8-13/15 for none/[PINNED]).
                if r.get("pinned"):
                    base += 0.10
                boost = 0.0
                if keyword_weight and q_sig:
                    f_sig = _significant_tokens(r.get("content", ""))
                    if f_sig:
                        boost = float(keyword_weight) * (len(q_sig & f_sig) / len(q_sig))
                r["_score"] = base + boost
                # Reciprocal rank fusion, as an ORDERING only. `_score` above stays exactly
                # as it was: the tier labels, relevance_margin and the redundancy gate all
                # read it, and it is the only ABSOLUTE signal in the system - the sourdough
                # sentinel is caught precisely because its best hit scores 0.52, below the
                # 0.75 gate. Set-relative schemes (min-max, z-score) would map that same
                # hit to 1.0 and destroy the distinction, which is why they are not used
                # here despite producing a tidier-looking number.
                #
                # WHAT THIS FIXES. `_score` cannot rank a keyword-only hit at all: such a
                # hit is floored at min_similarity and can gain at most keyword_weight, so
                # on a profile with min_similarity 0.30 its ceiling is 0.55 while ordinary
                # vector hits score ~0.77. Measured: the one question whose answer the
                # vector pass misses (cosine rank 197 of 34,474) is found by the keyword
                # pass at rank 20, scores 0.44, and is discarded every time. RRF compares
                # POSITIONS instead of scores, so "20th by keyword" can beat "3rd by
                # vector" - which is the entire reason two channels are run.
                fid_ = r.get("id")
                r["_rrf"] = ((1.0 / (rank_fusion_k + vec_rank[fid_]) if fid_ in vec_rank else 0.0)
                             + (1.0 / (rank_fusion_k + fts_rank[fid_]) if fid_ in fts_rank else 0.0))
                scored.append(r)
            # Two-axis ranking: contextual relevance first, resonance (established strength) tiebreak.
            # With fusion enabled the ORDER comes from the fused rank instead, relevance falling
            # back to a tiebreak; every downstream consumer still reads the same `_score`.
            if rank_fusion_k > 0:
                scored.sort(key=lambda h: (h["_rrf"], h["_score"],
                                           float(h.get("resonance_count") or 0.0)),
                            reverse=True)
            else:
                scored.sort(key=lambda h: (h["_score"], float(h.get("resonance_count") or 0.0)),
                            reverse=True)
            # Buried-but-pluckable (P2b): a DORMANT fact (resonance below dormant_floor - decayed,
            # long un-recalled) is withheld from normal recall UNLESS a STRONG contextual cue
            # (score >= strong_cue) resurfaces it - recognition vs recall, the "intentional
            # relevance" pluck. Off when dormant_floor is None (preserves current behavior).
            if dormant_floor is not None:
                kept = []
                for h in scored:
                    res = float(h.get("resonance_count") or 0.0)
                    if res >= float(dormant_floor):
                        kept.append(h)
                    elif h["_score"] >= float(strong_cue):
                        h["_dormant"] = True   # plucked from dormancy by a strong cue
                        kept.append(h)
                scored = kept
            # Adaptive precision gate (opt-in): drop candidates well below the top relevance.
            if relevance_margin is not None and scored:
                cutoff = scored[0]["_score"] - float(relevance_margin)
                scored = [h for h in scored if h["_score"] >= cutoff]

            # REDUNDANCY GATE (opt-in). Relevance ranking alone returns the most
            # on-topic facts, and in a corpus that deliberately holds many sources on
            # one topic that means the top of the list is many phrasings of one answer.
            # Measured on a 6,983-fact clinical corpus: asking "there's blood in his
            # pee" returned five restatements of "hematuria means blood in the urine"
            # at ranks 1-5, and five more at 6-10.
            #
            # The tempting fix is to MERGE those facts. The census says otherwise:
            # 16,432 pairs sit above the store's own 0.78 similarity_threshold and 65%
            # of them carry DIFFERENT numbers, so 0.78 is a TOPIC threshold here, not a
            # duplicate one - three laboratories' reference intervals are three facts.
            # Merging is destructive and irreversible; skipping a redundant neighbour at
            # READ time costs nothing and keeps every source.
            #
            # Greedy: walk the ranked list, admit a candidate only when it is below
            # `redundancy_ceiling` cosine to everything already admitted. Off (None)
            # preserves the exact previous ordering.
            if redundancy_ceiling is not None and scored:
                scored = self._drop_redundant(scored, float(redundancy_ceiling), limit)

            final_results: List[Dict] = []
            for hit in scored[:limit]:
                hit["relevance"] = round(float(hit["_score"]), 3)
                hit["relevance_tier"] = _relevance_tier(hit["_score"])
                hit.pop("_score", None)
                if hit.pop("_dormant", False):
                    hit["dormant"] = True   # surfaced from dormancy by a strong cue (A22)
                self._sanitize_fact(hit)
                final_results.append(hit)
            return final_results

    def _sanitize_fact(self, fact: Dict) -> None:
        """Strip internal / non-serialisable fields before returning to agent."""
        for field in ("distance", "rank", "hrr_vector", "embedding", "__rowid__", "rowid"):
            fact.pop(field, None)
        fact.setdefault("resonance_count", 1)
        fact.setdefault("conflict_group_id", None)
        fact.setdefault("tier", "short")
        # resonance_count is REAL in the DB (decay produces values like 2.875).
        # Round for the model-facing payload so recall blocks read cleanly;
        # stored precision is untouched.
        try:
            fact["resonance_count"] = round(float(fact["resonance_count"]), 2)
        except (TypeError, ValueError):
            pass
        # A22 confidence picture (P4b): surface the PEAK ('was this ever important' -
        # max_resonance_seen) and the ENTRY cycle ('how long has it been known' -
        # learned_at_cycle) alongside current strength, so the agent reasons over
        # uncertainty, not just present resonance. Round the peak like resonance.
        if "max_resonance_seen" in fact:
            try:
                fact["peak_resonance"] = round(float(fact["max_resonance_seen"]), 2)
            except (TypeError, ValueError):
                fact["peak_resonance"] = None
            fact.pop("max_resonance_seen", None)
        # pinned (A5): identity-level / never-forget. Present as a clean bool.
        if "pinned" in fact:
            fact["pinned"] = bool(fact.get("pinned"))


class BlindRetriever(LatticeRetriever):
    """Tier-1 blind recall: homomorphic cosine over encrypted embeddings (E2.3).

    The store keeps only CKKS ciphertext (``semantic_he``); recall runs as: the
    client encrypts the query vector under the public key -> the store scores it
    against every stored ciphertext homomorphically WITHOUT decrypting
    (``EvalInnerProduct``, eval keys only) -> the client decrypts the scalar scores
    and ranks them top-k. Mirrors ``LatticeRetriever.search``'s contract (returns
    sanitized fact dicts, applies the same ``min_similarity`` floor, excludes
    superseded facts) but over ``semantic_he`` instead of the plaintext
    ``semantic_vec`` index, reusing the inherited ``_get_embedding`` /
    ``_sanitize_fact``.

    ``blind`` is an HE CLIENT instance - on the live path a ``he_crypto.BlindRecallPRE``
    (cosine + PRE), holding the secret key unwrapped from the keystore under the master
    passphrase. It is DUCK-TYPED - the only methods used are ``encrypt_unit_vector`` /
    ``cosine_score`` / ``decrypt_score`` - so it is the single seam that needs openfhe;
    BlindRetriever itself carries no HE import. On a host without openfhe a real client
    cannot be constructed, so blind recall is naturally unavailable there while the store
    side (``store_blind``) stays dependency-free.

    HRR recall (``blind_hrr_scores`` / ``blind_hrr_search`` over ``semantic_he_hrr``) uses a
    SEPARATE ``blind_hrr`` client sized for the 2·hrr_dim lift (Option A multi-keyset tier) -
    distinct from the embed-dim ``blind`` recall client; see ``__init__``. There is no FTS5
    half here: keyword search over ciphertext is impossible, so blind recall is vector/HRR-only
    (roadmap §7.4 / E7 drop-FTS). Ranking is client-side, which leaks only the fact COUNT
    (§7.3). The same client logically does both the (store-side) scoring and the (client-side)
    decrypt here; the store-cannot-decrypt property was proven 3-process in the E2 core (see
    he_crypto)."""

    def __init__(self, store: LatticeStore, ollama_endpoint: str, embed_model: str,
                 blind, min_similarity: float = 0.30, blind_hrr=None,
                 scan_batch: int = 0, scan_concurrency: int = 1, gpu_backend=None):
        super().__init__(store, ollama_endpoint, embed_model,
                         min_similarity=min_similarity)
        self.blind = blind
        # Optional GPU blind-recall accelerator (openfhe-gpu-backend). When provided AND
        # available, blind_scores routes the homomorphic cosine scan through the native
        # FIDESlib/OpenFHE GPU scorer; otherwise the CPU streaming scan below runs unchanged.
        # None (the default) keeps behavior byte-identical to the CPU-only path.
        self._gpu_backend = gpu_backend
        # A1 streaming-scan sizing: scan_batch 0 = auto (size from measured per-ct footprint +
        # detected RAM in _resolve_scan_batch), a positive value pins a fixed batch;
        # scan_concurrency tells the auto-tuner how many blind recalls may run at once so each
        # gets a proportional RAM share. Bounds the recall scan's memory instead of the old
        # fetchall-the-whole-corpus behavior (see store_blind.stream_he_vectors).
        self._scan_batch = int(scan_batch)
        self._scan_concurrency = max(1, int(scan_concurrency))
        # Separate HE client for the HRR (cos,sin) lift over ``semantic_he_hrr`` (Option A:
        # sized 2·hrr_dim, a DISTINCT context from the embed-dim ``blind`` recall client). The
        # HRR methods MUST use this, never ``self.blind`` - encrypting a 2·hrr_dim lift with the
        # embed-dim recall context is a dimension/context mismatch. Falls back to ``blind`` when
        # not given (single-client callers / a context sized for both, e.g. older tests).
        self.blind_hrr = blind_hrr if blind_hrr is not None else blind

    def search(self, query: str, limit: int = 8) -> List[Dict]:
        """Override the plaintext hybrid search with homomorphic vector-only recall, so the
        provider's recall path (``_compute_prefetch`` -> ``retriever.search``) transparently
        runs blind when ``encryption_mode=blind``. There is no FTS half over ciphertext
        (roadmap §7.4 / E7); blind recall is vector-only."""
        return self.blind_search(query, limit=limit)

    def blind_search(self, query: str, limit: int = 8) -> List[Dict]:
        """Blind recall for a text query (embeds via Ollama, then blind_search_vec)."""
        if not query:
            return []
        query_vec = self._get_embedding(query)
        if not query_vec:
            return []
        return self.blind_search_vec(query_vec, limit=limit)

    def blind_search_vec(self, query_vec: List[float], limit: int = 8) -> List[Dict]:
        """Blind recall from a precomputed query embedding (no Ollama dependency).

        Split out so deterministic fixtures (and the E2.4 comparison) can drive
        recall without the embed service."""
        scored = self.blind_scores(query_vec)
        scored.sort(key=lambda t: t[1], reverse=True)   # cosine descending
        return self._materialize_blind(scored, limit)

    def _resolve_scan_batch(self, table: str) -> int:
        """Auto-size the streaming blind-recall batch (A1) from the MEASURED per-ct footprint
        and DETECTED available RAM (``store_blind.resolve_scan_batch``), honoring an explicit
        ``self._scan_batch`` override. Portable: adapts to the host, nothing hardcoded."""
        import store_blind as _sb
        return _sb.resolve_scan_batch(
            self._scan_batch,
            self.store.he_blob_size(table),
            self.store.count_he_vectors(table),
            _sb.detect_available_ram_bytes(),
            self._scan_concurrency,
        )

    def blind_scores(self, query_vec: List[float]) -> List[tuple]:
        """``[(fact_id, decrypted_cosine), …]`` over every stored ciphertext.

        Encrypt the query under the public key -> homomorphic ``EvalInnerProduct`` per stored
        ct (store side, eval keys only, never decrypts) -> client decrypts each scalar score.
        The scan STREAMS the ciphertexts in auto-sized batches (A1: bounded RAM instead of
        materializing the whole corpus) and deserializes the query ONCE up front, so a
        full-corpus recall no longer holds every ciphertext in memory at the same time.

        When a GPU backend is configured and available, the whole scan is offloaded to the
        native scorer (returning the same ``[(id, score), …]`` shape); any failure there is
        logged and falls through to the CPU scan, so the GPU is a pure accelerator."""
        if self._gpu_backend is not None:
            try:
                if self._gpu_backend.available():
                    return self._gpu_backend.scores(query_vec)
            except Exception as e:
                logger.warning("GPU blind backend failed (%s); falling back to CPU scan", e)
        table = "semantic_he"
        q_blob = self.blind.encrypt_unit_vector(query_vec)
        q_prep = self.blind.prepare_query(q_blob) if hasattr(self.blind, "prepare_query") else q_blob
        batch = self._resolve_scan_batch(table)
        scored = []
        for fid, ct in self.store.stream_he_vectors(table, batch=batch):
            scored.append((fid, self.blind.decrypt_score(self.blind.cosine_score(q_prep, ct))))
        return scored

    def _materialize_blind(self, scored: List[tuple], limit: int) -> List[Dict]:
        """Fetch + sanitize fact rows for a score-descending ``[(id, score), …]`` list.

        Stops once ``limit`` non-superseded facts are collected; ``break``s at the
        ``min_similarity`` floor (safe because the list is sorted descending).
        Superseded facts are filtered for parity with the plaintext path; the
        decrypted cosine is kept as ``blind_similarity`` for inspection."""
        results: List[Dict] = []
        with self.store._lock:
            for fid, score in scored:
                if score < self.min_similarity:
                    break
                row = self.store._conn.execute(
                    "SELECT id, content, category, tier, resonance_count, "
                    "conflict_group_id, conflict_since_cycle, source_session, source_ref, "
                    "last_confirmed_cycle, learned_at_cycle, max_resonance_seen, pinned "
                    "FROM semantic_facts "
                    "WHERE id = ? AND tier != 'superseded'",
                    (int(fid),),
                ).fetchone()
                if row is None:
                    continue   # missing or superseded - skip, keep filling top-k
                fact = dict(row)
                fact["blind_similarity"] = round(float(score), 6)
                self._sanitize_fact(fact)
                results.append(fact)
                if len(results) >= limit:
                    break
        return results

    def blind_hrr_scores(self, probe_phases) -> List[tuple]:
        """``[(fact_id, hrr_similarity), …]`` over the encrypted HRR lifts (semantic_he_hrr).

        Lifts the plaintext phase probe to ``(cos, sin)/√dim`` CLIENT-SIDE
        (``holographic.hrr_lift``), encrypts it under the public key, and homomorphically
        scores it against every stored HRR-lift ciphertext - the inner product of two lifts
        IS the HRR phase-cosine similarity (E4 4a), so this is blind conflict / fuzzy content
        scoring with the SAME cosine primitive embeddings use. Uses ``self.blind_hrr`` (the
        2·hrr_dim HRR context, distinct from the embed-dim recall ``self.blind``). The store
        never sees the phases."""
        import holographic as _hg
        table = "semantic_he_hrr"
        lift = _hg.hrr_lift(probe_phases)
        q_blob = self.blind_hrr.encrypt_unit_vector(lift.tolist())
        q_prep = self.blind_hrr.prepare_query(q_blob) if hasattr(self.blind_hrr, "prepare_query") else q_blob
        batch = self._resolve_scan_batch(table)
        scored = []
        for fid, ct in self.store.stream_he_vectors(table, batch=batch):
            scored.append((fid, self.blind_hrr.decrypt_score(self.blind_hrr.cosine_score(q_prep, ct))))
        return scored

    def blind_hrr_search(self, probe_phases, limit: int = 8) -> List[Dict]:
        """Blind HRR recall: score the phase probe against the encrypted HRR lifts, rank
        descending, and materialize the top-``limit`` non-superseded facts (like
        ``blind_search_vec`` but over ``semantic_he_hrr`` with HRR similarity)."""
        scored = self.blind_hrr_scores(probe_phases)
        scored.sort(key=lambda t: t[1], reverse=True)
        return self._materialize_blind(scored, limit)


class BlindWriter:
    """Tier-1 blind WRITE (roadmap 0b/E4): encrypt a fact's unit vector client-side and persist
    ONLY the ciphertext to the store. ``table``-parametrized - ``semantic_he`` for the embedding
    (E2), ``semantic_he_hrr`` for the HRR (cos,sin) lift (E4). The plaintext never reaches the
    blind store through this path - encryption happens here under the public key
    (``blind.encrypt_unit_vector``), and the store keeps an opaque blob (``store_he_vector``).

    ``blind`` is a duck-typed he_crypto client (``BlindRecallPRE``/``BlindCrypto``) exposing
    ``encrypt_unit_vector`` - the only HE seam, so BlindWriter carries no openfhe import. The
    provider drives it from the consolidation/ingest path when ``encryption_mode=blind``:
    after a fact is added (id known), ``write_fact(fact_id, embedding)`` stores the ct.
    """

    def __init__(self, store: LatticeStore, blind, he_version: int = 1,
                 table: str = "semantic_he"):
        self.store = store
        self.blind = blind
        self.he_version = int(he_version)
        self.table = table          # semantic_he (embedding) | semantic_he_hrr (HRR lift)

    def write_fact(self, fact_id: int, vector: List[float]) -> bool:
        """Encrypt a unit ``vector`` under the public key and store its ciphertext for
        fact_id in this writer's table. For the HRR writer, pass ``holographic.hrr_lift(hrr)``.

        Returns True on success; False (logged, non-fatal) on any failure, so a blind-write
        hiccup never blocks or unwinds the plaintext write path that already committed the
        fact. A reinforced (already-stored) fact harmlessly overwrites its single ct."""
        if vector is None or len(vector) == 0 or fact_id is None or int(fact_id) <= 0:
            return False
        try:
            ct = self.blind.encrypt_unit_vector(list(vector))
            self.store.store_he_vector(int(fact_id), ct, he_version=self.he_version,
                                       table=self.table)
            return True
        except Exception as e:
            logger.warning("Blind write failed for fact %s (non-fatal): %s", fact_id, e)
            return False


class BlindMaintainer:
    """E5 5b: client-side orchestration of the BLIND dream-cycle maintenance over encrypted
    resonance (``semantic_he_meta``). The store DECAYS the encrypted resonance every cycle
    WITHOUT reading it (fully blind); the trusted client (holding the secret) SETTLES
    promotion/eviction on a visit by decrypting and thresholding in plaintext - the
    client-assisted model (roadmap §8 E5 5b). The store never sees plaintext resonance.

    ``maint`` is a duck-typed ``he_crypto.BlindMaintenance`` (``encrypt_scalars`` /
    ``serialize_ct`` / ``decay`` for the store ops, ``decrypt_scalars`` for settle), so
    BlindMaintainer carries no openfhe import. Resonance is SCALED to ~[0,1]
    (``resonance / max_resonance``) before encryption so the optional homomorphic threshold
    (``maint.ge_threshold``) stays in range; ``settle`` uses plaintext thresholds on the
    decrypted value (the recommended client-assisted path)."""

    def __init__(self, store: LatticeStore, maint, table: str = "semantic_he_meta"):
        self.store = store
        self.maint = maint
        self.table = table

    def set_resonance(self, fact_id: int, scaled_value: float) -> bool:
        """Encrypt a fact's scaled resonance (~[0,1]) and store its ciphertext. Non-fatal."""
        if fact_id is None or int(fact_id) <= 0:
            return False
        try:
            ct = self.maint.encrypt_scalars([float(scaled_value)])
            self.store.store_he_vector(int(fact_id), self.maint.serialize_ct(ct), table=self.table)
            return True
        except Exception as e:
            logger.warning("Blind resonance write failed for fact %s (non-fatal): %s", fact_id, e)
            return False

    def decay_all(self, factor: float) -> int:
        """STORE-side blind decay: multiply every stored encrypted resonance by ``factor`` and
        write it back - runs WITHOUT the secret (the store never reads the value). Returns the
        count decayed. The per-cycle forgetting, fully blind.

        ⚠️ DEPTH CONTRACT (read before wiring E5 - BlindMaintainer is NOT instantiated anywhere
        yet, so this is a forward-looking guard, not a live bug): this is IN-PLACE COMPOUNDING
        decay - it reads the current ct and re-multiplies it each cycle, spending ONE
        multiplicative level per call. It therefore requires a maint context built at the DEEP
        ``he_crypto._MAINT_DEPTH`` (~14). It MUST NOT run against the light
        ``_MAINT_BLIND_DEPTH`` (=1) keyset that ``crypto_keys.setup_or_load_blind_contexts``
        generates: a depth-1 ct survives exactly ONE decay, then the next EvalMult exhausts it and
        the value becomes garbage. That light keyset is reserved for an UNIMPLEMENTED
        decay-FROM-ORIGIN maintainer (one EvalMult of the preserved original by
        ``factor**elapsed`` off the public cycle clock); implement that before wiring the depth-1
        keyset to any decay path."""
        n = 0
        for fid, blob in self.store.iter_he_vectors(table=self.table):
            try:
                decayed = self.maint.decay(blob, factor)
                self.store.store_he_vector(fid, self.maint.serialize_ct(decayed), table=self.table)
                n += 1
            except Exception as e:
                logger.warning("Blind decay failed for fact %s (non-fatal): %s", fid, e)
        return n

    def get_resonance(self, fact_id: int) -> Optional[float]:
        """CLIENT-assisted: decrypt one fact's scaled resonance (needs the secret)."""
        blob = self.store.get_he_vector(int(fact_id), table=self.table)
        if blob is None:
            return None
        return float(self.maint.decrypt_scalars(blob, 1)[0])

    def settle(self, promote_threshold: float, prune_threshold: float) -> Dict[str, List[int]]:
        """CLIENT-assisted promotion/eviction (needs the secret): decrypt every stored resonance
        and classify into ``{'promote': [...], 'evict': [...]}`` by plaintext thresholds. The
        caller applies the decisions to the plaintext store (promote_facts / remove_fact) and may
        re-encrypt updated values via ``set_resonance``. The store never sees plaintext."""
        promote: List[int] = []
        evict: List[int] = []
        for fid, blob in self.store.iter_he_vectors(table=self.table):
            try:
                v = float(self.maint.decrypt_scalars(blob, 1)[0])
            except Exception as e:
                logger.warning("Blind settle decrypt failed for fact %s: %s", fid, e)
                continue
            if v >= promote_threshold:
                promote.append(fid)
            elif v < prune_threshold:
                evict.append(fid)
        return {"promote": promote, "evict": evict}


class BlindEntityStore:
    """E7 7b (client-side, no-leak): per-fact entity NAME sets are AEAD-encrypted at rest in
    ``semantic_he_entities`` (one opaque, RANDOMIZED blob per fact) so the untrusted store can't
    read entity names and identical sets are indistinguishable on disk - the store learns NO
    entity co-occurrence. Overlap / conflict detection are CLIENT-side ops on the decrypted sets
    (roadmap §7.4; the SSE/PSI store-side-leakage path was declined). The store-side PSI option
    is deliberately NOT built - overlap never runs on the untrusted store.

    ``encrypt_fn`` / ``decrypt_fn`` are the trusted-client binders over
    ``crypto_keys.encrypt_entities`` / ``decrypt_entities`` with the entity key, so
    BlindEntityStore carries no crypto import and is testable with a stand-in cipher."""

    def __init__(self, store: LatticeStore, encrypt_fn, decrypt_fn,
                 table: str = "semantic_he_entities"):
        self.store = store
        self._enc = encrypt_fn
        self._dec = decrypt_fn
        self.table = table

    def set_entities(self, fact_id: int, entities: List[str]) -> bool:
        """Encrypt a fact's entity-name set and store the opaque blob. Non-fatal."""
        if fact_id is None or int(fact_id) <= 0:
            return False
        try:
            self.store.store_he_vector(int(fact_id), self._enc(list(entities or [])),
                                       table=self.table)
            return True
        except Exception as e:
            logger.warning("Blind entity write failed for fact %s (non-fatal): %s", fact_id, e)
            return False

    def get_entities(self, fact_id: int) -> Optional[List[str]]:
        """CLIENT-side: decrypt one fact's entity set (None if absent)."""
        blob = self.store.get_he_vector(int(fact_id), table=self.table)
        return None if blob is None else list(self._dec(blob))

    def overlap(self, fact_id_a: int, fact_id_b: int) -> int:
        """CLIENT-side: number of shared entities between two facts (decrypts both sets)."""
        a = set(self.get_entities(fact_id_a) or [])
        b = set(self.get_entities(fact_id_b) or [])
        return len(a & b)

    def find_conflicts(self, fact_id: int, min_overlap: int = 1,
                       limit: int = 20) -> List[tuple]:
        """CLIENT-side conflict-candidate finder: facts sharing >= ``min_overlap`` entities with
        ``fact_id``, strongest first. Decrypts every stored set in TRUSTED RAM - the untrusted
        store never computes this (no co-occurrence leakage). Returns [(other_fact_id, overlap), …]."""
        target = set(self.get_entities(fact_id) or [])
        if not target:
            return []
        out: List[tuple] = []
        for fid, blob in self.store.iter_he_vectors(table=self.table):
            if fid == fact_id:
                continue
            try:
                ov = len(target & set(self._dec(blob)))
            except Exception:
                continue
            if ov >= min_overlap:
                out.append((fid, ov))
        out.sort(key=lambda t: t[1], reverse=True)
        return out[:limit]


class BlindContentStore:
    """§5-1 sealed-content mirror (client-side): a fact's CONTENT surface (content text +
    category + source quote/ref) is AEAD-encrypted at rest in ``semantic_he_content`` - one
    opaque, RANDOM-nonce blob per fact, so the untrusted store holds only ciphertext and
    identical content is indistinguishable on disk. The blind analogue of BlindEntityStore for
    the natural-language surface; the first step of §5's move to a full blind store where the
    ciphertext becomes the source of truth. Additive today: the plaintext ``semantic_facts``
    stays authoritative until the §5-4 seal.

    ``encrypt_fn`` / ``decrypt_fn`` are the trusted-client binders over
    ``crypto_keys.encrypt_sealed`` / ``decrypt_sealed`` (domain 'content') with the content key,
    so BlindContentStore carries no crypto import and is testable with a stand-in cipher. The
    canonical payload dict is assembled here, so callers pass the plaintext fields directly."""

    _FIELDS = ("content", "category", "source_quote", "source_ref")

    def __init__(self, store: LatticeStore, encrypt_fn, decrypt_fn,
                 table: str = "semantic_he_content"):
        self.store = store
        self._enc = encrypt_fn
        self._dec = decrypt_fn
        self.table = table

    def set_content(self, fact_id: int, fields: Dict) -> bool:
        """Encrypt a fact's content surface (the ``_FIELDS`` subset of its plaintext row) and store
        the opaque blob. ``fields`` is any dict/row exposing those keys (e.g. store.get_fact);
        missing keys serialize as None. Non-fatal - a blind-write hiccup never unwinds the
        already-committed plaintext write."""
        if fact_id is None or int(fact_id) <= 0:
            return False
        try:
            payload = {k: (fields.get(k) if fields else None) for k in self._FIELDS}
            self.store.store_he_vector(int(fact_id), self._enc(payload), table=self.table)
            return True
        except Exception as e:
            logger.warning("Blind content write failed for fact %s (non-fatal): %s", fact_id, e)
            return False

    def get_content(self, fact_id: int) -> Optional[Dict]:
        """CLIENT-side: decrypt one fact's content surface back to the ``_FIELDS`` dict (None if
        absent). Needs the content key (via the decrypt binder)."""
        blob = self.store.get_he_vector(int(fact_id), table=self.table)
        return None if blob is None else dict(self._dec(blob))


class BlindSealedStore:
    """§5-1b generic sealed TEXT surface (episodes / triples / summaries): store one opaque AEAD
    payload - a str or a dict - keyed by its SOURCE row id, in the given ``table``. The blind
    analogue of the plaintext episode/triple/summary text; unlike BlindContentStore (which shapes a
    fixed fact-content field set) the payload is passed through as-is, so the reconcile loop shapes
    it per surface. ``encrypt_fn`` / ``decrypt_fn`` are the trusted-client binders over
    ``crypto_keys.encrypt_sealed`` / ``decrypt_sealed`` for this surface's domain + key, so this
    carries no crypto import and is testable with a stand-in cipher."""

    def __init__(self, store: LatticeStore, encrypt_fn, decrypt_fn, table: str):
        self.store = store
        self._enc = encrypt_fn
        self._dec = decrypt_fn
        self.table = table

    def set_payload(self, row_id: int, payload) -> bool:
        """Encrypt ``payload`` (str or dict) and store the opaque blob keyed by ``row_id``.
        Non-fatal - a blind-write hiccup never unwinds the committed plaintext write."""
        if row_id is None or int(row_id) <= 0 or payload is None:
            return False
        try:
            self.store.store_he_vector(int(row_id), self._enc(payload), table=self.table)
            return True
        except Exception as e:
            logger.warning("Blind sealed write (%s) failed for row %s (non-fatal): %s",
                           self.table, row_id, e)
            return False

    def get_payload(self, row_id: int):
        """CLIENT-side: decrypt one row's sealed payload (None if absent)."""
        blob = self.store.get_he_vector(int(row_id), table=self.table)
        return None if blob is None else self._dec(blob)


class BlindVisitor:
    """§5-2 client visitor: assembles the dream cycle's per-fact WORKING SET from the §5-1 sealed
    ciphertext (content / entities / triples) plus the sealed episode/summary text, instead of the
    plaintext store - the read surface a cognition pass (clustering, conflict adjudication, gist,
    narrative) needs to run WITHOUT the plaintext content, which is the property the §5-4 seal
    requires. Read-only, trusted-client (it holds the decrypt binders via the sealed stores).

    Pre-seal it runs BESIDE the authoritative plaintext store; the §5-2 parity harness asserts it
    reconstructs the SAME working set the plaintext path sees, so re-routing a pass through it cannot
    change outcomes. Only the sealed TEXT is served from ciphertext - STRUCTURAL metadata that never
    held content (a fact's relation-id list, row ids) still reads from the plaintext ``store``, and
    those columns survive the seal. A missing surface (e.g. no entity store) yields empty, not an
    error, so the visitor degrades to whatever subset is mirrored."""

    def __init__(self, store: LatticeStore, *, content=None, entities=None, sealed=None):
        self.store = store
        self._content = content            # BlindContentStore   | None
        self._entities = entities          # BlindEntityStore    | None
        self._sealed = sealed or {}        # {episode|triple|summary: BlindSealedStore}

    def fact_view(self, fact_id: int) -> Optional[Dict]:
        """The decrypted per-fact working set - ``{content, category, source_quote, source_ref,
        entities:[...]}`` - or None if the fact's content is not mirrored. Content/quote/category
        come from ``semantic_he_content``; entities from the AEAD set (normalized). This is exactly
        what an abstraction/gist/conflict pass reads per fact, served blind."""
        if self._content is None:
            return None
        view = self._content.get_content(fact_id)
        if view is None:
            return None
        view = dict(view)
        view["entities"] = (self._entities.get_entities(fact_id) or []) if self._entities else []
        return view

    def triples(self, fact_id: int) -> List[Dict]:
        """Decrypted ``{subject, relation, object}`` triples for a fact - relation ids read
        STRUCTURALLY from the plaintext store, text decrypted from ``semantic_he_triples``."""
        ts = self._sealed.get("triple")
        if ts is None:
            return []
        out: List[Dict] = []
        for rid in self.store.relation_ids_for_fact(fact_id):
            payload = ts.get_payload(rid)
            if payload is not None:
                out.append(payload)
        return out

    def summary(self, summary_id: int) -> Optional[str]:
        """Decrypted session-summary text (None if absent / not mirrored)."""
        ss = self._sealed.get("summary")
        return ss.get_payload(summary_id) if ss is not None else None

    def episode(self, episode_id: int) -> Optional[Dict]:
        """Decrypted episode ``{role, content}`` (None if absent / not mirrored)."""
        es = self._sealed.get("episode")
        return es.get_payload(episode_id) if es is not None else None