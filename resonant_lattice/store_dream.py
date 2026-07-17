"""store_dream.py - DreamCycleMixin: Hebbian maintenance (decay, dwell,
promotion, conflict bleed/resolution, pruning, HRR re-encode).

Mixed into LatticeStore; relies on the composite for self._conn/_lock and
the cycle/threshold config attributes."""

import logging
import re
from typing import Callable, Dict, Optional

from store_common import hrr, _HRR_AVAILABLE, serialize_vector

# ── Policy-contradiction detection (entity-less, opposite-polarity) ──────────
# Content-similarity + entity overlap cannot pair two policies that say the
# opposite thing in different words ("never auto-approve" vs "auto-approval is
# enabled"): their embedding similarity is ~0 and they carry no entities. This
# deterministic lexical path flags such pairs by SHARED ACTION TOKEN + OPPOSITE
# STANCE (tighten vs loosen). Conservative + auditable (explicit word lists, no
# fuzzy scoring); gated to policy-like facts to keep precision high.
_POLICY_STOP = {
    "the", "a", "an", "is", "are", "be", "to", "of", "for", "and", "or", "on", "in",
    "any", "all", "now", "with", "that", "this", "it", "you", "your", "we", "our",
    "policy", "rule", "update", "note", "reminder", "heads", "fyi", "needed", "need",
    "do", "does", "new", "via", "per", "use", "used", "must", "may", "can", "will",
    "should", "every", "each", "into", "over", "under", "than", "then", "but", "not",
    "no", "longer", "never", "always", "require", "required", "allowed", "enabled",
    "optional", "fine", "acceptable", "exempt", "freely", "without", "skip", "skipped",
}
# Stance markers. LOOSEN takes priority (a loosening phrase signals relaxing a rule
# even when it names the requirement it removes).
_LOOSEN = ("now enabled", "is enabled", "no longer", "is fine", "are fine", "acceptable",
           "optional", "are exempt", "is exempt", "freely", "without a", "without an",
           "no human", "no approval", "not required", "can be skipped", "may export",
           "may be", "is now", "anyway", "ignore", "unnecessary")
_TIGHTEN = ("never", "must not", "must ", "always require", "require", "required",
            "forbidden", "prohibited", "not allowed", "only ", "mandatory", "second-approver")


def _policy_like(content: str, category: str) -> bool:
    c = (content or "").lower()
    return (category in ("policy", "rule", "compliance", "guardrail")
            or c.startswith("policy") or c.startswith("rule")
            or "policy:" in c or "rule:" in c)


def _policy_stance(content: str) -> str:
    t = " " + (content or "").lower() + " "
    if any(w in t for w in _LOOSEN):
        return "loosen"
    if any(w in t for w in _TIGHTEN):
        return "tighten"
    return ""


def _policy_stems(content: str) -> set:
    """Significant topic stems (action nouns/verbs), polarity words removed, so two
    policies about the SAME action share a stem regardless of phrasing."""
    out = set()
    for w in re.findall(r"[a-z]{3,}", (content or "").lower()):
        for suf in ("ation", "tion", "ions", "ing", "ed", "es", "al", "ly", "ment", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[:-len(suf)]
                break
        if len(w) >= 4 and w not in _POLICY_STOP:
            out.add(w[:6])
    return out


# ── Procedural-contradiction detection (tool-superstition sweep) ─────────────
# The dream cycle distills per-tool usage heuristics ("[web_extract] avoid X",
# "[web_search] prefer Y") as category='procedural' facts. Field finding from
# overnight curriculum runs: this layer accumulates CONTRADICTORY guidance
# ("bundle many URLs per call" vs "prefer a single URL per invocation") that
# contradicts by PARAPHRASE - and the general HRR pass excludes the category
# (template-parallel, entity-thin), so the pairs were never flagged. An agent
# recalling both flip-flops between strategies, or worse, recalls loop
# promoters ("retry the exact same failed request"). The sweep pairs facts
# about the SAME tool that share a SPECIFIC topic stem, then flags via two
# precision-first lanes (see the pass in resolve_hrr_conflicts):
#   lane 1, deterministic: opposite stance (avoid vs prefer);
#   lane 2, adjudicated:   stance-ambiguous pairs go to the injected
#                          reason-model adjudicator (capped per pass) and lock
#                          ONLY on an explicit True. Unlike the poison-policy
#                          pass this is NOT fail-open: superstition is
#                          self-inflicted, not adversarial - a missed pair
#                          waits for the next cycle; a false duel would lock a
#                          useful heuristic for nothing.
_PROC_TOOL_RE = re.compile(r"^\s*\[([a-z0-9_\-]+)\]", re.IGNORECASE)
# Stance markers. Deliberately narrow: procedural facts often carry mixed
# polarity in one sentence ("X often fails; prefer Y"), so bare "fail"/"use"
# would misclassify - ambiguity returns "" and defers to the adjudicator lane.
_PROC_AVOID = ("avoid", "do not", "don't", "never", "must not",
               "consistently fail", "consistently result", "unreliable")
_PROC_PREFER = ("prefer", "always set", "always use", "always enable",
                "recommended", "reliably succeed", "increases success",
                "has been successful", "works well", "improves")
# Stems too generic to establish that two same-tool directives concern the
# same TOPIC: tool mechanics and outcome words appear in nearly every
# heuristic, so sharing only these must not pair facts (post-stemming form,
# i.e. already suffix-stripped + truncated to 6 chars like _policy_stems).
_PROC_GENERIC_STEMS = frozenset({
    "avoid", "prefer", "recomm", "fail", "failur", "succes", "succee",
    "error", "result", "attemp", "consis", "reliab", "often", "tend",
    "work", "call", "request", "reques", "extrac", "search", "termin",
    "tool", "url", "urls", "when",
})
_PROC_ADJUDICATION_CAP = 12   # reason-model calls per sweep (cycles-not-seconds)


def _proc_tool(content: str) -> str:
    m = _PROC_TOOL_RE.match(content or "")
    return m.group(1).lower() if m else ""


def _proc_stance(content: str) -> str:
    t = " " + (content or "").lower() + " "
    avoid = any(w in t for w in _PROC_AVOID)
    prefer = any(w in t for w in _PROC_PREFER)
    if avoid == prefer:          # neither, or both -> ambiguous
        return ""
    return "avoid" if avoid else "prefer"


def _proc_topic_stems(content: str, tool: str) -> set:
    """Specific topic stems of a procedural directive: the '[tool]' prefix is
    stripped, and the tool's own name stems plus generic mechanics/outcome
    stems are removed, so only genuinely topical words (a domain, a flag, a
    strategy) can count as shared topic between two heuristics."""
    body = _PROC_TOOL_RE.sub("", content or "", count=1)
    stems = _policy_stems(body) - _policy_stems(tool.replace("_", " "))
    return stems - _PROC_GENERIC_STEMS

logger = logging.getLogger(__name__)


class DreamCycleMixin:

    def _phases_from_blob(self, blob: Optional[bytes]) -> Optional["np.ndarray"]:
        """Decode an HRR blob, guarding against dimension drift.

        Stored HRR vectors are float64 (8 bytes/element). If the blob length
        doesn't match the configured hrr_dim, the vector was encoded at a
        different dim - return None so callers skip it rather than crashing
        a whole dream cycle on a numpy shape mismatch.
        """
        if blob is None or not _HRR_AVAILABLE:
            return None
        try:
            if len(blob) != self.hrr_dim * 8:
                return None
            return hrr.bytes_to_phases(blob)
        except Exception:
            return None


    def reencode_hrr_if_needed(self, target_version: str = None) -> int:
        """One-shot re-encode of all stored HRR vectors to the current encoder/dim.

        Why: encode_fact() was upgraded from bag-of-words to encode_text_rich
        (positional + bigram aware). Old BoW vectors have systematically lower
        cross-similarity with new rich vectors, which degrades conflict
        detection and abstraction clustering during the transition. This
        recomputes hrr_vector for every fact from its stored content + entities,
        normalizing the whole store to one encoding.

        Self-gating via meta['hrr_reencoded']; runs once, then no-ops. Invoked
        off the hot path (from the dream cycle), not at startup. LLM-free.
        Returns the number of facts re-encoded (0 if skipped).
        """
        if not _HRR_AVAILABLE:
            return 0
        # FIX: gate on encoder version AND hrr_dim. A dim change must trigger a
        # fresh re-encode; previously the version-only gate stayed satisfied and
        # old-length vectors were guarded to None forever.
        if target_version is None:
            # rich-v2: encode_text_rich switched to NON-commutative rolled
            # bigrams (phase-addition bind is commutative, so v1 bigrams could
            # not distinguish "A B" from "B A"). Bumping the version makes the
            # self-gating migration re-fire once on every pre-v2 DB.
            target_version = f"rich-v2:dim{self.hrr_dim}"
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT value FROM meta WHERE key='hrr_reencoded'"
                ).fetchone()
                if row and row["value"] == target_version:
                    return 0  # already done

                facts = self._conn.execute(
                    "SELECT id, content FROM semantic_facts"
                ).fetchall()
                if not facts:
                    # Fresh store - nothing to re-encode; just stamp.
                    self._conn.execute(
                        "INSERT OR REPLACE INTO meta(key, value) VALUES ('hrr_reencoded', ?)",
                        (target_version,),
                    )
                    self._conn.commit()
                    return 0

                # Pre-fetch entity map in one query (avoid N+1).
                ent_rows = self._conn.execute(
                    """
                    SELECT fe.fact_id, e.name
                    FROM fact_entities fe
                    JOIN entities e ON e.entity_id = fe.entity_id
                    """
                ).fetchall()
                ent_map: Dict[int, list] = {}
                for r in ent_rows:
                    ent_map.setdefault(r["fact_id"], []).append(r["name"])

                updated = 0
                for f in facts:
                    try:
                        vec = hrr.encode_fact(
                            f["content"], ent_map.get(f["id"], []), dim=self.hrr_dim
                        )
                        self._conn.execute(
                            "UPDATE semantic_facts SET hrr_vector = ? WHERE id = ?",
                            (hrr.phases_to_bytes(vec), f["id"]),
                        )
                        updated += 1
                    except Exception as e:
                        logger.debug("Re-encode failed for fact %s: %s", f["id"], e)

                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES ('hrr_reencoded', ?)",
                    (target_version,),
                )
                # The store's vectors are now genuinely at self.hrr_dim - update
                # the stamped dim so _stamp_meta() stops reporting a mismatch
                # that the re-encode just fixed.
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES ('hrr_dim', ?)",
                    (str(self.hrr_dim),),
                )
                self._conn.commit()
                logger.info(
                    "HRR re-encode complete - %d facts normalized to %s encoding.",
                    updated, target_version,
                )
                return updated
            except Exception as e:
                logger.error("HRR re-encode migration failed: %s", e)
                return 0


    def reembed_if_needed(self, embed_fn, target_model: str, new_dim: "int | None" = None) -> int:
        """One-shot re-embed of all stored embeddings to a NEW embed_model (P4d migration enabler).

        Why: switching the embedding model (e.g. nomic-embed-text → embeddinggemma:300m) leaves the
        stored ``semantic_vec`` vectors in the OLD model's space, so a new-model query embedding no
        longer matches them and recall silently degrades. This recomputes every fact's embedding
        from its stored content via ``embed_fn`` (the live Ollama embedder the provider passes), and
        REBUILDS ``semantic_vec`` at the new dimension if it changed - so a model swap on an existing
        store is turnkey. The plaintext analogue of ``reencode_hrr_if_needed``; runs off the hot path
        from the dream cycle, self-gating via ``meta['embed_model']``.

        GATE (idempotent, no spurious rebuild on upgrade):
          * meta absent  → STAMP the current target and do NOTHING. First observation assumes the
            stored vectors already match the configured model (true for every pre-existing store), so
            installing this feature never triggers a needless full re-embed.
          * meta == target → no-op (already migrated).
          * meta != target → genuine switch: re-embed all, rebuild on dim change, then stamp.

        Held under ``self._lock`` for the whole pass (like the cosine rebuild) so no fact can be
        inserted at the old model/dim mid-migration. Network-bound but one-shot + background. Only
        stamps (locks in the new model) when EVERY fact re-embedded - a transient embedder outage
        leaves the gate open so the next dream cycle retries. ``embed_fn(text) -> list[float]`` (None
        on failure). Returns the number of facts re-embedded (0 if skipped)."""
        if embed_fn is None or not target_model:
            return 0
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT value FROM meta WHERE key='embed_model'"
                ).fetchone()
                stored = row["value"] if row else None
                if stored is None:
                    # First observation: assume the existing vectors match the configured model.
                    self._conn.execute(
                        "INSERT OR REPLACE INTO meta(key, value) VALUES ('embed_model', ?)",
                        (target_model,),
                    )
                    self._conn.commit()
                    return 0
                if stored == target_model:
                    return 0  # already on this model

                facts = self._conn.execute(
                    "SELECT id, content FROM semantic_facts"
                ).fetchall()
                if not facts:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO meta(key, value) VALUES ('embed_model', ?)",
                        (target_model,),
                    )
                    self._conn.commit()
                    return 0

                logger.warning(
                    "Migration: re-embedding %d facts %s → %s (embed_model change).",
                    len(facts), stored, target_model,
                )
                new_vecs = {}
                failed = 0
                for f in facts:
                    emb = None
                    try:
                        emb = embed_fn(f["content"])
                    except Exception as e:
                        logger.debug("Re-embed failed for fact %s: %s", f["id"], e)
                    if emb:
                        new_vecs[f["id"]] = emb
                    else:
                        failed += 1
                if not new_vecs:
                    # Embedder down - leave the gate OPEN (don't stamp) so we retry next cycle.
                    logger.error("Migration: re-embed produced no vectors (embedder down?) - "
                                 "will retry next dream cycle.")
                    return 0

                probe_dim = len(next(iter(new_vecs.values())))
                if probe_dim != self.vector_dim:
                    # Dimension changed (e.g. 768 → 2560): rebuild the vec0 table at the new dim.
                    logger.warning("Migration: rebuilding semantic_vec %d → %d for new embedder.",
                                   self.vector_dim, probe_dim)
                    self._conn.execute("DROP TABLE IF EXISTS semantic_vec")
                    self._conn.execute(
                        f"CREATE VIRTUAL TABLE semantic_vec USING vec0("
                        f"id INTEGER PRIMARY KEY, "
                        f"embedding float[{probe_dim}] distance_metric=cosine)"
                    )
                    self.vector_dim = probe_dim
                    self.degraded = False   # a dim-mismatch that pinned us to FTS is now resolved

                for fid, emb in new_vecs.items():
                    # vec0 virtual tables don't honor INSERT OR REPLACE on the PK; delete the
                    # existing row first (a no-op on the freshly-rebuilt dim-change table) then
                    # insert the new-model vector.
                    self._conn.execute("DELETE FROM semantic_vec WHERE id = ?", (fid,))
                    self._conn.execute(
                        "INSERT INTO semantic_vec (id, embedding) VALUES (?, ?)",
                        (fid, serialize_vector(emb)),
                    )
                if failed == 0:
                    # Only lock in the new model once EVERY fact is migrated; otherwise leave the
                    # gate open so a partial run (some embeds failed) is finished next cycle.
                    self._conn.execute(
                        "INSERT OR REPLACE INTO meta(key, value) VALUES ('embed_model', ?)",
                        (target_model,),
                    )
                self._conn.commit()
                logger.info("Migration: re-embedded %d/%d facts to %s%s.",
                            len(new_vecs), len(facts), target_model,
                            "" if failed == 0 else f" ({failed} failed - gate left open for retry)")
                return len(new_vecs)
            except Exception as e:
                logger.error("Re-embed migration failed: %s", e)
                return 0


    # ====================== HEBBIAN DREAM CYCLE OPERATIONS ======================
    def apply_cycle_decay(self, protect_conflicts: bool = False, peak_discount: float = 0.0,
                          importance_discount: float = 0.0) -> None:
        """Exponential decay: stronger memories decay slower.

        ``protect_conflicts`` holds contested facts (active conflict group) in sustained-resonance
        limbo (conflict-limbo, A9/A13). ``peak_discount`` (0..1) is surprise/importance-weighted
        retention (A11): a fact that EVER mattered - high ``max_resonance_seen``, e.g. a surprising
        one-off that entered high via ``novelty_boost``, or a reinforced fact - fades SLOWER (up to
        ``peak_discount`` less decay once its peak reaches ``promotion_threshold``), so the unique
        one-off is retained longer before going dormant. Both default off (legacy uniform decay)."""
        # Pinned facts (A5, P4a) are identity-level - exempt from decay entirely (here and in
        # staleness/prune/cap), so a never-forget fact never bleeds toward dormancy.
        where = "tier IN ('short', 'mid') AND COALESCE(pinned, 0) = 0"
        if protect_conflicts:
            where += " AND conflict_group_id IS NULL"
        # Compose the decay term from optional discount FACTORS (each <=1 slows
        # decay): base exponential, then peak/surprise discount, then importance-
        # by-category discount. Params are appended in the SAME order as the ?s.
        decay_term = "? * (1.0 / (resonance_count + 1.0))"
        params = [self.decay_per_cycle]
        if peak_discount and peak_discount > 0:
            thr = max(1.0, float(self.promotion_threshold))
            decay_term += " * (1.0 - ? * MIN(1.0, COALESCE(max_resonance_seen, 0) / ?))"
            params += [float(peak_discount), thr]
        imp_cats = getattr(self, "importance_categories", set())
        if importance_discount and importance_discount > 0 and imp_cats:
            # A high-stakes-category fact decays (1 - discount) as fast, so an
            # important-but-unused fact survives the dwell-to-long gauntlet and is
            # retained where generic noise prunes. Importance != frequency.
            cats = list(imp_cats)
            ph = ",".join("?" * len(cats))
            decay_term += f" * (1.0 - CASE WHEN category IN ({ph}) THEN ? ELSE 0.0 END)"
            params += cats + [float(importance_discount)]
        with self._lock:
            self._conn.execute(
                "UPDATE semantic_facts SET resonance_count = MAX(0, resonance_count - (" + decay_term + ")), "
                "updated_at = CURRENT_TIMESTAMP WHERE " + where,
                params,
            )
            self._conn.commit()


    def apply_staleness_decay(self, current_cycle: int, boost: float,
                              halflife: float = 50.0) -> int:
        """Phase 2 'use it or lose it' - extra decay for weak AND long-unconfirmed
        facts. Off by default (boost <= 0 ⇒ no-op).

        Targets only the facts that are already fading on both axes: short/mid
        tier (long is decay-exempt), below the promotion bar (already weak), and
        with a last_confirmed stamp older than the current cycle. The extra bleed
        ramps linearly with staleness, capped at `boost` once staleness reaches
        one half-life (linear so it needs no SQL math extension). Never drives
        resonance below 0. Cycle-driven (current_cycle vs last_confirmed_cycle),
        never wall-clock. Returns rows touched.
        """
        if boost <= 0:
            return 0
        hl = halflife if halflife and halflife > 0 else 1.0
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE semantic_facts
                SET resonance_count = MAX(0.0,
                        resonance_count - (? * MIN(1.0,
                            CAST(? - last_confirmed_cycle AS REAL) / ?))),
                    updated_at = CURRENT_TIMESTAMP
                WHERE tier IN ('short', 'mid')
                  AND COALESCE(pinned, 0) = 0
                  AND last_confirmed_cycle IS NOT NULL
                  AND (? - last_confirmed_cycle) > 0
                  AND resonance_count < ?
                """,
                (boost, current_cycle, hl, current_cycle, self.promotion_threshold),
            )
            self._conn.commit()
            return cur.rowcount or 0


    def apply_procedural_staleness_decay(self, current_cycle: int, bleed: float,
                                         grace_cycles: int = 0,
                                         categories: tuple = ('procedural',)) -> int:
        """Perishable-knowledge decay for PROCEDURAL (tool-use) facts.

        Domain facts are largely permanent, but procedural knowledge - which
        search operator works, which URL pattern to fetch - ages as tools and
        backends change. Two design facts make stale procedural advice sticky: it
        is EXCLUDED from conflict detection (so a corrected rule can't duel it
        out), and once it reaches the long tier it is decay-EXEMPT. That
        combination let a batch of superseded ``site:``-operator rules keep
        dominating recall after the backend stopped supporting them.

        This gives procedural facts a use-it-or-lose-it half-life the domain tiers
        don't need: a procedural fact not confirmed within ``grace_cycles`` bleeds
        ``bleed`` resonance each dream cycle, in ANY tier (unlike apply_cycle_decay,
        which exempts long). ``last_confirmed_cycle`` is refreshed by every
        reinforcement (add_or_reinforce_fact) - and an in-use pattern is re-distilled
        and reinforced each dream cycle - so a pattern that stays useful never
        bleeds; once a better/pinned rule wins and the stale one stops being
        re-confirmed, it fades and prunes on its own. Pinned facts and facts in an
        active conflict group are exempt. Off when ``bleed`` <= 0. Returns rows hit.
        """
        if bleed <= 0:
            return 0
        cats = [c for c in (categories or ('procedural',))] or ['procedural']
        ph = ",".join("?" * len(cats))
        with self._lock:
            cur = self._conn.execute(
                f"""
                UPDATE semantic_facts
                SET resonance_count = MAX(0.0, resonance_count - ?),
                    updated_at = CURRENT_TIMESTAMP
                WHERE category IN ({ph})
                  AND COALESCE(pinned, 0) = 0
                  AND conflict_group_id IS NULL
                  AND (? - COALESCE(last_confirmed_cycle, learned_at_cycle, ?)) > ?
                """,
                [bleed] + cats + [current_cycle, current_cycle, grace_cycles],
            )
            self._conn.commit()
            return cur.rowcount or 0


    def increment_tier_cycles(self) -> None:
        """Advance the tier-dwell counter by one dream cycle.

        Turn/cycle-based only - never wall-clock. Counts dwell for the tiers
        that can still promote (short, mid); 'long' is terminal.
        """
        with self._lock:
            self._conn.execute(
                """
                UPDATE semantic_facts
                SET cycles_in_tier = cycles_in_tier + 1
                WHERE tier IN ('short', 'mid')
                """
            )
            self._conn.commit()


    def promote_facts(self) -> None:
        """Resonance + dwell-based tier promotion.

        A fact promotes only when it has BOTH sustained resonance
        (resonance_count >= promotion_threshold) AND survived enough dream
        cycles in its current tier (cycles_in_tier >= the tier threshold).
        cycles_in_tier resets to 0 on promotion so the next tier's dwell is
        counted fresh. mid→long runs first so a fact cannot skip a tier in a
        single call (the reset also prevents this independently)."""
        with self._lock:
            # mid → long FIRST - requires mid-tier dwell satisfied
            self._conn.execute(
                """
                UPDATE semantic_facts
                SET tier = 'long', cycles_in_tier = 0, updated_at = CURRENT_TIMESTAMP
                WHERE tier = 'mid'
                  AND resonance_count >= ?
                  AND cycles_in_tier >= ?
                """,
                (self.promotion_threshold, self.mid_tier_cycles),
            )
            # short → mid - requires short-tier dwell satisfied
            self._conn.execute(
                """
                UPDATE semantic_facts
                SET tier = 'mid', cycles_in_tier = 0, updated_at = CURRENT_TIMESTAMP
                WHERE tier = 'short'
                  AND resonance_count >= ?
                  AND cycles_in_tier >= ?
                """,
                (self.promotion_threshold, self.short_tier_cycles),
            )
            self._conn.commit()

            
    def apply_conflict_decay(self, floor: float = 0.0) -> None:
        """
        Dialectic Equilibrium: Conflicting facts bleed resonance each cycle.
        If all facts in a group hit 0, the newest one is resurrected to break the tie.

        `floor` clamps the conflict bleed: with floor=0.0 (default) the original
        lethal duel is preserved; with floor>0 a flagged fact is only deprioritized
        and final resolution must come from explicit feedback or ordinary tier decay.

        NOTE: Winner-freeing happens in free_conflict_winners(), called from the
        dream cycle AFTER prune_weak_facts() has deleted the 0-resonance losers.
        """
        with self._lock:
            # 1. Bleed resonance from all active conflicts (clamped at `floor`).
            self._conn.execute(
                """
                UPDATE semantic_facts
                SET resonance_count = MAX(?, resonance_count - 1),
                    updated_at = CURRENT_TIMESTAMP
                WHERE conflict_group_id IS NOT NULL
                """,
                (floor,),
            )
 
            # 2. Tie-Breaker: if ALL facts in a group are at 0 or below,
            # resurrect the newest one (highest id) to resonance=1 so it
            # survives prune_weak_facts() and wins the duel.
            self._conn.execute(
                """
                UPDATE semantic_facts
                SET resonance_count = 1
                WHERE conflict_group_id IN (
                    SELECT conflict_group_id
                    FROM semantic_facts
                    WHERE conflict_group_id IS NOT NULL
                    GROUP BY conflict_group_id
                    HAVING MAX(resonance_count) <= 0
                )
                AND id IN (
                    SELECT MAX(id)
                    FROM semantic_facts
                    WHERE conflict_group_id IS NOT NULL
                    GROUP BY conflict_group_id
                )
                """
            )
            self._conn.commit()

            # Step 3 (free winner) deliberately removed - see free_conflict_winners()

    def free_conflict_winners(self) -> None:
        """Remove conflict_group_id from facts whose opponents have been pruned.
 
        Called from the dream cycle AFTER prune_weak_facts() so that losers
        (resonance <= 0) are already deleted. Any conflict group that now has
        only one member is resolved - that survivor is the winner and gets
        its conflict lock cleared.
 
        This was previously an unguarded _conn.execute() call in __init__.py.
        Moving it here ensures the lock is always held during the UPDATE.
        """
        with self._lock:
            self._conn.execute(
                """
                UPDATE semantic_facts
                SET conflict_group_id = NULL, conflict_since_cycle = NULL
                WHERE conflict_group_id IN (
                    SELECT conflict_group_id
                    FROM semantic_facts
                    WHERE conflict_group_id IS NOT NULL
                    GROUP BY conflict_group_id
                    HAVING COUNT(id) = 1
                )
                """
            )
            self._conn.commit()


    def get_pending_conflicts(self, min_age_cycles: int = 0, limit: int = 20) -> list:
        """Phase 6: list ACTIVE conflict groups for explicit disambiguation (read-only).

        One entry per live conflict group (>= 2 non-superseded members) with each
        competing fact's id/content/resonance/tier/last_confirmed and the group's
        age in cycles. Groups younger than min_age_cycles are omitted so the duel
        runs a little before anything is surfaced. No side effects.
        """
        with self._lock:
            cur_cycle = self._current_memory_cycle()
            rows = self._conn.execute(
                """
                SELECT id, content, category, resonance_count, tier,
                       conflict_group_id, conflict_since_cycle, last_confirmed_cycle
                FROM semantic_facts
                WHERE conflict_group_id IS NOT NULL AND tier != 'superseded'
                ORDER BY conflict_group_id, resonance_count DESC, id
                """
            ).fetchall()
        groups: Dict[str, list] = {}
        for r in rows:
            groups.setdefault(r["conflict_group_id"], []).append(r)
        out = []
        for gid, members in groups.items():
            if len(members) < 2:
                continue
            sinces = [m["conflict_since_cycle"] for m in members
                      if m["conflict_since_cycle"] is not None]
            since = min(sinces) if sinces else None
            age = (cur_cycle - since) if since is not None else None
            if min_age_cycles > 0 and (age is None or age < min_age_cycles):
                continue
            out.append({
                "conflict_group_id": gid,
                "age_cycles": age,
                "facts": [
                    {"id": m["id"], "content": m["content"], "category": m["category"],
                     "resonance_count": round(m["resonance_count"], 2),
                     "tier": m["tier"], "last_confirmed_cycle": m["last_confirmed_cycle"]}
                    for m in members
                ],
            })
            if len(out) >= limit:
                break
        return out


    def resolve_conflict(self, winner_id: int, current_cycle: Optional[int] = None,
                         winner_boost: float = 2.0) -> Optional[Dict]:
        """Phase 6: resolve a conflict explicitly (user/agent-driven).

        The winner is boosted, confirmed, and freed; the OTHER live members of its
        group are SUPERSEDED (P1 - retired to tier='superseded', superseded_by=winner),
        never deleted. Returns {winner_id, conflict_group_id, superseded:[ids]} or None
        if winner_id is not in an active conflict group. This is the explicit, auditable
        resolution path - nothing is silently overwritten beyond the existing duel.
        """
        with self._lock:
            if current_cycle is None:
                current_cycle = self._current_memory_cycle()
            row = self._conn.execute(
                "SELECT conflict_group_id FROM semantic_facts WHERE id = ?", (winner_id,)
            ).fetchone()
            if not row or row["conflict_group_id"] is None:
                return None
            gid = row["conflict_group_id"]
            losers = [
                r["id"] for r in self._conn.execute(
                    "SELECT id FROM semantic_facts "
                    "WHERE conflict_group_id = ? AND id != ? AND tier != 'superseded'",
                    (gid, winner_id),
                ).fetchall()
            ]
            if losers:
                ph = ",".join("?" * len(losers))
                self._conn.execute(
                    f"""
                    UPDATE semantic_facts
                    SET tier = 'superseded',
                        superseded_by = ?,
                        superseded_at_cycle = ?,
                        conflict_group_id = NULL,
                        conflict_since_cycle = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id IN ({ph})
                    """,
                    (winner_id, current_cycle, *losers),
                )
            # Boost + confirm + free the winner (saturating, like recall/feedback bumps).
            self._conn.execute(
                """
                UPDATE semantic_facts
                SET resonance_count = MIN(resonance_count + ?, 50.0),
                    max_resonance_seen = MAX(COALESCE(max_resonance_seen, 0),
                                             MIN(resonance_count + ?, 50.0)),
                    conflict_group_id = NULL,
                    conflict_since_cycle = NULL,
                    last_confirmed_cycle = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (winner_boost, winner_boost, current_cycle, winner_id),
            )
            self._conn.commit()
            return {"winner_id": winner_id, "conflict_group_id": gid, "superseded": losers}


    def dismiss_conflict(self, group_id: Optional[str] = None,
                         fact_id: Optional[int] = None,
                         current_cycle: Optional[int] = None,
                         confirm_bump: float = 0.5) -> Optional[Dict]:
        """Phase 6: dismiss a conflict group as a FALSE POSITIVE (members compatible).

        The second resolution verb. resolve_conflict picks a winner and retires
        every other member as superseded - which is WRONG when arbitration finds
        the facts COMPATIBLE (both true), e.g. the template-parallel detection
        false-positive class ("gl_FragCoord is available in GLSL ES 1.00/3.00"
        vs "gl_FrontFacing is available in GLSL ES 1.00/3.00"). Before this verb
        existed, an agent that correctly verified both sides had NO in-band way
        to unlock them (observed in production 2026-07-07; the operator had to
        clear conflict_group_id via SQL).

        Unlocks ALL live members of the group: clears the conflict lock, stamps
        last_confirmed_cycle (arbitration just verified them), and applies a
        small SATURATING confirm bump - symmetric across members, unlike the
        winner boost. Nothing is superseded, nothing is destroyed. Accepts the
        group id or any member's fact_id. Returns {conflict_group_id,
        dismissed:[ids]} or None when nothing matches an ACTIVE group."""
        with self._lock:
            if current_cycle is None:
                current_cycle = self._current_memory_cycle()
            gid = group_id
            if gid is None and fact_id is not None:
                row = self._conn.execute(
                    "SELECT conflict_group_id FROM semantic_facts WHERE id = ?",
                    (fact_id,),
                ).fetchone()
                gid = row["conflict_group_id"] if row else None
            if not gid:
                return None
            members = [r["id"] for r in self._conn.execute(
                "SELECT id FROM semantic_facts "
                "WHERE conflict_group_id = ? AND tier != 'superseded' ORDER BY id",
                (gid,),
            ).fetchall()]
            if not members:
                return None
            ph = ",".join("?" * len(members))
            self._conn.execute(
                f"""
                UPDATE semantic_facts
                SET conflict_group_id = NULL,
                    conflict_since_cycle = NULL,
                    resonance_count = MIN(resonance_count + ?, 50.0),
                    max_resonance_seen = MAX(COALESCE(max_resonance_seen, 0),
                                             MIN(resonance_count + ?, 50.0)),
                    last_confirmed_cycle = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id IN ({ph})
                """,
                (confirm_bump, confirm_bump, current_cycle, *members),
            )
            self._conn.commit()
            return {"conflict_group_id": gid, "dismissed": members}


    def supersede_conflict_losers(self, current_cycle: int,
                                  max_history: int = 2000) -> int:
        """Phase 1b: retire conflict losers as superseded HISTORY, not deletion.

        Run in the dream cycle BEFORE prune_weak_facts(). apply_conflict_decay()
        has already bled each active conflict group; a member at resonance <= 0 is
        a loser the prune would otherwise DELETE. For each such group the surviving
        member with the highest resonance (tie-break newest id - matching the
        resurrection bias in apply_conflict_decay) is the winner. Every loser is
        moved to a terminal tier='superseded' with superseded_by=winner and
        superseded_at_cycle=current_cycle, and its conflict_group_id is cleared so
        the group resolves (free_conflict_winners then frees the lone winner).

        Superseded rows are FROZEN: excluded from decay/promotion by tier, from
        new conflict detection (tier != 'long'), from recall by the exclusion
        sweep, and from prune_weak_facts. They are belief history, not active
        belief - no content is rewritten. Non-lethal conflict mode
        (conflict_decay_floor > 0) never drives a member to <= 0, so there are no
        losers and this is a no-op. Returns the number of facts superseded.
        """
        superseded = 0
        with self._lock:
            # Only groups that actually have a loser this cycle.
            loser_groups = [
                r["conflict_group_id"]
                for r in self._conn.execute(
                    """
                    SELECT conflict_group_id
                    FROM semantic_facts
                    WHERE conflict_group_id IS NOT NULL AND resonance_count <= 0
                    GROUP BY conflict_group_id
                    """
                ).fetchall()
            ]
            for gid in loser_groups:
                members = self._conn.execute(
                    """
                    SELECT id, resonance_count
                    FROM semantic_facts
                    WHERE conflict_group_id = ?
                    ORDER BY resonance_count DESC, id DESC
                    """,
                    (gid,),
                ).fetchall()
                if not members:
                    continue
                winner = members[0]
                # Degenerate: no surviving member (can happen only outside the
                # default lethal duel). Leave the group for next cycle's
                # resurrection/prune rather than supersede to a dead 'winner'.
                if winner["resonance_count"] <= 0:
                    continue
                loser_ids = [m["id"] for m in members[1:] if m["resonance_count"] <= 0]
                if not loser_ids:
                    continue
                placeholders = ",".join("?" * len(loser_ids))
                self._conn.execute(
                    f"""
                    UPDATE semantic_facts
                    SET tier = 'superseded',
                        superseded_by = ?,
                        superseded_at_cycle = ?,
                        conflict_group_id = NULL,
                        conflict_since_cycle = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id IN ({placeholders})
                    """,
                    (winner["id"], current_cycle, *loser_ids),
                )
                superseded += len(loser_ids)
            if superseded:
                self._conn.commit()
                logger.info(
                    "🪦 Supersedion: retired %d conflict loser(s) as history (cycle %d)",
                    superseded, current_cycle,
                )
            # Keep the superseded history bounded.
            if max_history and max_history > 0:
                self._enforce_superseded_cap(max_history)
        return superseded


    def _enforce_superseded_cap(self, max_history: int) -> int:
        """Bound the superseded-fact history: delete the oldest beyond the cap.

        Oldest = lowest superseded_at_cycle, then lowest id. These are terminal
        history rows already excluded from recall/promotion, so dropping the
        eldest when over the cap is the intended bound, not loss of active belief.
        Caller already holds self._lock. Returns rows deleted.
        """
        total = self._conn.execute(
            "SELECT COUNT(*) FROM semantic_facts WHERE tier = 'superseded'"
        ).fetchone()[0]
        if total <= max_history:
            return 0
        cur = self._conn.execute(
            """
            DELETE FROM semantic_facts
            WHERE id IN (
                SELECT id FROM semantic_facts
                WHERE tier = 'superseded'
                ORDER BY superseded_at_cycle ASC, id ASC
                LIMIT ?
            )
            """,
            (total - max_history,),
        )
        self._conn.commit()
        removed = cur.rowcount or 0
        if removed:
            logger.info(
                "Supersedion cap: pruned %d oldest superseded rows (cap=%d)",
                removed, max_history,
            )
        return removed


    def prune_weak_facts(self, forget_after_cycles: int = 0, protect_conflicts: bool = False) -> None:
        """Remove facts that have completely faded, regardless of tier. (Phase 1b: tier='superseded'
        is EXCLUDED - those sit at resonance <= 0 by design but are retained conflict history.)

        Forget policy (buried-but-pluckable, P2b-store):
          * ``forget_after_cycles == 0`` - delete a dormant fact (resonance <= 0) immediately (legacy
            behavior; the method default, so existing callers/tests are unchanged).
          * ``forget_after_cycles  > 0`` - DEMOTE then deep-delete: a dormant fact is KEPT (low
            resonance, still pluckable by a strong cue) and deleted only after it has stayed dormant
            for ``forget_after_cycles`` cycles. 'Eventually fades, preserve the essence' - cycle-
            driven, no wall-clock.
          * ``forget_after_cycles  < 0`` - never delete (pure archive).
        Dormancy is stamped/cleared on ``dormant_since_cycle`` against the logical memory clock.
        When ``protect_conflicts`` is set, facts in an ACTIVE conflict group are never demoted or
        deleted - held in limbo until arbitration (conflict-limbo, A9/A13). Default off (legacy)."""
        contested = " AND conflict_group_id IS NULL" if protect_conflicts else ""
        # Pinned facts (A5, P4a) are never demoted or deleted (they also never decay to 0, but a
        # fact pinned while already dormant must still be protected) - guard every delete/stamp.
        contested += " AND COALESCE(pinned, 0) = 0"
        with self._lock:
            if forget_after_cycles == 0:
                self._conn.execute(
                    "DELETE FROM semantic_facts WHERE resonance_count <= 0 AND tier != 'superseded'"
                    + contested)
                self._conn.commit()
                return
            cur_cycle = self._current_memory_cycle()
            # Stamp newly-dormant facts; clear the stamp on any reinforced back above 0 (revival).
            self._conn.execute(
                "UPDATE semantic_facts SET dormant_since_cycle = ? "
                "WHERE resonance_count <= 0 AND tier != 'superseded' AND dormant_since_cycle IS NULL"
                + contested,
                (cur_cycle,))
            self._conn.execute(
                "UPDATE semantic_facts SET dormant_since_cycle = NULL "
                "WHERE resonance_count > 0 AND dormant_since_cycle IS NOT NULL")
            # Deep-delete only after the dormant grace has elapsed (cycle-driven). <0 => never.
            if forget_after_cycles > 0:
                self._conn.execute(
                    "DELETE FROM semantic_facts "
                    "WHERE resonance_count <= 0 AND tier != 'superseded' "
                    "AND dormant_since_cycle IS NOT NULL AND (? - dormant_since_cycle) >= ?"
                    + contested,
                    (cur_cycle, int(forget_after_cycles)))
            self._conn.commit()


    def enforce_long_tier_cap(self, max_long_facts: int = 0) -> int:
        """Evict the weakest long-tier facts when the long tier exceeds a cap.

        Disabled when max_long_facts <= 0 (default). When set, keeps the strongest
        (highest resonance, then most-recently-updated) long facts and deletes the
        rest - a bounded-memory safety valve. Returns the number evicted.
        """
        if max_long_facts <= 0:
            return 0
        with self._lock:
            try:
                cur = self._conn.execute(
                    """
                    DELETE FROM semantic_facts
                    WHERE tier = 'long'
                      AND COALESCE(pinned, 0) = 0
                      AND id NOT IN (
                          SELECT id FROM semantic_facts
                          WHERE tier = 'long'
                          ORDER BY resonance_count DESC, updated_at DESC
                          LIMIT ?
                      )
                    """,
                    (max_long_facts,),
                )
                self._conn.commit()
                removed = cur.rowcount or 0
                if removed:
                    logger.info("Long-tier cap: evicted %d weak long facts (cap=%d)",
                                removed, max_long_facts)
                return removed
            except Exception as e:
                logger.error("Long-tier cap enforcement failed: %s", e)
                return 0


    def _relation_subjects(self, fact_id: int) -> set:
        """Normalized subject strings from the fact's stored relation triples."""
        rows = self._conn.execute(
            "SELECT DISTINCT subject FROM fact_relations WHERE fact_id = ?",
            (fact_id,),
        ).fetchall()
        return {str(r["subject"]).strip().lower() for r in rows
                if r["subject"] and str(r["subject"]).strip()}


    def _parallel_subject_veto(self, fid1: int, ents1: set,
                               fid2: int, ents2: set) -> bool:
        """True when a band-matched pair is PARALLEL facts about different
        subjects - not a contradiction.

        The false-positive class this kills: "gl_FragCoord is available in GLSL
        ES 1.00/3.00" vs "gl_FrontFacing is available in GLSL ES 1.00/3.00".
        Shared context entities (the version numbers) push overlap past 0.5 and
        the shared sentence template lands similarity mid-band - the exact
        contradiction signature - yet both facts are TRUE. What separates them
        from a real attribute contradiction ("user lives in Seattle" vs "user
        lives in Portland") is the SUBJECT role: parallel facts have disjoint
        subjects, contradictions share one. Entity sets are role-blind, so this
        reads the (subject, relation, object) triples relation extraction
        already stores.

        CONSERVATIVE by construction: vetoes only when BOTH facts carry relation
        subjects AND no subject on either side matches the other side (equality,
        containment, or membership in the other fact's entity set). Missing role
        info or any cross-mention → no veto, and the pair proceeds to the
        adjudicator/limbo path, which never destroys. A wrong match therefore
        only restores the old behavior, never suppresses a real duel."""
        subj1 = self._relation_subjects(fid1)
        subj2 = self._relation_subjects(fid2)
        if not subj1 or not subj2:
            return False                      # no role information - cannot veto
        e1 = {str(e).strip().lower() for e in ents1}
        e2 = {str(e).strip().lower() for e in ents2}

        def _matches(a: str, b: str) -> bool:
            return a == b or a in b or b in a

        for s1 in subj1:
            if any(_matches(s1, s2) for s2 in subj2):
                return False                  # shared subject → possible contradiction
            if any(_matches(s1, e) for e in e2):
                return False                  # other fact mentions this subject
        for s2 in subj2:
            if any(_matches(s2, e) for e in e1):
                return False
        return True


    def resolve_hrr_conflicts(
        self,
        adjudicator: Optional[Callable[[str, str], Optional[bool]]] = None,
    ) -> None:
        """HRR-powered conflict detection for established (mid + long tier) facts.

        Scans the 300 most-recently-updated mid/long facts (was long-only, which
        missed contradictions until BOTH facts survived to long tier - a
        contradiction is most worth surfacing when one belief is fresh). Pairs are
        gated by entity overlap-COEFFICIENT >= 0.5 (was Jaccard > 0.5, which
        excluded attribute contradictions whose differing values are entities) and
        then by the content-similarity band [conflict_sim_low, conflict_sim_high]
        - the band, unchanged, remains the real discriminator. Detected pairs are
        marked CONTESTED, not bled (conflict_limbo default ON), so a false positive
        surfaces for user arbitration rather than destroying a fact.

        Two false-positive guards run AFTER the band, BEFORE the lock:
        - the parallel-subject veto (_parallel_subject_veto, deterministic,
          gated by self.conflict_subject_veto): template-parallel facts about
          different subjects are skipped;
        - an optional ``adjudicator`` callback (content_a, content_b) ->
          True (contradict) / False (compatible: skip) / None (unknown: flag).
          Injected by the consolidation layer (typically an LLM second opinion)
          so the store itself stays model-free. FAIL-OPEN: errors and None keep
          the pre-existing conservative behavior - flag the pair; conflict_limbo
          protects it and arbitration (resolve_conflict / dismiss_conflict)
          decides.

        Two SPECIALIZED passes follow the general one, each catching a class the
        content+entity gates cannot pair: the policy pass (opposite-polarity
        rules, gated by detect_policy_conflicts) and the procedural pass
        (same-tool usage heuristics that contradict by paraphrase, gated by
        detect_procedural_conflicts - the adjudicator is reused there but locks
        only on an explicit True)."""
        if not _HRR_AVAILABLE:
            return
 
        import uuid

        with self._lock:
            # Phase 6: stamp the cycle a conflict pair forms so the 'conflicts as
            # conversation' surface can age-gate (don't nag the instant a duel starts).
            cur_cycle = self._current_memory_cycle()
            # FIX 9 LOGIC: Ensure we only evaluate facts not currently in a conflict group.
            # Exclude abstractions (category='abstract'): an abstraction shares
            # entities with its source facts but is intentionally more general, so
            # its content similarity is low - exactly the conflict signature. Left
            # in, abstractions would duel the very facts they summarize.
            rows = self._conn.execute(
                """
                SELECT id, content, hrr_vector
                FROM semantic_facts
                WHERE tier IN ('mid', 'long')
                  AND hrr_vector IS NOT NULL
                  AND conflict_group_id IS NULL
                  AND category NOT IN ('abstract', 'procedural')
                ORDER BY updated_at DESC
                LIMIT 300
                """
            ).fetchall()
 
            # NOTE: no early-return on <2 mid/long rows. The policy-contradiction
            # pass below also scans SHORT-tier policy facts and must run even when
            # the general (mid+long) pass has nothing to pair - else a fresh poison
            # policy (short) vs one established rule (long) is never checked. The
            # general pairwise loop is a no-op on <2 rows, so this is safe.
 
            entity_rows = self._conn.execute(
                """
                SELECT fe.fact_id, e.name
                FROM fact_entities fe
                JOIN entities e ON e.entity_id = fe.entity_id
                WHERE fe.fact_id IN (
                    SELECT id FROM semantic_facts
                    WHERE tier IN ('mid', 'long')
                    AND conflict_group_id IS NULL
                    AND category NOT IN ('abstract', 'procedural')
                )
                """
            ).fetchall()
 
            entity_map: Dict[int, set] = {}
            for r in entity_rows:
                entity_map.setdefault(r["fact_id"], set()).add(r["name"])
 
            rows = list(rows)
            conflicts_found = 0
            assigned_this_pass: set = set()   # don't re-group a fact already locked this pass

            # Contradiction band (TUNABLE). Real contradictions about the same
            # subject share most of their words (high sim) but differ on the
            # crucial term, so they land mid-range. >HIGH ⇒ effectively the same
            # fact; <LOW ⇒ unrelated facts that merely share an entity. Wide
            # bands manufacture false conflicts that then bleed a real fact to
            # death, so keep this band narrow.
            CONFLICT_SIM_LOW = self.conflict_sim_low
            CONFLICT_SIM_HIGH = self.conflict_sim_high

            # We deliberately re-encode with encode_text_rich (content only)
            # rather than using the stored hrr_vector. The stored vector
            # includes bind(..., ROLE_CONTENT) + bind(entity, ROLE_ENTITY)
            # components. For conflict detection we want pure content
            # similarity so that entity overlap (already filtered > 0.5)
            # does not artificially inflate similarity between contradictory
            # statements about the same entities.
            #
            # Cost: ~300 rich encodes per dream cycle 
            content_vecs: Dict[int, "np.ndarray"] = {}
            for r in rows:
                if r["content"]:
                    content_vecs[r["id"]] = hrr.encode_text_rich(
                        r["content"], self.hrr_dim
                    )
 
            for i in range(len(rows)):
                if rows[i]["id"] in assigned_this_pass:
                    continue
                for j in range(i + 1, len(rows)):
                    f1, f2 = rows[i], rows[j]
                    if f2["id"] in assigned_this_pass:
                        continue
 
                    ents1 = entity_map.get(f1["id"], set())
                    ents2 = entity_map.get(f2["id"], set())
                    if not ents1 or not ents2:
                        continue
 
                    # Overlap COEFFICIENT (shared / smaller set), not Jaccard.
                    # For an attribute contradiction ("X lives in A" vs "X lives in
                    # B") the differing VALUES are themselves entities, which drags
                    # Jaccard below 0.5 (|{X}| / |{X,A,B}| = 0.33) and wrongly
                    # excludes exactly the pair we want to catch. The coefficient
                    # keys on the shared subject(s) (=0.5 here); the content-
                    # similarity band below still does the real discrimination, so
                    # this does not widen what counts as a conflict.
                    shared = ents1 & ents2
                    if not shared:
                        continue
                    overlap = len(shared) / min(len(ents1), len(ents2))
                    if overlap < 0.5:
                        continue
 
                    v1 = content_vecs.get(f1["id"])
                    v2 = content_vecs.get(f2["id"])
                    if v1 is None or v2 is None:
                        continue
                    sim = hrr.similarity(v1, v2)
 
                    if CONFLICT_SIM_LOW <= sim <= CONFLICT_SIM_HIGH:
                        # Parallel-fact veto (deterministic, relation-role-aware):
                        # template-parallel facts about DIFFERENT subjects are not
                        # contradictions - skip before spending an adjudication call.
                        if getattr(self, "conflict_subject_veto", True) and \
                                self._parallel_subject_veto(f1["id"], ents1,
                                                            f2["id"], ents2):
                            continue
                        # Optional second opinion (injected by the consolidation
                        # layer; the store stays model-free). FAIL-OPEN: None or an
                        # exception keeps the pre-existing conservative behavior -
                        # flag; limbo protects, arbitration decides.
                        if adjudicator is not None:
                            verdict = None
                            try:
                                verdict = adjudicator(f1["content"], f2["content"])
                            except Exception as e:
                                logger.debug(
                                    "Conflict adjudicator failed (fail-open): %s", e)
                            if verdict is False:
                                continue
                        conflict_id = str(uuid.uuid4())[:8]
                        self._conn.execute(
                            "UPDATE semantic_facts SET conflict_group_id = ?, "
                            "conflict_since_cycle = ? WHERE id IN (?, ?)",
                            (conflict_id, cur_cycle, f1["id"], f2["id"])
                        )
                        assigned_this_pass.add(f1["id"])
                        assigned_this_pass.add(f2["id"])
                        conflicts_found += 1
                        break   # f1 is now grouped; move to the next i

            # ── Policy-contradiction pass (entity-less, opposite-polarity) ──
            # Catches what the content+entity pass cannot: two POLICY-like facts
            # that say the opposite thing in different words (sim ~0, no entities).
            # Signal = shared action stem + OPPOSITE stance (tighten vs loosen).
            # Gated to policy-like facts; conservative for precision.
            #
            # NOTE: includes SHORT tier (unlike the general pass above, which stays
            # mid+long to dodge distractor churn). A poison policy is adversarial,
            # FRESH input - it must be checked against established policies the
            # moment it lands, not after it dwells up to mid (that latency let a
            # just-injected poison go unflagged). Safe to scan short here because
            # _policy_like gates to rare, deliberate policy facts (no churn).
            if getattr(self, "detect_policy_conflicts", True):
                pol_rows = self._conn.execute(
                    "SELECT id, content, category FROM semantic_facts "
                    "WHERE tier IN ('short', 'mid', 'long') AND conflict_group_id IS NULL "
                    "ORDER BY updated_at DESC LIMIT 300"
                ).fetchall()
                meta = [(r["id"], _policy_stance(r["content"]), _policy_stems(r["content"]))
                        for r in pol_rows
                        if r["id"] not in assigned_this_pass
                        and _policy_like(r["content"], r["category"])]
                for i in range(len(meta)):
                    id1, st1, stem1 = meta[i]
                    if not st1 or id1 in assigned_this_pass:
                        continue
                    for j in range(i + 1, len(meta)):
                        id2, st2, stem2 = meta[j]
                        if id2 in assigned_this_pass or not st2:
                            continue
                        if st1 == st2:                     # same stance -> consistent
                            continue
                        if len(stem1 & stem2) < 1:         # must concern the same action
                            continue
                        conflict_id = str(uuid.uuid4())[:8]
                        self._conn.execute(
                            "UPDATE semantic_facts SET conflict_group_id = ?, "
                            "conflict_since_cycle = ? WHERE id IN (?, ?)",
                            (conflict_id, cur_cycle, id1, id2))
                        assigned_this_pass.add(id1)
                        assigned_this_pass.add(id2)
                        conflicts_found += 1
                        break

            # ── Procedural-contradiction pass (same-tool superstition sweep) ──
            # See the module comment above _PROC_TOOL_RE for the field finding
            # and the two-lane design. Scans ALL live tiers: superstition does
            # its damage from the moment it lands (recall does not tier-gate),
            # and procedural facts sit in importance_categories so decay will
            # not remove a fresh contradiction either.
            if getattr(self, "detect_procedural_conflicts", True):
                proc_rows = self._conn.execute(
                    "SELECT id, content FROM semantic_facts "
                    "WHERE category = 'procedural' "
                    "  AND tier IN ('short', 'mid', 'long') "
                    "  AND conflict_group_id IS NULL "
                    "ORDER BY updated_at DESC LIMIT 200"
                ).fetchall()
                pmeta = []
                for r in proc_rows:
                    if r["id"] in assigned_this_pass:
                        continue
                    tool = _proc_tool(r["content"])
                    if not tool:      # precision-first: unprefixed facts skipped
                        continue
                    pmeta.append((r["id"], r["content"], tool,
                                  _proc_stance(r["content"]),
                                  _proc_topic_stems(r["content"], tool)))
                adjudications = 0
                for i in range(len(pmeta)):
                    id1, c1, tool1, st1, stems1 = pmeta[i]
                    if id1 in assigned_this_pass:
                        continue
                    for j in range(i + 1, len(pmeta)):
                        id2, c2, tool2, st2, stems2 = pmeta[j]
                        if id2 in assigned_this_pass or tool2 != tool1:
                            continue
                        shared_topic = stems1 & stems2
                        contradict = False
                        if st1 and st2 and st1 != st2 and len(shared_topic) >= 1:
                            # Lane 1: plain do-vs-don't about the same topic.
                            contradict = True
                        elif (adjudicator is not None
                              and len(shared_topic) >= 2
                              and adjudications < _PROC_ADJUDICATION_CAP):
                            # Lane 2: paraphrase contradictions ("bundle URLs"
                            # vs "one URL per call") need semantics. Lock only
                            # on an explicit True - None/error/False skip.
                            adjudications += 1
                            try:
                                contradict = adjudicator(c1, c2) is True
                            except Exception as e:
                                logger.debug(
                                    "Procedural adjudicator failed (skip): %s", e)
                        if not contradict:
                            continue
                        conflict_id = str(uuid.uuid4())[:8]
                        self._conn.execute(
                            "UPDATE semantic_facts SET conflict_group_id = ?, "
                            "conflict_since_cycle = ? WHERE id IN (?, ?)",
                            (conflict_id, cur_cycle, id1, id2))
                        assigned_this_pass.add(id1)
                        assigned_this_pass.add(id2)
                        conflicts_found += 1
                        break

            if conflicts_found:
                self._conn.commit()
                logger.info("🔥 HRR conflict detection: marked %d conflicting fact pairs", conflicts_found)
