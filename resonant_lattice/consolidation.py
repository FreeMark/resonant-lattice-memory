"""consolidation.py — ConsolidationMixin: the waking (consolidation) epoch,
the Hebbian dream cycle, abstraction / procedural-distillation kicks,
tool-action ingest, and source_quote attestation.

Mixed into LatticeMemoryProvider; relies on the composite for self._store,
config attrs, locks/counters, and sibling methods via self."""

import json
import logging
import re
import time
import urllib.request

from attestation import _attest_source_quote
from store_common import hrr, _HRR_AVAILABLE
from self_write_gate import is_task_process_meta

logger = logging.getLogger(__name__)


def _ollama_post_with_retry(url: str, payload: dict, timeout: float, max_attempts: int = 3) -> dict:
    """Basic retry helper for Ollama POSTs (Phase 7 resilience).
    Returns parsed JSON or raises on final failure. Respects timeout.
    """
    last_err = None
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            if attempt < max_attempts - 1:
                sleep = 0.5 * (2 ** attempt)
                logger.debug("Ollama call attempt %d failed (%s); retrying in %.1fs", attempt+1, e, sleep)
                time.sleep(sleep)
    raise last_err  # re-raise last error after retries


def _strip_goal_injection(content: str) -> str:
    """Collapse the autonomous goal-loop's re-injected standing-goal message to a
    stub BEFORE consolidation, so the extractor never mines the repeated task
    INSTRUCTIONS as durable user facts. The goal loop re-sends the full goal text
    as a USER turn every cycle ('[Continuing toward your standing goal] Goal: …');
    only the assistant's responses carry new domain content worth extracting.
    No-ops on any turn without the marker (manual chats, and the raw turn-1 goal —
    whose one-off residue the self_write_gate catches)."""
    if content and "[Continuing toward your standing goal]" in content:
        return "[standing-goal continuation]"
    return content


class ConsolidationMixin:

#===========
# NEW private helper methods for Tool & Action Memory.
# They are self-contained and reuse existing rich HRR + entity machinery.
#===========
    def _infer_tool_success(self, result_text: str, tool_name: str) -> bool:
        """Heuristic to determine if a tool call succeeded from its result text."""
        if not result_text or not result_text.strip():
            return False
        text = result_text.lower()
        # Unambiguous structured failure markers (substring is safe here).
        strong_signals = [
            '"status": "error"', '"status":"error"', "status: error",
            "traceback (most recent call last)", "permission denied",
            "connection refused", "service unavailable",
        ]
        if any(sig in text for sig in strong_signals):
            return False
        # Weaker word signals — match WHOLE words so "0 errors", "no error", and
        # "invalidate" don't trip a false failure. (Still heuristic; see note.)
        # Tightened: "not found" and "invalid" appear in plenty of SUCCESSFUL
        # outputs ("0 invalid rows", "user not found, created new"), so they no
        # longer hard-fail on their own. Keep the unambiguous failure verbs.
        weak_word_patterns = (
            r"\berror\b", r"\bfailed\b", r"\bfailure\b", r"\bexception\b",
            r"\btimed?\s*out\b", r"\brefused\b",
        )
        if any(re.search(p, text) for p in weak_word_patterns):
            return False
        # Known success patterns for common tools (extend as needed)
        if tool_name in ("lattice_store", "search") and "results" in text:
            return True
        return True  # Default optimistic if we got a non-empty result


    def _extract_tool_actions(self, messages: list) -> list[dict]:
        """Parse OpenAI-style messages list and extract structured tool action records."""
        if not messages:
            return []

        actions = []
        tool_results = {}
        for msg in messages:
            if msg.get("role") == "tool":
                tool_results[msg.get("tool_call_id")] = msg.get("content", "")

        for msg in messages:
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            for tc in msg.get("tool_calls", []):
                real_call_id = tc.get("id")
                fn = tc.get("function", {})
                tool_name = fn.get("name", "unknown_tool")
                # Never ingest the memory tool's own calls as facts — that creates
                # a self-referential loop (memory about searching memory) that
                # pollutes the store and runs away on resonance.
                if tool_name == "lattice_store":
                    continue
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    args = {"raw_arguments": fn.get("arguments", "")}

                # Dedup key: prefer the platform tool_call_id. When it is missing
                # (some providers omit ids), derive a DETERMINISTIC fingerprint
                # from tool_name + raw arguments so BOTH the in-memory gate and the
                # UNIQUE partial index on tool_episodes.call_id still reject
                # history replays after a restart. (hash() is per-process
                # randomized and would NOT survive a restart — use hashlib.)
                if real_call_id:
                    dedup_id = real_call_id
                else:
                    import hashlib as _hashlib
                    _fp = f"{tool_name}:{fn.get('arguments', '')}"
                    dedup_id = "synth-" + _hashlib.sha1(_fp.encode("utf-8")).hexdigest()[:16]

                # Only ingest each tool call once for the provider's lifetime.
                with self._tool_ingest_lock:
                    if dedup_id in self._ingested_tool_call_ids:
                        continue
                    # Bounded fast-path cache. Safe to reset: the UNIQUE
                    # partial index on tool_episodes.call_id rejects any
                    # replays at the DB layer (INSERT OR IGNORE).
                    if len(self._ingested_tool_call_ids) > 20_000:
                        self._ingested_tool_call_ids.clear()
                    self._ingested_tool_call_ids.add(dedup_id)

                result_text = tool_results.get(real_call_id, "")
                success = self._infer_tool_success(result_text, tool_name)

                actions.append({
                    "tool_name": tool_name,
                    "arguments": args,
                    "result": result_text[:2000],  # safety cap
                    "success": success,
                    "call_id": dedup_id,           # durable DB-level dedup key
                })
        return actions


    def _ingest_tool_action(self, action: dict, session_id: str) -> None:
        """Record a tool invocation as a RAW procedural episode (not a fact).

        Tool calls are episodic events: they must NOT be deduplicated, merged, or
        run through Hebbian resonance individually. They accumulate in
        tool_episodes and are later generalized into reusable 'procedural' facts
        by the dream-cycle distillation pass (_distill_procedural_memory).
        """
        if not self._store or not self._write_enabled:
            return
        try:
            self._store.add_tool_episode(
                session_id=session_id,
                tool_name=action.get("tool_name", "unknown"),
                arguments=action.get("arguments", {}),
                result=action.get("result", ""),
                success=bool(action.get("success", False)),
                memory_cycle=self._memory_cycle,
                call_id=action.get("call_id"),
            )
        except Exception as e:
            logger.debug("Tool episode ingestion failed (non-fatal): %s", e)


    def _attest_quote(self, quote: str, transcript: str) -> str:
        """Verify a model-emitted source_quote against the consolidation transcript.

        Pulls the quote's named entities via the store's own extractor, then
        delegates to the two-channel verifier (_attest_source_quote). Returns
        'attested' | 'soft' | 'specific_mismatch' | 'unattested'. Caller drops the
        fact on 'specific_mismatch'.
        """
        try:
            entities = self._store._extract_entities(quote) if self._store else []
        except Exception:
            entities = []
        return _attest_source_quote(
            quote, transcript, entities, ratio_threshold=self._quote_match_threshold
        )


    # ====================== WAKING CYCLE ======================
    def _run_consolidation_epoch(self, session_id: str, force_blocking: bool = False, suppress_dream: bool = False) -> None:
        """Distill new facts from recent conversation using the reasoning model.
        Includes robust JSON parsing, embedding generation, HRR vectors, and entity extraction.
        Purely cycle-driven — no timers."""

        if not self._write_enabled:
            logger.debug("Consolidation skipped — writes disabled for this agent context")
            return

        if force_blocking:
            # Wait up to 50 seconds to outlast the active Ollama HTTP call (45s timeout)
            if not self._consolidation_lock.acquire(blocking=True, timeout=50.0):
                logger.error("Failed to acquire consolidation lock during shutdown.")
                return
        else:
            if not self._consolidation_lock.acquire(blocking=False):
                logger.debug("Consolidation epoch already running — skipping")
                return

        try:
            # 1. Get recent conversation turns
            episodes = self._store.get_recent_episodes(
                limit=self._reflection_frequency * 2, 
                session_id=session_id
            )
            if not episodes:
                logger.debug("No recent episodes to distill")
                return

            transcript = "\n".join([
                f"{ep['role'].upper()}: {_strip_goal_injection(ep['content'])}" 
                for ep in episodes
            ])

            # 2. Build prompt for the reasoning model
            prompt = f"{self._extraction_prompt}\n\nLOG:\n{transcript}\n\nJSON OUTPUT:"

            # 3. Call the reasoning model (with retry)
            payload = {
                "model": self._reason_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            }

            try:
                result = _ollama_post_with_retry(
                    f"{self._ollama_endpoint_reason}/api/generate",
                    payload,
                    timeout=getattr(self, "_reason_timeout", 300.0)
                )
                response_text = result.get("response", "[]").strip()
            except Exception as e:
                logger.error("Reason model call failed after retries: %s", e)
                return  # skip this epoch on hard failure (non-fatal for agent)

            # 4. Robust JSON parsing with multiple fallback layers
            extracted_facts = []

            # Use the shared cleaner from store.py (respects the _LatticeStore alias)
            response_text = self._store._clean_llm_json(response_text)

            try:
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']')
                if start_idx != -1 and end_idx != -1:
                    clean_json = response_text[start_idx:end_idx + 1]
                    extracted_facts = json.loads(clean_json)
                else:
                    extracted_facts = json.loads(response_text)
            except json.JSONDecodeError:
                logger.debug("Strict JSON parsing failed — using regex fallback")
                content_pattern = r'"content"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
                cat_pattern = r'"category"\s*:\s*"([^"]+)"'

                for match in re.finditer(content_pattern, response_text, re.IGNORECASE):
                    content = match.group(1).replace('\\"', '"')
                    cat_match = re.search(cat_pattern, response_text[match.end():match.end() + 120])
                    category = cat_match.group(1) if cat_match else "general"
                    extracted_facts.append({"content": content, "category": category})

            # 4b. Normalize: tolerate a single object or a {"facts": [...]} wrapper,
            #     so a non-array LLM response doesn't crash and drop the whole epoch.
            if isinstance(extracted_facts, dict):
                if isinstance(extracted_facts.get("facts"), list):
                    extracted_facts = extracted_facts["facts"]
                elif "content" in extracted_facts:
                    extracted_facts = [extracted_facts]
                else:
                    extracted_facts = []
            if not isinstance(extracted_facts, list):
                extracted_facts = []

            # 5. Process each extracted fact
            quotes_dropped = 0   # facts dropped by source-quote attestation
            # Semantic ROLLBACK: stamp every fact this epoch writes with a batch id,
            # so a bad extraction run can be rolled back as a unit (closed in finally).
            if extracted_facts:
                self._store.begin_write_batch(
                    phase="consolidation", model=self._reason_model,
                    source_session=session_id, cycle=self._memory_cycle)
            for fact in extracted_facts:
                if not isinstance(fact, dict):
                    continue
                content = fact.get("content")
                if not content or not isinstance(content, str):
                    continue

                # Phase E policy gate: never autonomously persist the agent's own
                # config/infra/identity chatter as a user/domain fact. Drop before
                # we spend an embedding call on it.
                if self._gate_self_writes and (
                    self._is_self_referential_infra(content)
                    or is_task_process_meta(content)
                ):
                    logger.debug("Auto-store gate: dropped self/infra or task-process meta fact: %s", content[:60])
                    continue

                category = fact.get("category", "general")

                # Phase D grounding: thread the verbatim source snippet (+ optional
                # ref) the model derived this fact from. Light hygiene: keep a
                # non-empty trimmed string, coerce anything else to None, cap length.
                source_quote = fact.get("source_quote")
                source_quote = source_quote.strip()[:500] if isinstance(source_quote, str) and source_quote.strip() else None
                source_ref = fact.get("source_ref")
                source_ref = source_ref.strip()[:500] if isinstance(source_ref, str) and source_ref.strip() else None

                # Phase D+ attestation: verify the quote against the transcript via
                # the two-channel grounding verifier. A fabricated/changed hard
                # specific (number/ID/entity) DROPS the fact entirely; an
                # un-anchored quote keeps the fact but nulls the suspect quote and
                # flags it 'unattested'. quote_status records the verdict for the
                # substrate (NULL when there is no quote; 'unverified' when off).
                quote_status = None
                if source_quote:
                    if not self._verify_source_quote:
                        quote_status = "unverified"
                    else:
                        verdict = self._attest_quote(source_quote, transcript)
                        if verdict == "specific_mismatch":
                            quotes_dropped += 1
                            logger.debug(
                                "Quote attestation: DROPPED fact (fabricated specific) "
                                "— %s | quote=%s", content[:60], source_quote[:60],
                            )
                            continue   # user policy: drop on hard-specific mismatch
                        if verdict == "unattested":
                            source_quote = None   # keep the fact, flag the weak quote
                        quote_status = verdict

                emb = self._retriever._get_embedding(content)
                if not emb:
                    continue

                hrr_vector = None
                entities = []
                if _HRR_AVAILABLE:
                    entities = self._store._extract_entities(content)
                    try:
                        hrr_vector = hrr.encode_fact(content, entities, dim=self._hrr_dim)
                    except Exception as e:
                        logger.debug(f"HRR encoding failed for fact: {e}")

                try:
                    action, fid = self._store.add_or_reinforce_fact(
                        content, emb, category, session_id,
                        hrr_vector=hrr_vector, entities=entities,
                        source_quote=source_quote, source_ref=source_ref,
                        quote_status=quote_status,
                    )
                    logger.debug(f"Fact '{content[:40]}...' → {action} (ID: {fid})")
                    # Tier-1 blind mirror (embedding/entities/HRR → semantic_he*) is NOT done
                    # per-fact here anymore — it runs once at the end of the epoch via
                    # self._blind_reconcile() (§14 6a / write-path completeness), the single
                    # mechanism that ALSO catches facts created store-side (abstraction/gist/
                    # procedural) and backfills a store on first blind-enable.
                    # Phase 5a: extract (subject, relation, object) triples from a
                    # FRESHLY added fact into the relation graph (idempotent, but only
                    # on 'added' to skip redundant work on reinforcement). Gated +
                    # non-fatal via the helper so it never blocks consolidation.
                    if self._enable_relations and action == "added" and fid and fid > 0:
                        self._extract_relations_for_fact(fid, content, entities)
                except Exception as e:
                    logger.error(f"Failed to ingest fact '{content[:40]}...': {e}", exc_info=True)

            # 5b. Blind-tier write-path completeness (§14 6a): mirror this epoch's new facts'
            # embedding/entities/HRR into the encrypted tables (no-op off the blind path; reads
            # the just-written plaintext back, no Ollama). One mechanism for every write path.
            self._blind_reconcile()

            # 6. Update cycle counter
            self._memory_cycle += 1
            self._store.set_cycle_counts(memory_cycle=self._memory_cycle)
            logger.info(
                f"✅ Memory Cycle {self._memory_cycle} completed — "
                f"{len(extracted_facts)} facts processed"
                + (f", {quotes_dropped} dropped (unverified specifics)" if quotes_dropped else "")
            )
 
            # Determine whether a dream cycle should follow, but don't decide
            # inside the lock — we release first so on_session_end(force_blocking)
            # can acquire it quickly without timing out during LLM abstraction calls.
            trigger_dream = (self._memory_cycle % self._dream_every_n_consolidations == 0)
 
        except Exception as e:
            logger.error(f"Consolidation epoch failed: {e}", exc_info=True)
            trigger_dream = False   # don't fire dream cycle after a failed epoch
        finally:
            try:
                self._store.end_write_batch()   # close the consolidation batch (no-op if none)
            except Exception:
                pass
            self._consolidation_lock.release()   # ← released BEFORE dream cycle
 
        # Dream cycle runs outside the consolidation lock.
        # _dream_lock inside _run_dream_cycle() prevents concurrent dream cycles.
        # on_session_end suppresses this and runs its own single dream cycle to
        # avoid back-to-back duplicate maintenance at shutdown.
        if trigger_dream and not suppress_dream:
            logger.info("Triggering Dream Cycle after waking cycle")
            self._run_dream_cycle()




    # ====================== DREAM CYCLE (Hebbian Maintenance) ======================
    def _run_dream_cycle(self) -> None:
        """Full Hebbian Dream Cycle — decay, promotion, abstraction, conflict resolution.
 
        Step order matters:
          0. increment_tier_cycles  — advance tier-dwell counters (short/mid)
          0.5 reencode_hrr_if_needed — one-shot encoding migration (self-gating)
          0.6 reembed_if_needed     — one-shot re-embed on embed_model change (self-gating, P4d)
          1. apply_cycle_decay      — bleed short/mid resonance (long exempt)
          1.5 apply_staleness_decay — extra decay for weak+stale facts (gated; off by default)
          2. apply_conflict_decay   — bleed conflicting facts; resurrect tie-breaker
          3. promote_facts          — mid→long first, then short→mid; needs resonance + dwell
          4. abstraction pass       — every N cycles, LLM generalization (gated)
          4.5 distill_procedural    — every N cycles, generalize tool episodes → procedural facts (gated)
          4.9 supersede_conflict_losers — retire conflict losers as history (gated)
          4.95 consolidate_before_prune — gist dying earned facts (gated; off by default)
          5. prune_weak_facts       — delete resonance <= 0 (superseded losers kept)
          6. free_conflict_winners  — clear conflict lock from sole survivors
          7. resolve_hrr_conflicts  — detect new conflicts in long-tier facts
          8. prune_episodes         — keep only N most recent sessions (+ episode_max_rows cap)
          8.5 prune_tool_episodes   — bound the raw tool-episode log (gated)
          9. gc_orphan_entities     — drop entity rows no longer linked to any fact
          10. enforce_long_tier_cap — evict weakest long facts beyond max_long_facts (gated)
          11. memory-health audit   — read-only health snapshot every N cycles (gated)
          12. _blind_reconcile      — mirror this cycle's new facts into the blind tables (blind mode only)
        """
        if not self._store or not self._write_enabled:
            logger.debug(
                "Dream cycle skipped — store missing or writes disabled "
                "for this agent context (non-primary)."
            )
            return
        if not self._dream_lock.acquire(blocking=False):
            return

        # === 0. Dwell + migrations ===
        try:
            self._dream_cycle_count += 1
            self._store.set_cycle_counts(dream_cycle=self._dream_cycle_count)
            # Semantic ROLLBACK: stamp this cycle's generative writes (abstraction /
            # gist / procedural distillation) with a batch id, closed in finally.
            self._store.begin_write_batch(phase="dream", model=self._reason_model,
                                          cycle=self._memory_cycle)
            logger.info(f"🧠 Dream Cycle {self._dream_cycle_count} started (cycle-driven)")

            # === 0. Dwell + migrations (continued) ===
            # 0. Advance tier-dwell counters (turn/cycle-based, never wall-clock)
            self._store.increment_tier_cycles()

            # Reset the recall-reinforcement gate for the new window so recalled
            # facts can earn one more small bump over the next set of turns.
            with self._recall_gate_lock:
                self._recalled_this_cycle.clear()
                self._conflicts_surfaced.clear()   # Phase 6: re-allow one nudge per conflict

            # 0.5 One-time HRR re-encode to the current rich encoding (off the
            #     startup hot path; self-gates via meta so it runs exactly once).
            self._store.reencode_hrr_if_needed()

            # 0.6 One-time re-embed if the embed_model changed (P4d). Self-gates via
            #     meta['embed_model']; rebuilds semantic_vec at the new dim when needed,
            #     so swapping the embedder on an existing store is turnkey. No-op in
            #     blind mode / when nothing changed. Off the startup hot path.
            self._reembed_if_needed()

            # 1. Exponential decay — stronger memories forget slower (contested facts held in
            #    limbo when conflict_limbo is on — see step 2)
            self._store.apply_cycle_decay(protect_conflicts=self._conflict_limbo,
                                          peak_discount=self._surprise_decay_discount,
                                          importance_discount=self._importance_decay_discount)

            # 1.5 Phase 2 'use it or lose it' — extra decay for weak AND long-
            #     unconfirmed facts (gated; default off via stale_decay_boost=0).
            if self._stale_decay_boost > 0:
                self._store.apply_staleness_decay(
                    self._memory_cycle, self._stale_decay_boost,
                    self._freshness_halflife_cycles,
                )

            # 2. Conflict resolution. Limbo (default): DON'T auto-bleed — hold contested facts in
            #    sustained resonance, flagged on recall, until the USER arbitrates (resolve_conflict).
            #    Limbo off: the original auto-bleed-to-resolution duel (floor>0 ⇒ non-lethal).
            if not self._conflict_limbo:
                self._store.apply_conflict_decay(self._conflict_decay_floor)
 
            # === 3. Promotion ===
            # 3. Tier promotion (mid→long before short→mid to prevent tier-skip)
            self._store.promote_facts()
 
            # 4. Memory Abstraction / Generalization Layer
            if self._dream_cycle_count % self._abstraction_frequency == 0:
                self._perform_abstraction_pass()

            # 4.5 Procedural Memory Distillation — generalize raw tool episodes
            #     into reusable 'procedural' facts (episodes -> distill -> facts,
            #     mirroring the conversational consolidation path).
            if (self._enable_tool_memory
                    and self._dream_cycle_count % self._tool_distill_frequency == 0):
                self._distill_procedural_memory()
 
            # 4.9 Phase 1b: retire conflict losers as superseded history BEFORE
            #     pruning (gated; default on). Uses the memory_cycle clock so
            #     superseded_at_cycle is comparable to learned_at/last_confirmed.
            if self._keep_superseded:
                self._store.supersede_conflict_losers(
                    self._memory_cycle, self._max_superseded_history
                )

            # 4.95 Phase 4: gist-preserving forgetting (gated; default off). Before
            #      pruning, summarise dying-but-once-important facts into a 'gist' so
            #      meaning survives detail loss. Frequency-controlled (LLM cost).
            if (self._gist_before_prune
                    and self._dream_cycle_count % self._gist_frequency == 0):
                self._consolidate_before_prune()

            # 5. Prune completely faded facts — non-superseded losers deleted here
            self._store.prune_weak_facts(self._forget_after_dormant_cycles,
                                         protect_conflicts=self._conflict_limbo)
 
            # 6. Free conflict winners — NOW safe to call because losers are gone.
            #    Uses LatticeStore.free_conflict_winners() which holds _lock internally.
            #    Previously this was an unguarded self._store._conn.execute() call here,
            #    which bypassed the lock and risked collision with parallel sessions.
            self._store.free_conflict_winners()
 
            # 7. Detect new conflicts in long-term memory via HRR algebra
            self._resolve_long_term_conflicts()
 
            # 8. Keep episode table bounded — session window (prune_keep_sessions)
            #    plus optional total-row cap (episode_max_rows, 0 = unlimited).
            self._store.prune_episodes(
                keep_sessions=self._prune_keep_sessions,
                max_rows=self._episode_max_rows,
            )

            # 8.5 Keep the procedural (tool) episode log bounded.
            if self._enable_tool_memory:
                self._store.prune_tool_episodes(keep=self._tool_episode_keep)

            # 9. Sweep entity rows no longer linked to any fact
            self._store.gc_orphan_entities()

            # 10. Bounded-memory safety valve: cap the long tier (opt-in via
            #     max_long_facts > 0). Long facts never decay, so without this the
            #     long tier grows monotonically as abstraction adds more.
            if self._max_long_facts > 0:
                self._store.enforce_long_tier_cap(self._max_long_facts)

            # 11. Cycle-based memory-health audit (read-only) — log a snapshot
            #     every N dream cycles. Strictly dream-cycle-driven, no timers.
            if (self._health_check_every_n_dream_cycles > 0
                    and self._dream_cycle_count % self._health_check_every_n_dream_cycles == 0):
                self._run_memory_health_audit()

            # 12. Blind-tier write-path completeness (§14 6a): mirror facts created store-side
            #     during this cycle (abstraction / gist / procedural distillation — none of which
            #     can mirror inline, the store holds no HE client) into the encrypted tables, and
            #     incrementally backfill on first blind-enable. No-op off the blind path. Runs
            #     AFTER prune so facts about to be deleted aren't mirrored.
            self._blind_reconcile()

            logger.info(f"🧠 Dream Cycle {self._dream_cycle_count} completed successfully.")

        except Exception as e:
            logger.error(f"Dream Cycle failed: {e}")
        finally:
            try:
                self._store.end_write_batch()   # close the dream batch (no-op if none)
            except Exception:
                pass
            self._dream_lock.release()

    def _reembed_if_needed(self) -> int:
        """P4d migration: re-embed all facts to the configured embed_model if it changed.

        Delegates to LatticeStore.reembed_if_needed, passing the live embedder
        (retriever._get_embedding) and the configured model. Self-gating via meta['embed_model']
        — a no-op on every cycle except the first after an embed_model switch on an existing store
        (then it re-embeds + rebuilds semantic_vec at the new dim, turnkey). Skipped in blind mode
        (the blind tier owns its own re-encryption path) and when there is no retriever. Non-fatal."""
        if self._encryption_mode == "blind":
            return 0
        if not (self._store and self._retriever):
            return 0
        try:
            return self._store.reembed_if_needed(
                self._retriever._get_embedding, self._embed_model
            )
        except Exception as e:
            logger.debug("Re-embed migration failed (non-fatal): %s", e)
            return 0

    def _blind_reconcile(self, limit: int = 0) -> int:
        """Write-path completeness (roadmap §14 6a): mirror every fact that has a plaintext row but
        NO blind ciphertext into the encrypted tables (embedding/HRR/entities), read back from the
        plaintext store — NO Ollama. Delegates to the BlindTier collaborator (blind_tier.py), which
        owns the writers + entity store; a no-op off the blind path (``self._blind_tier`` is None).
        Called at the end of the consolidation epoch + the dream cycle. Returns embedding cts written."""
        bt = getattr(self, "_blind_tier", None)
        return bt.reconcile(self._store, limit) if (bt is not None and self._store) else 0

    def _run_memory_health_audit(self) -> dict:
        """Read-only memory-health snapshot. Logs at INFO on the triggering dream
        cycle (so it lands in normal agent logs) and returns the dict. No side
        effects — also reused by the memory_audit tool action.
        """
        if not self._store:
            return {}
        try:
            health = self._store.get_memory_health(near_cap=self._health_near_cap)
            health["memory_cycle"] = self._memory_cycle
            health["dream_cycle"] = self._dream_cycle_count
            logger.info(
                "🩺 Memory health (dream cycle %d): %d facts (long=%d, near_cap>=%.0f: %d) | "
                "conflicts: %d groups / %d facts | entities: %d (orphan %d) | "
                "abstractions: %d (evidence_gone %d) | tool episodes: %d (undistilled %d)%s",
                self._dream_cycle_count, health.get("total_facts", 0),
                health.get("long_tier_facts", 0), health.get("near_cap_threshold", 0),
                health.get("near_cap_facts", 0), health.get("active_conflict_groups", 0),
                health.get("conflicted_facts", 0), health.get("total_entities", 0),
                health.get("orphan_entities", 0), health.get("abstractions_tracked", 0),
                health.get("abstractions_evidence_gone", 0),
                health.get("tool_episodes_total", 0), health.get("tool_episodes_undistilled", 0),
                "  [DEGRADED — FTS-only]" if health.get("degraded") else "",
            )
            return health
        except Exception as e:
            logger.debug("Memory health audit failed (non-fatal): %s", e)
            return {}


    def _perform_abstraction_pass(self) -> None:
        """Memory Abstraction Layer — merges similar facts into higher-level generalizations."""
        if self._store:
            self._store.perform_abstraction_pass(
                self._reason_model,
                self._ollama_endpoint_reason,
                prompt=self._consolidation_prompt,
                max_facts=self._abstraction_max_facts,
                max_clusters=self._abstraction_max_clusters,
                min_cluster_size=self._abstraction_min_cluster_size,
                max_cluster_size=self._abstraction_max_cluster_size,
                cluster_hrr_similarity=self._cluster_hrr_similarity,
                cluster_entity_overlap=self._cluster_entity_overlap,
                dedup_threshold=self._abstraction_dedup_threshold,
            )


    def _consolidate_before_prune(self) -> None:
        """Phase 4 — gist dying-but-once-important facts before they prune.

        Delegates to LatticeStore.consolidate_before_prune, reusing the abstraction
        clustering thresholds + the live reasoning model. Gated by gist_before_prune
        (default off) and gist_frequency in _run_dream_cycle; non-fatal on failure.
        """
        if self._store and self._gist_before_prune:
            try:
                self._store.consolidate_before_prune(
                    self._reason_model,
                    self._ollama_endpoint_reason,
                    prompt=self._gist_prompt,
                    gist_floor=self._gist_floor,
                    min_peak_resonance=self._gist_min_peak_resonance,
                    cluster_hrr_similarity=self._cluster_hrr_similarity,
                    cluster_entity_overlap=self._cluster_entity_overlap,
                    min_cluster_size=self._gist_min_cluster_size,
                    max_cluster_size=self._abstraction_max_cluster_size,
                    max_clusters=self._gist_max_clusters,
                    dedup_threshold=self._abstraction_dedup_threshold,
                )
            except Exception as e:
                logger.debug("Gist consolidation failed (non-fatal): %s", e)


    def _extract_relations_for_fact(self, fact_id: int, content: str, entities: list) -> None:
        """Phase 5a — extract & store relational triples for one new fact.

        Delegates to LatticeStore.extract_and_store_relations (deterministic
        entity-grounded patterns + optional LLM pass). Passes the reasoning model
        only when relation_extract_llm is on. Non-fatal: a failure here never aborts
        the consolidation loop.
        """
        if not (self._store and self._enable_relations):
            return
        try:
            self._store.extract_and_store_relations(
                fact_id, content, entities=entities,
                min_confidence=self._relation_min_confidence,
                reason_model=self._reason_model,
                ollama_endpoint=self._ollama_endpoint_reason,
                use_llm=self._relation_extract_llm,
                llm_prompt=self._relation_prompt,
            )
        except Exception as e:
            logger.debug("Relation extraction failed (non-fatal): %s", e)


    def _generate_session_narrative(self, session_id: str) -> None:
        """Phase 8 — write a durable one-paragraph narrative gist of the session.

        Delegates to LatticeStore.summarize_session (gather episodes → LLM → store +
        bound). Called from on_session_end after final consolidation, gated by
        enable_narrative. Non-fatal: a failure never blocks session shutdown. Stamps
        the memory_cycle range the session spanned (started → current).
        """
        if not (self._store and self._enable_narrative):
            return
        try:
            self._store.summarize_session(
                self._reason_model,
                self._ollama_endpoint_reason,
                session_id,
                prompt=self._narrative_prompt,
                started_cycle=getattr(self, "_session_start_cycle", self._memory_cycle),
                ended_cycle=self._memory_cycle,
                created_cycle=self._memory_cycle,
                keep=self._narrative_keep,
                min_episodes=self._narrative_min_episodes,
            )
        except Exception as e:
            logger.debug("Session narrative generation failed (non-fatal): %s", e)


    def _distill_procedural_memory(self) -> None:
        """Generalize raw tool episodes into reusable procedural facts.

        Delegates to LatticeStore.distill_procedural_facts, which runs the LLM
        distillation and stores results as ordinary 'procedural' semantic facts
        (so they dedup, reinforce, promote, and surface through normal recall).
        """
        if self._store and self._enable_tool_memory:
            try:
                self._store.distill_procedural_facts(
                    self._reason_model,
                    self._ollama_endpoint_reason,
                    prompt=self._procedural_prompt,
                    min_episodes=self._tool_distill_min_episodes,
                    max_tools=self._tool_distill_max_tools,
                    sample_size=self._tool_distill_sample_size,
                )
            except Exception as e:
                logger.debug("Procedural distillation failed (non-fatal): %s", e)


    def _resolve_long_term_conflicts(self) -> None:
        """Delegate HRR conflict detection to LatticeStore.
 
        Scans long-tier facts with high entity overlap and low HRR content
        similarity — those are likely contradictions. Pairs are grouped under
        a conflict_group_id and enter the Duel-to-the-Death decay loop on
        the next dream cycle.
        """
        if self._store and _HRR_AVAILABLE:
            self._store.resolve_hrr_conflicts()
