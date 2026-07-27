"""consolidation.py - ConsolidationMixin: the waking (consolidation) epoch,
the Hebbian dream cycle, abstraction / procedural-distillation kicks,
tool-action ingest, and source_quote attestation.

Mixed into LatticeMemoryProvider; relies on the composite for self._store,
config attrs, locks/counters, and sibling methods via self."""

import json
import logging
import re
import time
import urllib.request
from reason_gate import reason_slot

from attestation import _attest_source_quote
from proc_lock import FinalizeLock, FINALIZE_LOCK_WAIT
from store_common import hrr, _HRR_AVAILABLE
from self_write_gate import is_task_process_meta
import source_index

logger = logging.getLogger(__name__)

# Web-facing tool names for the synthesis-session test. Matches web_search /
# web_extract / web_fetch / browser / fetch_url / firecrawl-style names but NOT
# local tools like search_files (must START with web/fetch/http or be a crawler).
_WEB_TOOL_RE = re.compile(r"^web|^browser|^fetch|^http|crawl", re.I)


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
            with reason_slot(), urllib.request.urlopen(req, timeout=timeout) as response:
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
    No-ops on any turn without the marker (manual chats, and the raw turn-1 goal -
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
        # Weaker word signals - match WHOLE words so "0 errors", "no error", and
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
                # Never ingest the memory tool's own calls as facts - that creates
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
                # randomized and would NOT survive a restart - use hashlib.)
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


    def _audit_extraction(self, session_id, attempts, quoteless_retries, facts,
                          quoted, t_chars, t_quotes, synthesis, outcome) -> None:
        """Record what the extraction call returned, in the database.

        Exists because `logger` is not a channel on this deployment: hermes emits no
        stderr, so every consolidation log message -- including the retry guard's -- has
        gone nowhere on every run. Without this row there is no way to distinguish "the
        quoteless retry fired three times and the model kept omitting the field" from
        "the guard never executed", and both were reported in one night because the
        evidence could not settle it.

        Never raises. An audit failure must not cost the epoch its facts -- the whole
        point is to observe the pipeline, not to become a new way for it to die.
        """
        try:
            store = getattr(self, "_store", None)
            conn = getattr(store, "_conn", None)
            if conn is None:
                return
            lock = getattr(store, "_lock", None)
            cycle = None
            try:
                cycle = store._current_memory_cycle()
            except Exception:                                    # noqa: BLE001
                pass
            sql = ("INSERT INTO extraction_audit (session_id, cycle, attempts, "
                   "quoteless_retries, facts, quoted, transcript_chars, "
                   "transcript_quotes, synthesis, outcome) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?)")
            args = (session_id, cycle, int(attempts), int(quoteless_retries),
                    int(facts), int(quoted), int(t_chars), int(t_quotes),
                    1 if synthesis else 0, str(outcome))
            if lock is not None:
                with lock:
                    conn.execute(sql, args)
                    conn.commit()
            else:
                conn.execute(sql, args)
                conn.commit()
        except Exception as e:                                   # noqa: BLE001
            logger.debug("extraction_audit write skipped (non-fatal): %s", e)


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
        Purely cycle-driven - no timers."""

        if not self._write_enabled:
            logger.debug("Consolidation skipped - writes disabled for this agent context")
            return

        if force_blocking:
            # Wait up to 50 seconds to outlast the active Ollama HTTP call (45s timeout)
            if not self._consolidation_lock.acquire(blocking=True, timeout=50.0):
                logger.error("Failed to acquire consolidation lock during shutdown.")
                return
        else:
            if not self._consolidation_lock.acquire(blocking=False):
                logger.debug("Consolidation epoch already running - skipping")
                return

        # Cross-PROCESS serialization (concurrent overnight workers + gateway on
        # one profile DB): same skip/wait semantics as the in-process lock above.
        # Fresh instance per finalize unit - see proc_lock module docstring.
        #
        # SCOPE (2026-07-25): the lock is taken LATE - immediately before the
        # write phase - not here. Measured on five concurrent overnight lanes,
        # holding it across the whole epoch serialized 111.3s / 5.4s / 121.1s
        # per lane with zero-gap handoffs, because the LLM stages sat inside it:
        # one extraction call (up to _reason_timeout, configured 900s, x3
        # attempts) plus ONE relation call PER newly added fact (60s each). None
        # of that touches state the lock protects - the lock exists for the
        # read-then-write dedup race in add_or_reinforce_fact and for Hebbian
        # maintenance. Serializing LLM latency behind it capped throughput at
        # 1/hold_time regardless of how many lanes ran.
        _flock = FinalizeLock(self._store.db_path)
        # Preserve the early-skip for SCHEDULED work: bail before spending a
        # reason-model call, exactly as when the lock was taken up front. Forced
        # (session-end) work still queues and waits below.
        if not force_blocking and _flock.held_by_other():
            logger.info(
                "Consolidation deferred - another process is finalizing this DB")
            self._consolidation_lock.release()
            return

        _t_epoch = time.monotonic()
        _t_extract = _t_write = _t_lockwait = _t_rel = 0.0
        # Separate name: `finally` runs on the early-return paths too (no
        # episodes, extraction failure), where a shared cursor would be unbound.
        _t_write_start = None
        _deferred_relations = []   # (fact_id, content, entities) - run unlocked
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

            # === Source index (default OFF) ===================================
            # Episodes carry only user/assistant rows, so the urls the agent read
            # are absent from `transcript` and it cannot cite them. Append a
            # compact `url | title` list harvested during sync_turn.
            #
            # This is HALF a feature on its own and would be a liability alone: a
            # list of urls is a menu, and a model handed a menu picks a
            # plausible-looking entry. The other half -- mechanical verification of
            # every ref the extractor attaches, below at `_ref_bodies` -- is what
            # makes supplying the list safe. Keep them together or neither.
            #
            # ATTESTED SEPARATELY from the quote channel on purpose: a wrong quote
            # and a wrong url are different failures with different blast radius.
            _ref_bodies: dict = {}
            if getattr(self, "_url_index_for_extraction", False):
                try:
                    _srcs = self._store.get_session_sources(
                        session_id, limit=400, with_body=True)
                    if _srcs:
                        _ref_bodies = {s["url"]: s["body"] for s in _srcs
                                       if s.get("url") and s.get("body")}
                        _blk = (source_index.build_index_block(_srcs)
                                if getattr(self, "_url_index_in_prompt", False)
                                else "")
                        if _blk:
                            transcript = "%s\n\n%s" % (transcript, _blk)
                            # INFO, not DEBUG. Without this line there is no way to
                            # tell "the index was absent" from "the index was
                            # offered and the model ignored it" -- two problems with
                            # completely different fixes. Diagnosing a zero-ref
                            # result cost a round trip for exactly that reason.
                            logger.info(
                                "Source index: %d urls offered to extraction "
                                "(%d chars)", len(_srcs), len(_blk))
                except Exception as e:                           # noqa: BLE001
                    logger.debug("Source index unavailable (non-fatal): %s", e)

            # Re-sync the logical clock from the DB before announcing/stamping:
            # in a long-lived process the cached counter lags behind one-shot
            # runs that advanced the shared profile DB while we sat idle.
            self._memory_cycle = self._store.get_cycle_counts()[0]

            # Announce the cycle BEFORE the (potentially slow) reason-model call,
            # so a long cloud round-trip reads as "consolidating" rather than a
            # silent hang. The completion line alone can look like a stall while
            # the reason model is mid-flight (esp. on a rate-limited endpoint).
            logger.info(
                f"🧠 Memory Cycle {self._memory_cycle + 1} started "
                f"({len(episodes)} recent episodes, reason model)"
            )

            # [SYNTHESIZED] provenance detection (label gauntlet 2026-07-11): a
            # session that READ its own memory (lattice_store reads, tracked
            # in-process) and touched NO web tools is reflection, not research -
            # facts born from it restate or recombine stored knowledge. They get
            # source_ref "synthesized:<session>" below so recall can mark them and
            # no secondhand URL masquerades as firsthand provenance. Domain
            # category is KEPT (unlike mental_note): a synthesized webdev
            # principle is still webdev knowledge.
            synthesis_session = False
            if session_id in getattr(self, "_lattice_read_sessions", set()):
                try:
                    _profile = self._store.session_tool_names(session_id)
                except Exception:
                    _profile = set()
                synthesis_session = not any(_WEB_TOOL_RE.search(t or "") for t in _profile)
                if synthesis_session:
                    logger.info("Synthesis session detected (%s): memory reads, "
                                "zero web tools - facts will carry [SYNTHESIZED]",
                                session_id)

            # 2. Build prompt for the reasoning model
            prompt = f"{self._extraction_prompt}\n\nLOG:\n{transcript}\n\nJSON OUTPUT:"

            # 3. Call the reasoning model + parse (one attempt).
            payload = {
                "model": self._reason_model,
                "prompt": prompt,
                "stream": False,
                # ollama exposes thinking as a per-REQUEST field only - there is
                # no Modelfile parameter for it ("unknown parameter 'think'"), so
                # a thinking-capable reason model must be told here or it loses
                # its whole answer into an unclosed think block. Measured on .46:
                # gemma4:12b omitted -> 32.6s / done_reason=length / 0 chars;
                # think=false -> 1.6s / done_reason=stop / correct JSON. This
                # already cost a run: five lanes did 571s of research each and
                # banked ZERO facts because consolidation got empty strings back.
                # Safe on every profile - a model WITHOUT the capability accepts
                # and ignores it (granite4.1:8b-q8: 0.9s, no error).
                "think": False,
                "options": {"temperature": 0.1}
            }

            # A RETRY MUST DIFFER FROM WHAT IT RETRIES. The first version of this guard
            # re-sent an IDENTICAL payload at temperature 0.1, and a live run proved that
            # buys nothing: session 20260727_044913 spent FIVE attempts on one block and
            # every one returned the same 44 facts with zero source_quote, from a report
            # whose own text carried quoted fragments on 28 of 47 fact lines. At that
            # temperature the call is effectively deterministic, so N attempts are one
            # attempt billed N times -- 5-10 minutes of a 36-minute block, for nothing.
            # Each corrective retry now (a) names the omission in the prompt and
            # (b) raises temperature, so the model is asked something genuinely new.
            _QUOTE_CORRECTION = (
                "\n\nCORRECTION - YOUR PREVIOUS RESPONSE WAS REJECTED. You returned {n} "
                "facts and EVERY ONE omitted the \"source_quote\" field. The LOG above "
                "contains {q} quotation characters, so the material to quote is present. "
                "Re-extract now. Every object MUST carry \"source_quote\" copied "
                "CHARACTER-FOR-CHARACTER from the LOG above - not paraphrased, not your "
                "own wording. If you genuinely cannot find a supporting fragment in the "
                "LOG for a fact, DROP that fact rather than emit it without a quote."
            )
            # Appended AFTER the log and before the output cue on purpose: ollama
            # truncates an oversized prompt by keeping the TAIL (measured -- a marker at
            # the front of a 26k-token prompt was gone, the window collapsing to the last
            # 8,195 tokens), so a correction placed at the front is the first thing
            # discarded, exactly when it matters most.
            _RETRY_TEMPS = (0.1, 0.45, 0.7)

            def _extract_once(correction: str = "", attempt: int = 0) -> list:
                """One reason-model call + robust JSON parse → list of fact dicts.

                With no correction and attempt 0 this sends the ORIGINAL payload
                unchanged, so the healthy path is byte-for-byte what it always was.
                """
                _payload = payload
                if correction or attempt:
                    _payload = dict(payload)
                    _payload["prompt"] = (
                        f"{self._extraction_prompt}\n\nLOG:\n{transcript}"
                        f"{correction}\n\nJSON OUTPUT:")
                    _payload["options"] = dict(payload.get("options") or {})
                    _payload["options"]["temperature"] = _RETRY_TEMPS[
                        min(attempt, len(_RETRY_TEMPS) - 1)]
                result = _ollama_post_with_retry(
                    f"{self._ollama_endpoint_reason}/api/generate",
                    _payload,
                    timeout=getattr(self, "_reason_timeout", 300.0),
                )
                rt = self._store._clean_llm_json(result.get("response", "[]").strip())
                facts: object = []
                try:
                    s_i, e_i = rt.find('['), rt.rfind(']')
                    facts = json.loads(rt[s_i:e_i + 1] if (s_i != -1 and e_i != -1) else rt)
                except json.JSONDecodeError:
                    logger.debug("Strict JSON parsing failed - using regex fallback")
                    cp = r'"content"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
                    kp = r'"category"\s*:\s*"([^"]+)"'
                    facts = []
                    for m in re.finditer(cp, rt, re.IGNORECASE):
                        c = m.group(1).replace('\\"', '"')
                        km = re.search(kp, rt[m.end():m.end() + 120])
                        facts.append({"content": c, "category": km.group(1) if km else "general"})
                # Tolerate a single object or a {"facts": [...]} wrapper.
                if isinstance(facts, dict):
                    if isinstance(facts.get("facts"), list):
                        facts = facts["facts"]
                    elif "content" in facts:
                        facts = [facts]
                    else:
                        facts = []
                return facts if isinstance(facts, list) else []

            # Reasoning models non-deterministically return an empty array for a
            # transcript that plainly contains extractable facts (observed ~half the
            # time on the reason endpoint), which would silently drop a whole turn's
            # knowledge. Re-attempt a few times when the transcript is substantial.
            #
            # AND THE SAME FLAKINESS HAS A SECOND FACE: a full batch of facts with the
            # source_quote field omitted from every single one. Measured on a live vet
            # corpus of 2,357 facts: 279 (11.6%) carry no quote, and 252 of those --
            # 90% -- came from just NINE sessions where NOT ONE fact had a quote. The
            # distribution is switch-like rather than gradual: 8 sessions entirely
            # quoteless, 6 mixed, 114 clean. One failing session banked 39 facts with
            # zero quotes from a report whose own text held 42 quotation marks and a
            # verbatim quotation in its first fact -- the material was there and the
            # extraction call dropped the field wholesale.
            #
            # The old break condition (`if extracted_facts`) could not see this: a
            # quoteless batch is non-empty, so it broke on the first attempt. Those
            # facts get banked with quote_status='no_quote' -- unverifiable, and
            # ineligible for mechanical ref attachment, which is how one bad extraction
            # call costs a whole block its provenance. It also corrupted a measurement:
            # the same failure sat inside BOTH arms of the verbatim-quote experiment and
            # dragged its aggregate from 61% to 41%.
            #
            # Retrying is gated tightly, because a spurious retry buys nothing:
            #   * the batch must be at least _quoteless_retry_floor facts. A 1-3 fact
            #     turn with nothing quotable is ordinary; 39 is not.
            #   * the TRANSCRIPT must itself contain quotation marks. With nothing to
            #     quote, a quoteless extraction is the CORRECT answer, and retrying
            #     would burn a model call on every tool-only turn.
            #   * synthesis sessions are exempt outright. A dream/abstraction cycle has
            #     no source page, so its facts SHOULD be quoteless.
            #
            # The best batch seen is kept rather than the last, so a retry that returns
            # empty or worse can never leave this epoch with less than attempt 1 had.
            extracted_facts = []
            _max_attempts = max(1, int(getattr(self, "_extraction_max_attempts", 3)))
            _quoteless_floor = max(1, int(getattr(self, "_quoteless_retry_floor", 5)))
            # Two quote characters = ONE complete quoted phrase, the honest minimum for
            # "there was something here to quote". Set at 4 first, which would have
            # excused an extractor that missed the only quoted phrase in a 16KB report;
            # the measured failure had 42, so the threshold was never load-bearing for
            # it, and the looser bar costs at most one extra call on a turn that has
            # exactly one quotation.
            _transcript_has_quotes = (transcript or "").count('"') >= 2

            def _n_quoted(fs) -> int:
                return sum(1 for f in fs
                           if isinstance(f, dict)
                           and str(f.get("source_quote") or "").strip())

            _best: list = []
            # Counters mirrored into extraction_audit. Logs are not a usable channel
            # here -- hermes emits no stderr, so every logger call below has gone to a
            # void on every run, which is why a deployed retry guard could not be
            # evaluated at all. `_ea_attempts` being >1 in the table is the only
            # available proof the retry actually fired.
            _ea_attempts = 0
            _ea_quoteless_retries = 0
            _ea_outcome = "ok"
            # Quoteless retries are capped SEPARATELY from _max_attempts. Two corrective
            # attempts is the honest budget: the failure is a refusal far more often than
            # flakiness, so attempts three through five measured as pure waste.
            #
            # WHY 2 AND NOT 1. Settled over one 131-block run: 6 firings in 119 audited
            # sessions (5.0%), five of them recovered -- THREE on the first corrective
            # retry (86 of 86 facts quoted) and TWO on the second at temperature 0.7 (57 of
            # 59). A cap of 2 STRICTLY DOMINATES a cap of 1, because a session that
            # recovers at attempts=2 breaks out there whatever the cap is: raising the cap
            # costs nothing on those and gains the 57 facts the attempts=3 cases would
            # otherwise have banked unquoted. The only price of a higher cap is one extra
            # call on a genuine refusal. Corpus effect: no_quote fell 11.6% -> 1.4%.
            #
            # Worth recording how this was learned. An earlier revision of this comment
            # asserted that no firing had ever recovered at attempts=2 -- on a sample of
            # two -- and the very next firing did exactly that. The sample was far too
            # small for the absolute claim it carried, which is why this is a dominance
            # argument now and not a floor.
            # Transcript quote density is NOT the discriminator either: the
            # 28/28 recovery came from a transcript with 38 quote characters while the
            # pre-fix total failure had 70, which is why the fix targets how the model is
            # ASKED rather than how much there is to quote.
            _quoteless_max = max(1, int(getattr(self, "_quoteless_max_retries", 2)))
            _pending_correction = ""
            _pending_attempt = 0
            _last_quoteless_n = None
            _t0 = time.monotonic()
            for _ea in range(_max_attempts):
                try:
                    extracted_facts = _extract_once(_pending_correction, _pending_attempt)
                    _ea_attempts += 1
                except Exception as e:
                    logger.error("Reason model call failed after retries: %s", e)
                    self._audit_extraction(
                        session_id, _ea_attempts, _ea_quoteless_retries, 0, 0,
                        len(transcript or ""), (transcript or "").count('"'),
                        synthesis_session, "call_failed")
                    return  # skip this epoch on hard failure (non-fatal for agent)

                # Prefer more QUOTED facts, tie-break on batch size. Ordering matters:
                # a 40-fact quoteless batch must not beat a 30-fact attested one.
                if (_n_quoted(extracted_facts), len(extracted_facts)) > \
                        (_n_quoted(_best), len(_best)):
                    _best = extracted_facts

                if len(transcript) < 200:
                    _ea_outcome = "short_transcript"
                    break
                if not extracted_facts:
                    _ea_outcome = "zero_facts"
                    logger.info(
                        "Consolidation extraction returned 0 facts (attempt %d/%d) from "
                        "a %d-char transcript; retrying (reasoning-model flakiness "
                        "guard)", _ea + 1, _max_attempts, len(transcript),
                    )
                    continue
                if (not synthesis_session and _transcript_has_quotes
                        and len(extracted_facts) >= _quoteless_floor
                        and _n_quoted(extracted_facts) == 0):
                    _n_q = len(extracted_facts)
                    # A CORRECTIVE retry that comes back with the same batch size and
                    # still no quotes is a deterministic refusal, not flakiness. Stop
                    # instead of spending the remaining attempts re-asking a question
                    # that has already been answered the same way twice.
                    if _pending_correction and _n_q == _last_quoteless_n:
                        _ea_outcome = "quoteless_deterministic"
                        logger.warning(
                            "Consolidation: corrective retry returned the SAME %d "
                            "quoteless facts for session %s; treating as a deterministic "
                            "refusal and stopping after %d attempts",
                            _n_q, session_id, _ea_attempts,
                        )
                        break
                    if _ea_quoteless_retries >= _quoteless_max:
                        break   # budget spent; the post-loop block labels the outcome
                    # logger.info, not debug: this is exactly the observability that was
                    # missing while nine sessions failed this way unnoticed.
                    logger.info(
                        "Consolidation extraction returned %d facts with NO "
                        "source_quote on any of them (attempt %d/%d) from a %d-char "
                        "transcript containing %d quotation marks; retrying WITH a "
                        "correction naming the omission at temperature %.2f",
                        _n_q, _ea + 1, _max_attempts, len(transcript),
                        (transcript or "").count('"'),
                        _RETRY_TEMPS[min(_ea_quoteless_retries + 1,
                                         len(_RETRY_TEMPS) - 1)],
                    )
                    _last_quoteless_n = _n_q
                    _ea_quoteless_retries += 1
                    _ea_outcome = "quoteless_retry"
                    _pending_correction = _QUOTE_CORRECTION.format(
                        n=_n_q, q=(transcript or "").count('"'))
                    _pending_attempt = _ea_quoteless_retries
                    continue
                _ea_outcome = "ok"
                break

            # Never end worse than the best attempt.
            if (_n_quoted(_best), len(_best)) > \
                    (_n_quoted(extracted_facts), len(extracted_facts)):
                extracted_facts = _best
            if (extracted_facts and _n_quoted(extracted_facts) == 0
                    and not synthesis_session and _transcript_has_quotes
                    and len(extracted_facts) >= _quoteless_floor):
                # Every attempt failed the same way. Bank the facts (they are often
                # true, which is why quote_status marks rather than drops) but say so
                # loudly, because this is the class that silently cost 252 facts their
                # provenance.
                logger.warning(
                    "Consolidation: all %d attempts returned quoteless batches for "
                    "session %s; banking %d facts as no_quote",
                    _ea_attempts, session_id, len(extracted_facts),
                )
                # Do not flatten the more informative label: "deterministic" says the
                # corrective retry was answered identically, which is the difference
                # between a model that cannot quote this transcript and one that is
                # merely flaky. Both bank, but only one is worth re-running research for.
                if _ea_outcome != "quoteless_deterministic":
                    _ea_outcome = "quoteless_exhausted"

            # One row per epoch that ran an extraction, written BEFORE the write lock is
            # taken and before anything can fail downstream: an audit that only survives
            # the happy path would be missing exactly when it is needed.
            self._audit_extraction(
                session_id, _ea_attempts, _ea_quoteless_retries,
                len(extracted_facts), _n_quoted(extracted_facts),
                len(transcript or ""), (transcript or "").count('"'),
                synthesis_session, _ea_outcome)

            _t_extract = time.monotonic() - _t0

            # ---- WRITE PHASE BEGINS: take the cross-process lock HERE ----
            # Everything above is read-only (episodes, cycle counts, tool names)
            # plus the extraction LLM call, so it can run fully in parallel
            # across lanes. From here on we mutate shared state, so we serialize.
            _t0 = time.monotonic()
            if not _flock.acquire(FINALIZE_LOCK_WAIT if force_blocking else 0.0):
                if force_blocking:
                    logger.error(
                        "Finalize lock still held by another process after %.0fs - "
                        "skipping forced consolidation (episodes remain durable).",
                        FINALIZE_LOCK_WAIT)
                else:
                    logger.info(
                        "Consolidation deferred - another process is finalizing "
                        "this DB")
                return
            _t_lockwait = time.monotonic() - _t0
            _t_write_start = time.monotonic()

            # 5. Process each extracted fact
            quotes_dropped = 0   # facts dropped by source-quote attestation
            refs_dropped = 0     # urls removed as unearned (ref verification)
            refs_verified = 0    # model-supplied urls kept (page contains the quote)
            refs_found = 0       # urls attached MECHANICALLY (the primary path)
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
                # Synthesis stamp: in a memory-only reflection session the agent
                # fetched nothing, so ANY ref the extractor emits is secondhand
                # (echoed from recalled content - the fact #2913 class). Record
                # the truthful origin instead.
                if synthesis_session:
                    source_ref = f"synthesized:{session_id}"

                # Phase D+ attestation: verify the quote against the transcript via
                # the two-channel grounding verifier. A fabricated/changed hard
                # specific (number/ID/entity) DROPS the fact entirely; an
                # un-anchored quote keeps the fact but nulls the suspect quote and
                # flags it 'unattested'. quote_status records the verdict for the
                # substrate (NULL when there is no quote; 'unverified' when off).
                quote_status = None
                if not source_quote:
                    # NO QUOTE AT ALL. The extraction prompt requires source_quote
                    # on every fact and says to DROP a candidate with no supporting
                    # snippet rather than guess -- so a quote-less fact is already a
                    # rule violation, and until now it was banked anyway with
                    # quote_status NULL. That is the worst of both worlds: NULL
                    # reads as "not checked yet", so an unverifiable claim became
                    # indistinguishable from a pending one. Measured on a live vet
                    # corpus: 25 of 1,010 facts, 15 of them from a single session.
                    #
                    # Marked, not dropped, and marked EXPLICITLY. Dropping would
                    # discard content that is often true, while a null status hides
                    # it; 'no_quote' lets any consumer filter deliberately and lets
                    # the operator see the class growing. Enforcement is mechanical
                    # here rather than another line of prompt, because the prompt
                    # has already asked twice.
                    quote_status = "no_quote"
                elif source_quote:
                    if not self._verify_source_quote:
                        quote_status = "unverified"
                    else:
                        verdict = self._attest_quote(source_quote, transcript)
                        if verdict == "specific_mismatch":
                            quotes_dropped += 1
                            logger.debug(
                                "Quote attestation: DROPPED fact (fabricated specific) "
                                "- %s | quote=%s", content[:60], source_quote[:60],
                            )
                            continue   # user policy: drop on hard-specific mismatch
                        if verdict == "unattested":
                            source_quote = None   # keep the fact, flag the weak quote
                        quote_status = verdict

                # === Ref verification ============================================
                # The url channel gets its own gate, and it is STRICTER than the
                # quote channel: a quote that drifts is a weak citation, but a url
                # pointing at a page that does not contain the claim is a FALSE
                # one, and a fabricated citation is the single failure that would
                # make a clinical corpus unsafe to hand to a professional.
                #
                # Measured justification for the strictness: on this corpus a
                # bag-of-words matcher over full-page candidates produced 74
                # "recovered" refs of which roughly 67 were false. Requiring a
                # verbatim run instead cut that to 7. See source_index.verify_ref.
                #
                # DROP THE REF, KEEP THE FACT. Losing a citation costs recall;
                # inventing one costs trust. Only refs the extractor chose from the
                # supplied index are checked -- a `synthesized:` stamp is our own
                # bookkeeping, not a claim about a page, so it passes through.
                # The quote as the extractor emitted it. Attestation may have nulled
                # source_quote above (verdict 'unattested'), but that verdict is
                # about the TRANSCRIPT -- a quote absent from the agent's paraphrase
                # can still be verbatim from a page the agent read, which is exactly
                # the case worth citing. So match on the original.
                _raw_quote = (fact.get("source_quote")
                              or source_quote or content or "")

                if (_ref_bodies and source_ref
                        and source_ref.startswith("http")):
                    if not source_index.verify_ref(_raw_quote, source_ref,
                                                   _ref_bodies):
                        logger.debug(
                            "Ref verification: DROPPED unearned ref %s for '%s'",
                            source_ref[:60], content[:50])
                        refs_dropped += 1
                        source_ref = None
                    else:
                        refs_verified += 1

                # MECHANICAL ATTACHMENT -- the primary path.
                #
                # Asking the model for the url does not work. Offering a
                # `url | title` index in the extraction prompt produced ZERO refs
                # across 28 facts, including a real block with 30 urls on offer. The
                # only case that worked had urls sitting inline beside each fact,
                # where the model copied a 100-char url correctly 3/3 -- so the
                # failure was ASSOCIATION (matching a fact to one of thirty titles),
                # not transcription. Citation markers would have made the copying
                # cheaper and left the association broken.
                #
                # So the system does the association itself: scan the pages this
                # session actually retrieved and attach the one that verbatim
                # contains the quote. Stronger than trusting the model, not just
                # cheaper -- the model can only cite pages it noticed, this checks
                # all of them -- and it cannot fabricate, because the match standard
                # is the same one verify_ref enforces. Ties are refused rather than
                # guessed (see find_ref).
                if _ref_bodies and not source_ref and not synthesis_session:
                    _hit = source_index.find_ref(_raw_quote, _ref_bodies)
                    if _hit:
                        source_ref = _hit[0]
                        refs_found += 1
                        logger.debug("Ref found mechanically: %s (%s) for '%s'",
                                     source_ref[:60], _hit[1], content[:50])

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
                    # per-fact here anymore - it runs once at the end of the epoch via
                    # self._blind_reconcile() (§14 6a / write-path completeness), the single
                    # mechanism that ALSO catches facts created store-side (abstraction/gist/
                    # procedural) and backfills a store on first blind-enable.
                    # Phase 5a: extract (subject, relation, object) triples from a
                    # FRESHLY added fact into the relation graph (idempotent, but only
                    # on 'added' to skip redundant work on reinforcement). Gated +
                    # non-fatal via the helper so it never blocks consolidation.
                    # DEFERRED (2026-07-25): this is an LLM call per added fact
                    # (timeout min(_reason_timeout, 60) each), and it was the
                    # dominant term in a 111-121s lock hold across five lanes.
                    # It writes only fact_relations rows for a fact_id THIS
                    # process just created, so it needs no cross-process
                    # exclusion - SQLite's WAL + 30s busy_timeout covers the
                    # write. Collect now, run after the lock is released.
                    if self._enable_relations and action == "added" and fid and fid > 0:
                        _deferred_relations.append((fid, content, entities))
                except Exception as e:
                    logger.error(f"Failed to ingest fact '{content[:40]}...': {e}", exc_info=True)

            # 5b. Blind-tier write-path completeness (§14 6a): mirror this epoch's new facts'
            # embedding/entities/HRR into the encrypted tables (no-op off the blind path; reads
            # the just-written plaintext back, no Ollama). One mechanism for every write path.
            self._blind_reconcile()

            # 6. Update cycle counter - atomic in-DB increment. Never write
            # back the process-cached value: other processes share this DB
            # and may have advanced the clock since we loaded it.
            self._memory_cycle = self._store.increment_cycle("memory_cycle")
            logger.info(
                f"✅ Memory Cycle {self._memory_cycle} completed - "
                f"{len(extracted_facts)} facts processed"
                + (f", {quotes_dropped} dropped (unverified specifics)" if quotes_dropped else "")
                + (f", refs {refs_found} matched/{refs_verified} verified/{refs_dropped} unearned"
                   if (refs_found or refs_verified or refs_dropped) else "")
            )
 
            # Determine whether a dream cycle should follow, but don't decide
            # inside the lock - we release first so on_session_end(force_blocking)
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
            if _t_write_start is not None:
                _t_write = time.monotonic() - _t_write_start
            _flock.release()
            self._consolidation_lock.release()   # ← released BEFORE dream cycle

        # 5c. Relation extraction, OUTSIDE both locks (see the deferral note in
        # the fact loop). One LLM call per newly added fact; non-fatal, and each
        # touches only its own fact_id, so a slow or hung relation model can no
        # longer stall every other lane's consolidation tail.
        _t_rel_start = time.monotonic()
        for _fid, _content, _entities in _deferred_relations:
            try:
                self._extract_relations_for_fact(_fid, _content, _entities)
            except Exception as e:
                logger.debug("Deferred relation extraction failed for fact %s: %s",
                             _fid, e)
        _t_rel = time.monotonic() - _t_rel_start

        # One line that makes the next contention question measurable instead of
        # arguable: lock_held is what OTHER lanes actually queue behind.
        if _t_extract or _t_write:
            logger.info(
                "Consolidation timing: extract %.1fs (unlocked) | lockwait %.1fs "
                "| write %.1fs (LOCK HELD) | relations %.1fs (unlocked, %d facts) "
                "| epoch %.1fs",
                _t_extract, _t_lockwait, _t_write, _t_rel,
                len(_deferred_relations), time.monotonic() - _t_epoch)

        # Dream cycle runs outside the consolidation lock.
        # _dream_lock inside _run_dream_cycle() prevents concurrent dream cycles.
        # on_session_end suppresses this and runs its own single dream cycle to
        # avoid back-to-back duplicate maintenance at shutdown.
        if trigger_dream and not suppress_dream:
            logger.info("Triggering Dream Cycle after waking cycle")
            self._run_dream_cycle()




    # ====================== DREAM CYCLE (Hebbian Maintenance) ======================
    # Meta key for the cross-process dream-cadence claim. Lives in the DB, not on
    # the instance: the whole point is that twelve separate hermes processes agree
    # on when the last dream happened.
    _DREAM_CLAIM_KEY = "last_dream_memory_cycle"

    def _run_dream_cycle(self, wait_lock: bool = False,
                         respect_cadence: bool = False) -> None:
        """Full Hebbian Dream Cycle - decay, promotion, abstraction, conflict resolution.

        ``wait_lock``: when another PROCESS is finalizing this DB, a scheduled
        dream (mid-session trigger, manual tool call) defers - the next trigger
        dreams. The session-end dream passes True and waits its turn so every
        session still gets its maintenance pass under N-way concurrency.

        ``respect_cadence``: honour dream_every_n_consolidations even on the
        session-end path. OFF by default so a manual `rlm_dream` tool call or an
        explicit maintenance request always dreams - a user who asks for a dream
        should get one.

        WHY IT EXISTS. on_session_end used to dream UNCONDITIONALLY, which meant
        dream_every_n_consolidations - documented as "Dream cycle every N
        consolidations" - had no effect on the path that does almost all the
        dreaming in batch use. With one agent that is invisible (one session, one
        dream). With twelve concurrent overnight lanes x 25 blocks it is 300
        dreams, each holding the FinalizeLock for ~18s measured, so roughly 90
        MINUTES per night of serialized lock time that every other lane queues
        behind. At cadence 12 that is ~25 dreams, about 7.5 minutes.

        Nothing is lost by dreaming less during bulk ingest: decay and promotion
        are near-no-ops while nothing is being recalled, and conflict detection
        over a fuller corpus is BETTER than over a corpus that grew by 20 facts
        since the last pass.

        The claim is read-and-written INSIDE the FinalizeLock, so two processes
        cannot both decide they are the one to dream.

        Step order matters:
          0. increment_tier_cycles  - advance tier-dwell counters (short/mid)
          0.5 reencode_hrr_if_needed - one-shot encoding migration (self-gating)
          0.6 reembed_if_needed     - one-shot re-embed on embed_model change (self-gating, P4d)
          1. apply_cycle_decay      - bleed short/mid resonance (long exempt)
          1.5 apply_staleness_decay - extra decay for weak+stale facts (gated; off by default)
          2. apply_conflict_decay   - bleed conflicting facts; resurrect tie-breaker
          3. promote_facts          - mid→long first, then short→mid; needs resonance + dwell
          4. abstraction pass       - every N cycles, LLM generalization (gated)
          4.5 distill_procedural    - every N cycles, generalize tool episodes → procedural facts (gated)
          4.9 supersede_conflict_losers - retire conflict losers as history (gated)
          4.95 consolidate_before_prune - gist dying earned facts (gated; off by default)
          5. prune_weak_facts       - delete resonance <= 0 (superseded losers kept)
          6. free_conflict_winners  - clear conflict lock from sole survivors
          7. resolve_hrr_conflicts  - detect new conflicts in long-tier facts
          8. prune_episodes         - keep only N most recent sessions (+ episode_max_rows cap)
          8.5 prune_tool_episodes   - bound the raw tool-episode log (gated)
          9. gc_orphan_entities     - drop entity rows no longer linked to any fact
          10. enforce_long_tier_cap - evict weakest long facts beyond max_long_facts (gated)
          11. memory-health audit   - read-only health snapshot every N cycles (gated)
          12. _blind_reconcile      - mirror this cycle's new facts into the blind tables (blind mode only)
        """
        if not self._store or not self._write_enabled:
            logger.debug(
                "Dream cycle skipped - store missing or writes disabled "
                "for this agent context (non-primary)."
            )
            return
        if not self._dream_lock.acquire(blocking=False):
            return
        # Cross-process serialization (see _run_consolidation_epoch / proc_lock).
        _flock = FinalizeLock(self._store.db_path)
        if not _flock.acquire(FINALIZE_LOCK_WAIT if wait_lock else 0.0):
            logger.info("Dream cycle deferred - another process is finalizing this DB")
            self._dream_lock.release()
            return

        # Cadence claim, INSIDE the lock so it is a claim and not a guess: two
        # processes reaching this line cannot both conclude the dream is due.
        if respect_cadence:
            _every = int(getattr(self, "_dream_every_n_consolidations", 2) or 1)
            if _every > 1:
                _now_mc = self._store.get_cycle_counts()[0]
                _last = self._store.get_meta_int(self._DREAM_CLAIM_KEY, -1)
                if _last >= 0 and (_now_mc - _last) < _every:
                    logger.info(
                        "Dream cycle skipped by cadence: %d consolidation(s) "
                        "since cycle %d, dream_every_n_consolidations=%d",
                        _now_mc - _last, _last, _every)
                    _flock.release()
                    self._dream_lock.release()
                    return
                # Claim it now, while still holding the lock. Recording BEFORE the
                # work means a crashed dream costs one skipped cadence slot rather
                # than letting every lane re-dream the same cycle on restart.
                self._store.set_meta_int(self._DREAM_CLAIM_KEY, _now_mc)

        # === 0. Dwell + migrations ===
        try:
            # Atomic in-DB increment (multi-process safe; see increment_cycle).
            self._dream_cycle_count = self._store.increment_cycle("dream_cycle")
            # The dream's decay/pruning math and batch stamps run on the
            # memory_cycle clock - refresh it from the DB too, so a stale
            # cached value can't under-age facts in a long-lived process.
            self._memory_cycle = self._store.get_cycle_counts()[0]
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

            # 1. Exponential decay - stronger memories forget slower (contested facts held in
            #    limbo when conflict_limbo is on - see step 2)
            self._store.apply_cycle_decay(protect_conflicts=self._conflict_limbo,
                                          peak_discount=self._surprise_decay_discount,
                                          importance_discount=self._importance_decay_discount)

            # 1.5 Phase 2 'use it or lose it' - extra decay for weak AND long-
            #     unconfirmed facts (gated; default off via stale_decay_boost=0).
            if self._stale_decay_boost > 0:
                self._store.apply_staleness_decay(
                    self._memory_cycle, self._stale_decay_boost,
                    self._freshness_halflife_cycles,
                )

            # 1.6 Procedural staleness - perishable tool-use knowledge fades if it
            #     stops being re-confirmed (conflict-excluded + long-tier-exempt, so
            #     otherwise immortal). Lets a superseded search/URL rule fade once a
            #     better/pinned rule wins recall. Gated (bleed <= 0 -> no-op).
            if getattr(self, "_procedural_staleness_bleed", 0.0) > 0:
                faded = self._store.apply_procedural_staleness_decay(
                    self._memory_cycle,
                    self._procedural_staleness_bleed,
                    getattr(self, "_procedural_staleness_grace_cycles", 5),
                )
                if faded:
                    logger.debug("Procedural staleness: bled %d unconfirmed procedural fact(s)", faded)

            # 2. Conflict resolution. Limbo (default): DON'T auto-bleed - hold contested facts in
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

            # 4.5 Procedural Memory Distillation - generalize raw tool episodes
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

            # 5. Prune completely faded facts - non-superseded losers deleted here
            self._store.prune_weak_facts(self._forget_after_dormant_cycles,
                                         protect_conflicts=self._conflict_limbo)
 
            # 6. Free conflict winners - NOW safe to call because losers are gone.
            #    Uses LatticeStore.free_conflict_winners() which holds _lock internally.
            #    Previously this was an unguarded self._store._conn.execute() call here,
            #    which bypassed the lock and risked collision with parallel sessions.
            self._store.free_conflict_winners()
 
            # 7. Detect new conflicts in long-term memory via HRR algebra
            self._resolve_long_term_conflicts()
 
            # 7.9 Consolidation debt: flag sessions with substantial episodes and
            #     ZERO born facts BEFORE the pruner runs, so their episodes are
            #     exempt from step 8 and the evidence survives (gated). Stamped on
            #     the dream_cycle clock; the retry itself runs post-dream, outside
            #     these locks (see _reconsolidate_debt).
            if getattr(self, "_reconsolidate_zero_fact_sessions", False):
                self._store.flag_consolidation_debt(
                    self._dream_cycle_count, exclude_session=self._session_id)

            # 8. Keep episode table bounded - session window (prune_keep_sessions)
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

            # 11. Cycle-based memory-health audit (read-only) - log a snapshot
            #     every N dream cycles. Strictly dream-cycle-driven, no timers.
            if (self._health_check_every_n_dream_cycles > 0
                    and self._dream_cycle_count % self._health_check_every_n_dream_cycles == 0):
                self._run_memory_health_audit()

            # 12. Blind-tier write-path completeness (§14 6a): mirror facts created store-side
            #     during this cycle (abstraction / gist / procedural distillation - none of which
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
            _flock.release()
            self._dream_lock.release()

        # Consolidation-debt retry - MUST run after the locks above are released:
        # the retry re-enters _run_consolidation_epoch, which takes its own
        # consolidation + finalize locks and would deadlock/skip inside the dream.
        if getattr(self, "_reconsolidate_zero_fact_sessions", False):
            try:
                self._reconsolidate_debt()
            except Exception as e:
                logger.debug("Debt reconsolidation failed (non-fatal): %s", e)

    def _reconsolidate_debt(self) -> None:
        """Retry the consolidation epoch for ONE seasoned open debt per dream cycle.

        The substrate never discards an experience it has not digested: a debt is a
        session with substantial episodes and zero born facts (flagged in dream step
        7.9, its episodes exempt from pruning). Bounded on every axis - one session
        per dream cycle, reconsolidation_max_attempts retries per session, and the
        one-cycle seasoning delay in get_open_consolidation_debts. The receipt
        check (born facts on source_session) runs BEFORE the retry so a debt that
        settled organically costs zero LLM calls, and again AFTER so recovery is
        judged at the substrate, not by the epoch's return path. suppress_dream
        stops the retried epoch from firing a recursive dream cycle."""
        store = self._store
        if store is None or not self._write_enabled:
            return
        max_attempts = max(1, int(getattr(self, "_reconsolidation_max_attempts", 2)))
        for debt in store.get_open_consolidation_debts(self._dream_cycle_count, limit=1):
            sid = debt["session_id"]
            if store.session_born_fact_count(sid) >= 1:
                store.settle_consolidation_debt(sid, "organic", self._dream_cycle_count)
                logger.info("Consolidation debt: %s settled organically", sid)
                continue
            attempts = store.record_debt_attempt(sid, self._dream_cycle_count)
            logger.info(
                "Consolidation debt: retrying epoch for %s (attempt %d/%d, %d episode rows)",
                sid, attempts, max_attempts, debt.get("episode_rows", 0))
            self._run_consolidation_epoch(sid, suppress_dream=True)
            born = store.session_born_fact_count(sid)
            if born >= 1:
                store.settle_consolidation_debt(sid, "recovered", self._dream_cycle_count)
                logger.info("Consolidation debt: RECOVERED %d fact(s) from %s", born, sid)
            elif attempts >= max_attempts:
                store.settle_consolidation_debt(sid, "exhausted", self._dream_cycle_count)
                logger.warning(
                    "Consolidation debt: %s EXHAUSTED after %d attempt(s), still 0 facts; "
                    "episodes released to normal pruning", sid, attempts)

    def _reembed_if_needed(self) -> int:
        """P4d migration: re-embed all facts to the configured embed_model if it changed.

        Delegates to LatticeStore.reembed_if_needed, passing the live embedder
        (retriever._get_embedding) and the configured model. Self-gating via meta['embed_model']
        - a no-op on every cycle except the first after an embed_model switch on an existing store
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
        plaintext store - NO Ollama. Delegates to the BlindTier collaborator (blind_tier.py), which
        owns the writers + entity store; a no-op off the blind path (``self._blind_tier`` is None).
        Called at the end of the consolidation epoch + the dream cycle. Returns embedding cts written."""
        bt = getattr(self, "_blind_tier", None)
        return bt.reconcile(self._store, limit) if (bt is not None and self._store) else 0

    def _run_memory_health_audit(self) -> dict:
        """Read-only memory-health snapshot. Logs at INFO on the triggering dream
        cycle (so it lands in normal agent logs) and returns the dict. No side
        effects - also reused by the memory_audit tool action.
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
                "abstractions: %d (evidence_gone %d) | tool episodes: %d (undistilled %d) | "
                "consolidation debt: %d open / %d exhausted%s",
                self._dream_cycle_count, health.get("total_facts", 0),
                health.get("long_tier_facts", 0), health.get("near_cap_threshold", 0),
                health.get("near_cap_facts", 0), health.get("active_conflict_groups", 0),
                health.get("conflicted_facts", 0), health.get("total_entities", 0),
                health.get("orphan_entities", 0), health.get("abstractions_tracked", 0),
                health.get("abstractions_evidence_gone", 0),
                health.get("tool_episodes_total", 0), health.get("tool_episodes_undistilled", 0),
                health.get("consolidation_debt_open", 0),
                health.get("consolidation_debt_exhausted", 0),
                "  [DEGRADED - FTS-only]" if health.get("degraded") else "",
            )
            return health
        except Exception as e:
            logger.debug("Memory health audit failed (non-fatal): %s", e)
            return {}


    def _perform_abstraction_pass(self) -> None:
        """Memory Abstraction Layer - merges similar facts into higher-level generalizations."""
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
                # Embeddings go to the EMBED server, not the reasoning one. Those are
                # the same host on a plain Ollama setup, and different the moment the
                # reason model is fronted by an OpenAI-compatible endpoint: a vLLM shim
                # serves the generate route but 404s the embeddings route, and that
                # failure is swallowed -- abstractions then insert with no vector.
                embed_endpoint=self._ollama_endpoint_embed,
            )


    def _consolidate_before_prune(self) -> None:
        """Phase 4 - gist dying-but-once-important facts before they prune.

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
                    embed_endpoint=self._ollama_endpoint_embed,
                )
            except Exception as e:
                logger.debug("Gist consolidation failed (non-fatal): %s", e)


    def _extract_relations_for_fact(self, fact_id: int, content: str, entities: list) -> None:
        """Phase 5a - extract & store relational triples for one new fact.

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
                reason_model=self._relation_model,
                ollama_endpoint=self._ollama_endpoint_relation,
                use_llm=self._relation_extract_llm,
                llm_prompt=getattr(self, "_relation_llm_prompt", self._relation_prompt),
                vocabulary=getattr(self, "_relation_vocabulary", None),
                examples=getattr(self, "_relation_examples", None),
                aliases=getattr(self, "_entity_aliases", None),
                require_entity=getattr(self, "_relation_require_entity", False),
            )
        except Exception as e:
            logger.debug("Relation extraction failed (non-fatal): %s", e)


    def _generate_session_narrative(self, session_id: str) -> None:
        """Phase 8 - write a durable one-paragraph narrative gist of the session.

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
        similarity - those are likely contradictions. Pairs are grouped under
        a conflict_group_id and enter the Duel-to-the-Death decay loop on
        the next dream cycle.

        When conflict_llm_adjudication is on, candidate pairs that survive the
        store's heuristic gates get a reason-model second opinion before being
        locked (the adjudicator lives HERE so the store stays model-free).
        """
        if self._store and _HRR_AVAILABLE:
            adjudicator = (self._conflict_adjudicator
                           if getattr(self, "_conflict_llm_adjudication", True)
                           else None)
            self._store.resolve_hrr_conflicts(adjudicator=adjudicator)


    def _conflict_adjudicator(self, content_a: str, content_b: str) -> "bool | None":
        """Reason-model second opinion for one conflict candidate pair.

        Returns True (contradict - lock the group), False (compatible - skip),
        or None (unknown - the store fails open and flags, preserving the old
        conservative behavior). Kept deliberately cheap: one short generate call
        at temperature 0, capped at 60s so a slow model cannot stall the dream
        cycle; any failure is a debug-level non-event."""
        prompt = (
            "Do these two statements CONTRADICT each other - that is, can they "
            "NOT both be true at the same time?\n"
            "Statements about DIFFERENT subjects are NOT contradictions (two "
            "different functions can both exist). Statements about the same "
            "subject under different versions/conditions are NOT contradictions "
            "when each names its own scope.\n\n"
            f"A: {content_a}\n"
            f"B: {content_b}\n\n"
            'Answer ONLY with JSON: {"contradict": true} or {"contradict": false}'
        )
        payload = {
            "model": self._reason_model,
            "prompt": prompt,
            "stream": False,
            # See the note at the extraction payload. This call runs with the
            # FinalizeLock HELD, so a reason model that loses its answer into an
            # unclosed think block stalls every OTHER lane too, not just this one.
            "think": False,
            "options": {"temperature": 0.0},
        }
        try:
            result = _ollama_post_with_retry(
                f"{self._ollama_endpoint_reason}/api/generate",
                payload,
                timeout=min(float(getattr(self, "_reason_timeout", 300.0)), 60.0),
            )
            text = self._store._clean_llm_json(
                (result.get("response") or "").strip())
            m = re.search(r'"contradict"\s*:\s*(true|false)', text, re.IGNORECASE)
            if m:
                return m.group(1).lower() == "true"
            low = text.lower()
            if "true" in low and "false" not in low:
                return True
            if "false" in low and "true" not in low:
                return False
            return None
        except Exception as e:
            logger.debug("Conflict adjudication call failed (fail-open): %s", e)
            return None
