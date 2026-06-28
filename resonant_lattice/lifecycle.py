"""lifecycle.py — LifecycleMixin: session switch/end, shutdown, and the
optional ABC hooks (on_memory_write, on_pre_compress, on_delegation).

Mixed into LatticeMemoryProvider; relies on the composite for self._store/
_retriever, locks/threads, config attrs, and sibling methods via self."""

import logging
import threading
from typing import Any, Dict, List, Optional

from store_common import hrr, _HRR_AVAILABLE

logger = logging.getLogger(__name__)


class LifecycleMixin:

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        """Refresh cached per-session state when the agent rotates session_id.

        Fires on /resume, /branch, /reset, /new, and context compression.
        Without this, episodes/facts keep being written under the stale
        session id captured in initialize(). On a genuine reset we also flush
        the per-session counters. Cycle counters (_memory_cycle,
        _dream_cycle_count) are global Hebbian state and intentionally preserved.
        """
        if not new_session_id:
            return
        self._session_id = new_session_id
        if reset:
            # Drain any in-flight ingest from the previous logical conversation
            # before zeroing counters, so its episodes don't bleed into the new one.
            if self._last_ingest_thread is not None and self._last_ingest_thread.is_alive():
                self._last_ingest_thread.join(timeout=5.0)
            self._turn_count = 0
            self._last_ingest_thread = None
            self._prefetch_cache.clear()
            logger.debug("on_session_switch: reset per-session state for %s", new_session_id)


    def on_session_end(self, messages) -> None:
        """Final cleanup + one last dream cycle when the chat session ends.
 
        Join order:
          1. Wait for the last ingest thread to finish writing episodes (5s timeout).
             Without this, the forced consolidation below might read an episode
             table that's missing the final turn's content.
          2. Run forced blocking consolidation — extracts facts from the complete
             episode log including the final turn.
          3. Run dream cycle on a non-daemon thread (join 60s) so Hebbian
             maintenance completes before the process exits.
        """
        if not self._store:
            return
 
        # Step 1: Drain the last ingest thread so final episodes are on-disk.
        if self._last_ingest_thread is not None and self._last_ingest_thread.is_alive():
            logger.debug("on_session_end: waiting for last ingest thread to flush episodes…")
            self._last_ingest_thread.join(timeout=5.0)
            if self._last_ingest_thread.is_alive():
                logger.warning(
                    "on_session_end: ingest thread still running after 5s — "
                    "final turn's episodes may be missing from consolidation."
                )
 
        # Step 2: Forced blocking consolidation (waits up to 50s for the lock
        # — long enough to outlast an in-flight 45s Ollama call).
        # Suppress its inline dream cycle — we run exactly one in Step 3.
        self._run_consolidation_epoch(self._session_id, force_blocking=True, suppress_dream=True)

        # Step 2.5: Phase 8 — durable autobiographical summary of this session,
        # generated from the (now-complete) episode log BEFORE the dream cycle can
        # prune episodes. Gated by enable_narrative; non-fatal.
        if self._enable_narrative:
            self._generate_session_narrative(self._session_id)

        # Step 3: Dream cycle on a non-daemon thread with generous timeout.
        t = threading.Thread(target=self._run_dream_cycle, daemon=False)
        self._last_dream_thread = t   # tracked so shutdown() drains it before close()
        t.start()
        t.join(timeout=60.0)


    def shutdown(self) -> None:
        # Drain any in-flight turn ingest before closing the handle — shutdown
        # can be reached without on_session_end (which normally does this).
        if self._last_ingest_thread is not None and self._last_ingest_thread.is_alive():
            self._last_ingest_thread.join(timeout=5.0)
        # Drain an in-flight dream cycle BEFORE close(). Otherwise close() could
        # grab self._lock between two dream-cycle steps and shut the connection,
        # so the next step would log a 'closed database' error and skip remaining
        # Hebbian maintenance. The dream thread is non-daemon (the process already
        # waits for it at exit) — this only sequences close() after it. Bounded so
        # a slow LLM abstraction can't hang shutdown: if still running after the
        # grace, skip close() and let the thread + process-exit cleanup release
        # the handle (SQLite WAL recovers to the last commit on next open).
        dream = self._last_dream_thread
        if dream is not None and dream.is_alive():
            dream.join(timeout=60.0)
            if dream.is_alive():
                logger.warning(
                    "shutdown: dream cycle still running after grace — leaving the "
                    "DB handle open for the non-daemon thread to finish; it will be "
                    "released on process exit."
                )
                return
        if self._store:
            self._store.close()


    # ------------------------------------------------------------------
    # Optional ABC hooks
    # ------------------------------------------------------------------
    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mirror built-in MEMORY.md / USER.md writes into the lattice store.

        Keeps the structured store in sync with the agent's flat built-in
        memory. add/replace ingest the content as a fact; remove is a no-op
        (no reliable content→id map, and decay will retire stale facts).
        Runs in the background and is gated to primary write contexts.
        """
        if not (self._write_enabled and self._store and self._retriever):
            return
        if action not in ("add", "replace") or not content or not content.strip():
            return
        # Phase E policy gate: the same self/infra denylist applies to mirrored
        # builtin-memory writes, so autonomous self-referential chatter can't
        # sneak in through the MEMORY.md/USER.md mirror path either.
        if self._gate_self_writes and self._is_self_referential_infra(content):
            logger.debug("Self-write gate: skipped self/infra mirror: %s", content[:60])
            return

        def _ingest() -> None:
            try:
                emb = self._retriever._get_embedding(content)
                if not emb:
                    return
                entities = self._store._extract_entities(content)
                hrr_vec = hrr.encode_fact(content, entities, dim=self._hrr_dim) if _HRR_AVAILABLE else None
                self._store.add_or_reinforce_fact(
                    content, emb, f"builtin_{target}", self._session_id,
                    hrr_vector=hrr_vec, entities=entities,
                )
            except Exception as e:
                logger.debug("on_memory_write mirror failed: %s", e)

        threading.Thread(target=_ingest, daemon=True).start()


    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract + surface memories before context compression discards turns.

        Episodes for these turns are already persisted by sync_turn, so the
        consolidation kick is a background convenience (non-blocking; the
        consolidation lock's non-blocking acquire skips if one is running).
        Returns a block of the strongest relevant facts for the compression
        summary prompt, so the compressor keeps what we already know.
        """
        if not self._store or not self._retriever:
            return ""
        if self._write_enabled:
            threading.Thread(
                target=self._run_consolidation_epoch,
                args=(self._session_id,),
                kwargs={"suppress_dream": True},
                daemon=True,
            ).start()
        recent_text = " ".join(
            m.get("content", "") for m in (messages or [])[-6:]
            if isinstance(m.get("content"), str)
            and m.get("role") in ("user", "assistant")
        ).strip()[:2000]
        if not recent_text:
            return ""
        try:
            results = self._retriever.search(recent_text, limit=6)
        except Exception as e:
            logger.debug("on_pre_compress recall failed: %s", e)
            return ""
        if not results:
            return ""
        lines = [
            f"- [{r.get('category', 'general')}] {r['content']}" for r in results
        ]
        return (
            "Durable facts already persisted in Resonant Lattice Memory "
            "(preserve these in the summary; no need to restate supporting detail):\n"
            + "\n".join(lines)
        )


    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Observe a completed subagent delegation as an episodic turn.

        The subagent runs with skip_memory=True, so the parent's provider is
        the only place this outcome can be remembered. Recording it as a
        normal turn lets the existing consolidation pipeline distill durable
        facts from delegation results (e.g. 'the X refactor is complete').
        """
        if not (self._store and self._write_enabled and task):
            return
        tag = f" (subagent {child_session_id})" if child_session_id else ""

        def _bg() -> None:
            try:
                self._store.add_turn(
                    self._session_id,
                    f"[DELEGATED TASK{tag}] {task[:1500]}",
                    (result or "")[:3000],
                )
            except Exception as e:
                logger.debug("on_delegation ingest failed (non-fatal): %s", e)

        threading.Thread(target=_bg, daemon=True).start()
