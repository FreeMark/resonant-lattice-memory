"""Process-wide serialization gate for memory-layer reasoning-model calls.

Every reasoning-model call the memory layer makes -- consolidation fact
extraction, relation-triple extraction, abstraction, procedural distillation,
conflict adjudication, and session narrative -- funnels through the SAME endpoint
(``reason_model`` / ``ollama_endpoint_reason``). On a rate-limited endpoint (e.g.
an Ollama Cloud plan with a small concurrent-call budget), two memory calls
overlapping steals a slot the primary agent needs for its own generation, which
surfaces to the user as the agent "hanging" while a memory cycle runs.

This module provides a single process-wide gate so the memory layer issues at
most N reasoning calls at once (default 1 = strictly serial). With N=1 the budget
arithmetic holds: primary(1) + compression(1) + memory(1) = 3 concurrent calls.

Design notes:
- Call-level, not operation-level: the gate wraps each individual HTTP call, so a
  background consolidation epoch and a dream cycle still make progress -- their
  ENDPOINT calls simply take turns instead of overlapping. This avoids the
  coarse-lock deadlock where an epoch holding a lock calls relation-extraction
  that would need the same lock.
- Capacity is set once at provider startup via ``set_capacity`` from the
  ``memory_reason_max_concurrency`` config key. ``reason_slot`` reads the live
  semaphore at acquire time and releases the SAME object, so reconfiguration
  before first use is honoured and a swap can never mis-target a release.
"""
from contextlib import contextmanager
import threading

_state_lock = threading.Lock()
_semaphore = threading.Semaphore(1)
_capacity = 1


def set_capacity(n: int) -> None:
    """Set the max number of concurrent memory reasoning calls (>=1). Call once at init."""
    global _semaphore, _capacity
    try:
        n = max(1, int(n))
    except (TypeError, ValueError):
        n = 1
    with _state_lock:
        _semaphore = threading.Semaphore(n)
        _capacity = n


def capacity() -> int:
    return _capacity


@contextmanager
def reason_slot():
    """Hold one memory-reason slot for the duration of a single endpoint call."""
    with _state_lock:
        sem = _semaphore
    sem.acquire()
    try:
        yield
    finally:
        sem.release()
