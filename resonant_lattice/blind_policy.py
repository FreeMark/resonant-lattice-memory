"""blind_policy.py — Tier-1 blind-store runtime SCOPE policy (E6, roadmap §7.2).

HE bounds PROVENANCE ("results of a query I ran") but NOT SCOPE (how much a query
returns). A hijacked agent holding the eval key could homomorphically "select
everything" and have the store re-encrypt it one query at a time — the math cannot stop
this; policy must (roadmap §7.2, tension #2). This leaf enforces the conservative,
auditable scope bounds the roadmap commits to, all on the LOGICAL memory cycle (never
wall-clock, per the cross-cutting principles), and keeps a re-encryption audit log the
user can review.

Pure-Python (no openfhe, no SQLite) — a leaf like store_common, fully unit-testable on
any host. The PRE/threshold crypto lives in he_crypto (BlindPRE / ThresholdAudit); this
module is the policy that bounds the blast radius the crypto deliberately does not."""

import logging
import secrets
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# Conservative defaults — a single local user's blind recall, not a service. Tunable.
DEFAULT_TOPK_CEILING = 16              # a recall never re-encrypts more than this many hits
DEFAULT_PER_CYCLE_QUERY_CAP = 8       # max blind recalls the agent may run per memory cycle
DEFAULT_PER_CYCLE_REENCRYPT_CAP = 64  # max results re-encrypted per cycle (sum of k)


class ScopeExceededError(RuntimeError):
    """Raised when a blind-recall request would exceed a per-cycle scope cap."""


class TokenError(RuntimeError):
    """Raised when a re-encryption is attempted with an unknown, exhausted, or replayed token."""


@dataclass(frozen=True)
class ReEncryptEvent:
    """One audited grant: the store re-encrypted ``k`` results for ``query_token`` in
    logical memory ``cycle``."""
    cycle: int
    query_token: str
    k: int


class ReEncryptAuditLog:
    """Append-only record of every re-encryption the store performs for the agent — the
    user-reviewable trail that makes the §7.2 policy bound auditable rather than implicit."""

    def __init__(self) -> None:
        self._events: List[ReEncryptEvent] = []

    def record(self, cycle: int, query_token: str, k: int) -> None:
        self._events.append(ReEncryptEvent(int(cycle), query_token, int(k)))

    def events(self) -> List[ReEncryptEvent]:
        return list(self._events)

    def query_count(self, cycle: int) -> int:
        return sum(1 for e in self._events if e.cycle == cycle)

    def total_reencrypted(self, cycle: Optional[int] = None) -> int:
        return sum(e.k for e in self._events if cycle is None or e.cycle == cycle)


class ScopeLimiter:
    """Enforces conservative per-cycle scope on blind recall (roadmap §7.2).

    - **top-k ceiling**: a single recall can never request more than ``topk_ceiling``
      results — the query-shape constraint ("top-k vs one probe," never "return all").
    - **per-cycle query cap**: bounds how many blind recalls the agent runs per cycle.
    - **per-cycle re-encryption cap**: bounds the TOTAL results re-encrypted per cycle, so
      an agent cannot exfiltrate the store one small query at a time.

    A breach RAISES (the recall is refused, not silently truncated) so it is loud and
    lands in the audit log's absence rather than a quiet partial leak. ``authorize``
    returns a query token that binds the subsequent re-encryptions to this one approved
    query (provenance), which the store checks before applying ``rk_storage->agent``."""

    def __init__(self, topk_ceiling: int = DEFAULT_TOPK_CEILING,
                 per_cycle_query_cap: int = DEFAULT_PER_CYCLE_QUERY_CAP,
                 per_cycle_reencrypt_cap: int = DEFAULT_PER_CYCLE_REENCRYPT_CAP,
                 audit: Optional[ReEncryptAuditLog] = None) -> None:
        self.topk_ceiling = int(topk_ceiling)
        self.per_cycle_query_cap = int(per_cycle_query_cap)
        self.per_cycle_reencrypt_cap = int(per_cycle_reencrypt_cap)
        self.audit = audit or ReEncryptAuditLog()

    def authorize(self, cycle: int, k: int) -> str:
        """Authorize a blind recall of ``k`` results in memory ``cycle``; return a binding
        query token and record the grant. Raises ScopeExceededError if any cap would be
        exceeded."""
        if k <= 0:
            raise ValueError("k must be positive")
        if k > self.topk_ceiling:
            raise ScopeExceededError(f"k={k} exceeds top-k ceiling {self.topk_ceiling}")
        if self.audit.query_count(cycle) >= self.per_cycle_query_cap:
            raise ScopeExceededError(
                f"per-cycle query cap {self.per_cycle_query_cap} reached for cycle {cycle}")
        if self.audit.total_reencrypted(cycle) + k > self.per_cycle_reencrypt_cap:
            raise ScopeExceededError(
                f"per-cycle re-encryption cap {self.per_cycle_reencrypt_cap} would be exceeded "
                f"(cycle {cycle}: {self.audit.total_reencrypted(cycle)} + {k})")
        token = secrets.token_hex(8)
        self.audit.record(cycle, token, k)
        logger.debug("blind recall authorized: cycle=%s k=%s token=%s", cycle, k, token)
        return token


class BlindReEncryptGate:
    """STORE-side re-encryption gate (roadmap 6c) — the store-side half of the §7.2 bound.

    The store performs ``ReEncrypt`` ONLY for a token the agent presents from a
    ``ScopeLimiter.authorize()`` grant, and only up to that grant's ``k`` results
    (single-use budget). This binds every re-encryption to one approved query (provenance)
    and refuses both replay (re-registering a spent token) and over-spend (more ReEncrypts
    than authorized) — so even with a valid eval key the agent cannot have the store
    re-encrypt more than policy allows. Pure-Python; the actual ``ReEncrypt`` crypto stays
    in ``he_crypto.BlindRecallPRE.reencrypt_score``. The matching audit row is persisted by
    the store (``record_reencrypt_event``) so the trail is substrate-checkable."""

    def __init__(self) -> None:
        self._budget = {}   # token -> remaining re-encryptions
        self._seen = set()  # every token ever registered (no re-registration / replay)

    def register(self, token: str, k: int) -> None:
        """The store accepts a fresh token good for exactly ``k`` re-encryptions. A token may
        be registered only once — replaying a whole grant is refused."""
        if not token or int(k) <= 0:
            raise TokenError("token must be non-empty and k positive")
        if token in self._seen:
            raise TokenError("token already registered (replay refused)")
        self._seen.add(token)
        self._budget[token] = int(k)

    def spend(self, token: str) -> None:
        """Consume one re-encryption against ``token`` (call immediately before each
        ``ReEncrypt``); raise if the token is unknown/unregistered or its budget is spent."""
        rem = self._budget.get(token)
        if rem is None:
            raise TokenError("unknown or unregistered re-encryption token")
        if rem <= 0:
            raise TokenError("token budget exhausted")
        self._budget[token] = rem - 1

    def remaining(self, token: str) -> int:
        """Re-encryptions still authorized for ``token`` (0 if unknown/exhausted)."""
        return int(self._budget.get(token, 0))
