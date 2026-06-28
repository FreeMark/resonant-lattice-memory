"""store_identity.py — IdentityMixin: Phase-7 deliberate self-model.

Mixed into LatticeStore; uses self._conn/_lock. The agent_identity table is a
SEPARATE store from semantic_facts: the autonomous ingest path
(add_or_reinforce_fact) never touches it, so the ONLY writes are the deliberate
set_self_model / seed_self_model calls here — and the provider gates the
set_self_model tool action to the primary agent context. This structural
separation is the anti-fabrication guarantee: a consolidation/ingest LLM pass can
READ the self-model (to stay consistent) but can never WRITE it, so it can't
become a backdoor for the self-chatter the Phase-E gate suppresses.

The self-model is the POSITIVE counterpart to that suppression gate: a curated,
authoritative record of who the agent is, what it can do, and its standing
relationship with the user — surfaced deterministically in the system prompt
rather than reconstructed via fallible fuzzy recall.
"""

import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

_MAX_IDENTITY_KEY = 80
_MAX_IDENTITY_VALUE = 2000


class IdentityMixin:

    def set_self_model(self, key: str, value: str,
                       current_cycle: Optional[int] = None) -> Optional[Dict]:
        """Deliberately set/update one self-model entry (UPSERT on key).

        Lock-guarded. Normalizes the key to lowercase, trims + length-caps both
        fields, and stamps updated_cycle (the logical clock). Returns the stored
        row dict, or None if the key/value is empty/invalid (caller surfaces an
        error). This is the ONLY mutating entry point exposed to the agent, and the
        provider write-gates its tool action to the primary context.
        """
        if not key or not isinstance(key, str):
            return None
        key = key.strip().lower()[:_MAX_IDENTITY_KEY]
        if not key or value is None:
            return None
        value = str(value).strip()[:_MAX_IDENTITY_VALUE]
        if not value:
            return None
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO agent_identity (key, value, updated_cycle) "
                "VALUES (?, ?, ?)",
                (key, value, current_cycle),
            )
            self._conn.commit()
        return {"key": key, "value": value, "updated_cycle": current_cycle}

    def get_self_model(self, key: Optional[str] = None
                       ) -> Union[Optional[Dict], List[Dict]]:
        """Read the self-model: one entry (key given) or the full curated list.

        Read-only and open (the ingest LLM may read for consistency). With a key,
        returns that row dict or None; without one, the whole self-model ordered by
        key (stable for the system-prompt block).
        """
        with self._lock:
            if key is not None:
                row = self._conn.execute(
                    "SELECT key, value, updated_cycle FROM agent_identity WHERE key = ?",
                    (key.strip().lower(),),
                ).fetchone()
                return dict(row) if row else None
            rows = self._conn.execute(
                "SELECT key, value, updated_cycle FROM agent_identity ORDER BY key"
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_self_model(self, key: str) -> bool:
        """Remove one self-model entry (curation). Returns True if a row was removed."""
        if not key or not isinstance(key, str):
            return False
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM agent_identity WHERE key = ?", (key.strip().lower(),)
            )
            self._conn.commit()
            return (cur.rowcount or 0) > 0

    def seed_self_model(self, items: Dict[str, str],
                        current_cycle: Optional[int] = None,
                        overwrite: bool = False) -> int:
        """Seed identity entries from config. Returns the number of keys written.

        INSERT OR IGNORE by default so a restart NEVER clobbers values the agent has
        curated since first run (the config seed is a starting point, not an
        authority over live self-knowledge). overwrite=True forces a refresh from
        config. Same normalization/caps as set_self_model. Lock-guarded.
        """
        if not items or not isinstance(items, dict):
            return 0
        n = 0
        with self._lock:
            for k, v in items.items():
                if not k or v is None:
                    continue
                k2 = str(k).strip().lower()[:_MAX_IDENTITY_KEY]
                v2 = str(v).strip()[:_MAX_IDENTITY_VALUE]
                if not k2 or not v2:
                    continue
                if overwrite:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO agent_identity (key, value, updated_cycle) "
                        "VALUES (?, ?, ?)",
                        (k2, v2, current_cycle),
                    )
                    n += 1
                else:
                    cur = self._conn.execute(
                        "INSERT OR IGNORE INTO agent_identity (key, value, updated_cycle) "
                        "VALUES (?, ?, ?)",
                        (k2, v2, current_cycle),
                    )
                    n += (cur.rowcount or 0)
            self._conn.commit()
        return n
