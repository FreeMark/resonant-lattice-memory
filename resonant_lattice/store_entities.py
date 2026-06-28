"""store_entities.py — EntitiesMixin: entity extraction (delegates to the
enhanced extractor with a regex fallback), orphan GC, entity-graph reads.

Mixed into LatticeStore; relies on the composite for self._conn/_lock."""

import logging
import re
from typing import List, Dict

from store_common import _ENTITY_EXTRACTOR_AVAILABLE, _extract_entities_fn

logger = logging.getLogger(__name__)


class EntitiesMixin:

    def gc_orphan_entities(self) -> int:
        """Delete entity rows no longer linked to any fact.

        fact_entities rows are removed by the semantic_facts_ad trigger on fact
        delete, but the entities row itself persists. Without this sweep the
        entities table grows without bound across prune/conflict-death cycles
        and slows entity-map joins. Returns the number of orphans removed.
        """
        with self._lock:
            try:
                cur = self._conn.execute(
                    "DELETE FROM entities "
                    "WHERE entity_id NOT IN (SELECT DISTINCT entity_id FROM fact_entities)"
                )
                self._conn.commit()
                removed = cur.rowcount or 0
                if removed:
                    logger.debug("GC: removed %d orphan entities", removed)
                return removed
            except Exception as e:
                logger.debug("Orphan-entity GC failed: %s", e)
                return 0


    # ====================== FAST ENTITY EXTRACTION ======================
    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        """Entity extraction — delegates to the enhanced EntityExtractor.

        Primary: spaCy NER (lazy) + 14-pattern high-precision regex with
        confidence scoring, tech-vocab boosting, and compound rejection.
        Fallback (only if entity_extractor.py failed to import at startup):
        A minimal safe regex set (quoted terms + capitalized phrases).
        This keeps the store self-contained even if the companion module
        is accidentally missing, while eliminating duplicate maintenance.
        """
        if _ENTITY_EXTRACTOR_AVAILABLE and _extract_entities_fn is not None:
            return _extract_entities_fn(text)

        # Minimal safe fallback (only reached if import failed at module load)
        seen: set = set()
        candidates: list = []
        for m in re.finditer(r'"([^"]{2,60})"', text):
            name = m.group(1).strip().lower()   # lowercase: match extractor contract
            if name and name not in seen:
                seen.add(name)
                candidates.append(name)
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
            name = m.group(1).strip().lower()
            if name and name not in seen:
                seen.add(name)
                candidates.append(name)
        return candidates


    def get_entities_for_fact(self, fact_id: int) -> List[str]:
        """Return all entities linked to a fact."""
        with self._lock:
            rows = self._conn.execute("""
                SELECT e.name
                FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE fe.fact_id = ?
                ORDER BY e.name
            """, (fact_id,)).fetchall()
            return [r["name"] for r in rows]


    def get_related_entities(self, entity_name: str, min_shared: int = 2, 
                             limit: int = 20) -> List[Dict]:
        """Entities that co-occur with the given entity in at least min_shared facts."""
        with self._lock:
            rows = self._conn.execute("""
                SELECT e2.name, COUNT(*) as shared_facts
                FROM fact_entities fe1
                JOIN entities e1 ON e1.entity_id = fe1.entity_id
                JOIN fact_entities fe2 ON fe2.fact_id = fe1.fact_id
                JOIN entities e2 ON e2.entity_id = fe2.entity_id
                WHERE LOWER(e1.name) = ? AND e2.entity_id != e1.entity_id
                GROUP BY e2.entity_id
                HAVING COUNT(*) >= ?
                ORDER BY shared_facts DESC, e2.name
                LIMIT ?
            """, (entity_name.lower(), min_shared, limit)).fetchall()
            return [{"entity": r["name"], "shared_facts": r["shared_facts"]} for r in rows]
