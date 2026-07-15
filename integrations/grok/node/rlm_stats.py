#!/usr/bin/env python3
"""Read-only stats / health snapshot of the grok agent's OWN lattice. Prints a JSON summary:
tier distribution, pins, cycle clocks, relations, narratives, and pending conflicts."""
import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import build_provider  # noqa: E402


def main():
    try:
        p = build_provider(session_id="stats")
        s = p._store
        st = s.get_stats()
        cyc = s.get_cycle_counts()  # (memory_cycle, dream_cycle)
        c = s._conn
        pinned = c.execute("SELECT COUNT(*) FROM semantic_facts "
                           "WHERE pinned=1 AND superseded_by IS NULL").fetchone()[0]
        narratives = c.execute("SELECT COUNT(*) FROM session_summaries").fetchone()[0]
        try:
            relations = c.execute("SELECT COUNT(*) FROM fact_relations").fetchone()[0]
        except Exception:
            relations = 0
        try:
            conflicts = len(s.get_pending_conflicts(min_age_cycles=0, limit=200))
        except Exception:
            conflicts = 0
        print(json.dumps({
            "ok": True,
            "total_facts": st.get("total_facts"),
            "by_tier": st.get("by_tier"),
            "pinned": pinned,
            "memory_cycle": cyc[0], "dream_cycle": cyc[1],
            "episodes": st.get("total_episodes"),
            "entities": st.get("total_entities"),
            "relations": relations,
            "narratives": narratives,
            "pending_conflicts": conflicts,
        }))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:200]}))


if __name__ == "__main__":
    main()
