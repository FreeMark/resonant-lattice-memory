#!/usr/bin/env python3
"""Read-only DREAM-CYCLE health of the agent's OWN lattice: how memory is consolidating.
Tier flow (short/mid/long) + how many facts are READY to promote, dwell maturity, decay/fading,
contested facts, abstraction/gist output, and the DIALS in effect (so the numbers are legible:
you see the threshold next to how many cleared it). Prints one JSON line. build_provider opens
the store (running migrations); all queries are read-only."""
import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import build_provider  # noqa: E402


def main():
    try:
        p = build_provider(session_id="dream")
        s = p._store
        c = s._conn
        cfg = p._config
        short_cyc = int(cfg.get("short_tier_cycles", 3))
        mid_cyc = int(cfg.get("mid_tier_cycles", 6))
        prom = float(cfg.get("promotion_resonance_threshold", 4))
        forget_dorm = int(cfg.get("forget_after_dormant_cycles", 100))
        mc, dc = s.get_cycle_counts()

        def q(sql, *a):
            r = c.execute(sql, a).fetchone()
            return r[0] if r and r[0] is not None else 0

        tiers = {}
        for r in c.execute(
                "SELECT tier, COUNT(*) n, ROUND(AVG(resonance_count),2) ar, "
                "ROUND(AVG(COALESCE(cycles_in_tier,0)),2) ad FROM semantic_facts "
                "WHERE superseded_by IS NULL GROUP BY tier"):
            tiers[r["tier"] or "?"] = {"n": r["n"], "avg_res": r["ar"], "avg_dwell": r["ad"]}
        s2m = q("SELECT COUNT(*) FROM semantic_facts WHERE superseded_by IS NULL AND tier='short' "
                "AND COALESCE(cycles_in_tier,0)>=? AND resonance_count>=?", short_cyc, prom)
        m2l = q("SELECT COUNT(*) FROM semantic_facts WHERE superseded_by IS NULL AND tier='mid' "
                "AND COALESCE(cycles_in_tier,0)>=? AND resonance_count>=?", mid_cyc, prom)
        abstract = q("SELECT COUNT(*) FROM semantic_facts WHERE superseded_by IS NULL AND category LIKE '%abstract%'")
        gist = q("SELECT COUNT(*) FROM semantic_facts WHERE superseded_by IS NULL AND category LIKE '%gist%'")
        try:
            links = q("SELECT COUNT(*) FROM abstraction_sources")
        except Exception:
            links = 0
        contested = q("SELECT COUNT(*) FROM semantic_facts WHERE conflict_group_id IS NOT NULL AND superseded_by IS NULL")
        groups = q("SELECT COUNT(DISTINCT conflict_group_id) FROM semantic_facts WHERE conflict_group_id IS NOT NULL AND superseded_by IS NULL")
        superseded = q("SELECT COUNT(*) FROM semantic_facts WHERE superseded_by IS NOT NULL")
        pins = q("SELECT COUNT(*) FROM semantic_facts WHERE pinned=1 AND superseded_by IS NULL")
        faded = q("SELECT COUNT(*) FROM semantic_facts WHERE superseded_by IS NULL AND resonance_count < 2")
        stale_strong = q("SELECT COUNT(*) FROM semantic_facts WHERE superseded_by IS NULL AND resonance_count>=10 "
                         "AND (last_confirmed_cycle IS NULL OR ?-last_confirmed_cycle > ?)", mc, mid_cyc)
        print(json.dumps({
            "ok": True, "memory_cycle": mc, "dream_cycle": dc,
            "tiers": tiers,
            "promotion": {"short_to_mid_ready": s2m, "mid_to_long_ready": m2l,
                          "dials": {"promotion_resonance": prom,
                                    "short_tier_cycles": short_cyc, "mid_tier_cycles": mid_cyc}},
            "decay": {"faded_low_res": faded, "forget_after_dormant_cycles": forget_dorm,
                      "stale_but_strong": stale_strong},
            "conflicts": {"live_groups": groups, "contested_facts": contested,
                          "superseded_history": superseded},
            "abstraction": {"abstract_facts": abstract, "gist_facts": gist, "source_links": links},
            "pins": pins,
        }))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:200]}))


if __name__ == "__main__":
    main()
