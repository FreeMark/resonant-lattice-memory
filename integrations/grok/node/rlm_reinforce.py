#!/usr/bin/env python3
"""Config-gated resonance bump for a set of fact ids on the SELF lattice.

The consumption half of the per-turn prefetch split: the prefetch worker recalls with
--no-reinforce (precompute is not use), and the rlm_prefetch MCP tool fires this when the agent
actually READS the block, so reinforce-on-recall keeps meaning "this memory was used". Applies
the same MIN(+bump, 50) saturating update as the recall path; a no-op when reinforce_on_recall
is off or recall_bump is 0.
"""
import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import load_config, INSTANCE  # noqa: E402
import sqlite3  # noqa: E402

DB = os.path.join(INSTANCE, "resonant_lattice_memory.db")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="comma-separated fact ids")
    args = ap.parse_args()
    ids = [int(x) for x in args.ids.split(",") if x.strip().lstrip("-").isdigit()]
    try:
        cfg = load_config()
        bump = float(cfg.get("recall_bump", 0) or 0)
        if not ids or not cfg.get("reinforce_on_recall") or bump <= 0:
            print(json.dumps({"ok": True, "reinforced": 0, "note": "gated off or no ids"}))
            return
        conn = sqlite3.connect(DB)
        try:
            ph = ",".join("?" * len(ids))
            cur = conn.execute(
                f"UPDATE semantic_facts SET resonance_count = MIN(resonance_count + ?, 50.0) "
                f"WHERE id IN ({ph})", (bump, *ids))
            conn.commit()
            print(json.dumps({"ok": True, "reinforced": cur.rowcount or 0}))
        finally:
            conn.close()
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:200]}))


if __name__ == "__main__":
    main()
