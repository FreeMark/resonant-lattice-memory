r"""scale_ceiling.py - does recall stay accurate + fast at a LARGE live corpus?

The stress test plateaued ~7-9k LIVE rows because decay/prune kept it bounded.
This probe removes decay/prune entirely and ACCUMULATES up to 50,000 distinct
facts, so we measure vector recall + latency against a genuinely large store:
at each 5k checkpoint, recall@10 of planted golden needles (real search over the
full corpus), average query latency, DB size, and ingest rate.

Config (env): RL_CEIL_FACTS (default 50000), RL_CEIL_STEP (default 5000).
Pure substrate + embeddings (no LLM). Checkpoints to results/scale_ceiling_*.
"""
import json
import os
import sys
import time
import _common as C
from stress_longhorizon import build_goldens   # reuse the 30 distinct goldens

FACTS = int(os.environ.get("RL_CEIL_FACTS", "50000"))
STEP = int(os.environ.get("RL_CEIL_STEP", "5000"))


def recall_at_k(R, goldens, k=10):
    hit = 0
    lat = 0.0
    for g in goldens:
        t0 = time.time()
        hits = R.search(g["query"], limit=k)
        lat += time.time() - t0
        if g["id"] in [h.get("id") for h in hits]:
            hit += 1
    n = len(goldens)
    return round(hit / n, 3), round(1000 * lat / n, 1)


def main():
    if not C.ollama_up():
        print("Ollama not reachable."); return 2
    C.OUT_DIR.mkdir(exist_ok=True)
    jsonl = C.OUT_DIR / "scale_ceiling_metrics.jsonl"
    open(jsonl, "w").close()
    suite = C.Suite("Scale Ceiling (recall + latency at 50k live rows)", model="(no LLM)")

    s, R, _, db = C.make_store("ceiling.db")
    goldens = build_goldens()
    for g in goldens:
        _, fid = C.add_fact(s, R, g["fact"], category="golden", entities=g["anchors"], session="g")
        g["id"] = fid
    suite.report("golden needles planted", len(goldens))

    history = []
    t0 = time.time()
    i = 0
    while i < FACTS:
        b0 = time.time()
        n = min(STEP, FACTS - i)
        C.load_distractors(s, R, n, start=i)        # NO decay/prune: pure accumulation
        i += n
        rate = round(n / (time.time() - b0), 1)
        rows = s._conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
        r10, lat = recall_at_k(R, goldens, 10)
        r1, _ = recall_at_k(R, goldens, 1)
        snap = {"ingested": i, "live_rows": rows, "db_mb": round(os.path.getsize(db) / 1e6, 1),
                "recall@1": r1, "recall@10": r10, "latency_ms": lat,
                "ingest_per_s": rate, "elapsed_s": round(time.time() - t0, 1)}
        history.append(snap)
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(snap) + "\n")
        print(f"[{i}/{FACTS}] rows={rows} db={snap['db_mb']}MB recall@1={r1} recall@10={r10} "
              f"lat={lat}ms ingest={rate}/s elapsed={snap['elapsed_s']}s")

    last = history[-1]
    suite.report("final live rows", last["live_rows"])
    suite.report("final DB size MB", last["db_mb"])
    suite.report("recall@10 trajectory", [h["recall@10"] for h in history])
    suite.report("latency_ms trajectory", [h["latency_ms"] for h in history])
    suite.hard(f"recall@10 stays >=0.95 at {last['live_rows']} live rows",
               last["recall@10"] >= 0.95, f"{last['recall@10']}")
    suite.hard("recall@1 stays >=0.90 at full scale", last["recall@1"] >= 0.90, f"{last['recall@1']}")
    suite.hard("recall@10 did not degrade vs the first checkpoint",
               last["recall@10"] >= history[0]["recall@10"] - 0.05,
               f"{history[0]['recall@10']} -> {last['recall@10']}")
    suite.report("latency growth (first -> last)", f"{history[0]['latency_ms']}ms -> {last['latency_ms']}ms")
    s.close()
    return suite.finish("scale_ceiling_results.md")


if __name__ == "__main__":
    sys.exit(main())
