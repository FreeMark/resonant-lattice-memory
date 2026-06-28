r"""stress_longhorizon.py - scale + long-horizon recall-quality stress test.

The one piece of evidence that turns "sound mechanics" into a defensible
long-term claim: drive thousands of real (nomic-embed) facts through many dream
cycles and MEASURE whether the right things are still recalled, whether the DB
stays bounded, and whether recall stays fast + non-fabricating as the corpus
grows.

Design
------
* Plant 30 distinctive "golden" needle facts at epoch 0, in three cohorts:
    - PINNED (10)     : pinned -> must be recalled for the ENTIRE run.
    - REINFORCED (10) : re-confirmed every epoch ("use it or lose it") -> should persist.
    - COLD (10)       : added once, never reinforced. NOTE (validated): these are
                        DISTINCTIVE facts, so the novelty boost carries them through
                        the promotion dwell into the decay-exempt LONG tier - the
                        system retains salient one-offs BY DESIGN, so this cohort
                        does NOT fade. Forgetting operates on low-salience facts and
                        shows up in aggregate row pruning, not in this cohort's recall.
* Each epoch: ingest a batch of distinct distractor facts (real embeddings), run a
  full dream cycle (decay / promote / conflict-scan / prune / long-cap; periodic
  abstraction), then MEASURE on the golden set:
    - recall@1 / @5 / @10 and MRR per cohort (query is a PARAPHRASE sharing a rare
      anchor token, so it tests semantic+keyword recall, not exact match),
    - average recall latency,
    - false-confidence: queries for facts NEVER added -> top relevance should stay low,
    - substrate health: tier distribution, DB file size, entity/episode counts.
* Every epoch is CHECKPOINTED to results/stress_metrics.jsonl + results/stress_report.md,
  so a long background run yields incremental, inspectable results.

Config (env): RL_STRESS_FACTS (default 20000), RL_STRESS_EPOCH (default 400),
RL_STRESS_ABSTRACT_EVERY (default 10 epochs). Reason/embed models from _common.

Run (background recommended):
    python tests/stress_longhorizon.py
"""
import json
import os
import random
import sys
import time
import _common as C

FACTS = int(os.environ.get("RL_STRESS_FACTS", "20000"))
EPOCH = int(os.environ.get("RL_STRESS_EPOCH", "400"))
ABSTRACT_EVERY = int(os.environ.get("RL_STRESS_ABSTRACT_EVERY", "10"))
EPOCHS = max(1, FACTS // EPOCH)

rng = random.Random(1729)

# ---- distractor generators (unique tokens -> distinct rows, no merge) ----
FIRST = "Ava Noah Mia Liam Emma Ezra Iris Omar Lena Theo Nadia Cyrus Priya Soren Yusuf Greta Dario Hana Felix Ingrid".split()
LAST = "Reyes Okafor Lindqvist Tanaka Mwangi Costa Devi Halloran Voss Bauer Nakamura Abara Petrov Singh Ferro Yoon".split()
COMP = "Tanager Meridian Halcyon Vantyx Borealis Cindra Quokka Vellum Nimbus Orrery Saffron틈 Kestrel Lumen Pylon Draxis".split()
CITY = "Tromso Reykjavik Lisbon Osaka Nairobi Tallinn Boise Cusco Perth Ghent Almaty Hobart".split()
PLAN = "Starter Growth Enterprise Scale Atlas Sovereign".split()
PROD = "the billing API the data pipeline the auth gateway the search index the export job the webhook relay".split()
DEPT = "finance security platform growth support data".split()

def distractor(i):
    p = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    co = rng.choice(COMP)
    t = rng.choice([
        lambda: (f"Customer {co} (account ACT-{i:06d}) moved to the {rng.choice(PLAN)} plan; MRR {rng.randint(1000,900000)} cents.", [co.lower()]),
        lambda: (f"{p} from {co} filed ticket TK-{i:06d} about {rng.choice(PROD)} in {rng.choice(CITY)}.", [co.lower(), p.split()[0].lower()]),
        lambda: (f"Note N-{i:06d}: reviewed {rng.choice(DEPT)} metrics for {co}; follow up with {p}.", [co.lower()]),
        lambda: (f"Invoice INV-{i:06d}: {co} billed {rng.randint(1000,500000)} cents for {rng.choice(PROD)}.", [co.lower()]),
        lambda: (f"{p} prefers {rng.choice(['async standups','dark mode','weekly digests','no meetings Fridays'])} and works in {rng.choice(DEPT)} at {co}.", [p.split()[0].lower(), co.lower()]),
    ])
    return t()

# ---- golden needles: each anchored by a UNIQUE rare codename so all 30 are
#      distinct rows (no merge) and each is findable by a paraphrased query that
#      shares the codename. Codenames are disjoint from the NEVER (never-stored)
#      queries below.
CODENAMES = ["Tanager", "Meridian", "Halcyon", "Vantyx", "Borealis", "Cindra", "Nimbus",
             "Orrery", "Kestrel", "Zephyrine", "Quokka", "Vellum", "Saffron", "Pylon",
             "Draxis", "Lumen", "Ironwood", "Calyx", "Marrow", "Tindra", "Aster", "Cobalt",
             "Verdant", "Halite", "Onyx", "Cinnabar", "Peregrine", "Quillon", "Sable", "Wren"]
CITY = "Tromso Reykjavik Lisbon Osaka Nairobi Tallinn Boise Cusco Perth Ghent Almaty Hobart".split()
NAME = "Aurelia-Voss Soren-Lindqvist Priya-Devi Omar-Abara Greta-Bauer Yusuf-Okafor Hana-Yoon Dario-Ferro".split()

# (fact template, query template) - {S}=codename, deterministic secondary tokens.
SHAPES = [
    ("Project {S} rotates its encryption keys every {N} days per the security guild.",
     "How often does Project {S} rotate its encryption keys?"),
    ("The {S} board offsite is at the {CITY} annex, hosted by Dr. {NAME}.",
     "Where is the {S} board offsite and who hosts it?"),
    ("{S}'s enterprise renewal closed at {AMT} cents for a {Y}-year term.",
     "What was {S}'s enterprise renewal amount and term?"),
    ("{S} routes all spend approvals through the {CITY} finance desk run by {NAME}.",
     "Where and through whom do {S} spend approvals get routed?"),
    ("The {S} data migration is owned by {NAME}, due before the {CITY} summit.",
     "Who owns the {S} data migration and by what deadline?"),
    ("{S}'s support SLA is a {N}-minute first response, audited monthly by {NAME}.",
     "What is {S}'s support SLA and who audits it?"),
]


def build_goldens():
    goldens = []
    for i, code in enumerate(CODENAMES):           # 30 unique codenames
        ft, qt = SHAPES[i % len(SHAPES)]
        sub = {"S": code, "N": 17 + i, "CITY": CITY[i % len(CITY)],
               "NAME": NAME[i % len(NAME)].replace("-", " "),
               "AMT": (i + 3) * 410000, "Y": 2 + (i % 4)}
        cohort = ["pinned", "reinforced", "cold"][i % 3]   # 10 each, interleaved
        goldens.append({
            "cohort": cohort,
            "fact": ft.format(**sub),
            "query": qt.format(**sub),
            "anchors": [code.lower(), sub["NAME"].split()[0].lower()],
            "id": None,
        })
    return goldens   # 30 distinct

NEVER = [
    "What is the launch date of the Phobos satellite ground station?",
    "How many seats did Wallaby Logistics purchase for the Atlas plan?",
    "What is the refund policy for the Zephyr hardware bundle?",
    "Who approved the Drakefell merger due-diligence budget?",
    "What is the API rate limit for the Selene analytics tier?",
]


def recall_metrics(R, goldens, cohort):
    items = [g for g in goldens if g["cohort"] == cohort and g["id"]]
    if not items:
        return {"n": 0}
    r1 = r5 = r10 = 0
    mrr = 0.0
    lat = 0.0
    for g in items:
        t0 = time.time()
        hits = R.search(g["query"], limit=10)
        lat += time.time() - t0
        ids = [h.get("id") for h in hits]
        if g["id"] in ids:
            pos = ids.index(g["id"])
            mrr += 1.0 / (pos + 1)
            r1 += pos < 1
            r5 += pos < 5
            r10 += pos < 10
    n = len(items)
    return {"n": n, "recall@1": round(r1 / n, 3), "recall@5": round(r5 / n, 3),
            "recall@10": round(r10 / n, 3), "mrr": round(mrr / n, 3),
            "avg_latency_ms": round(1000 * lat / n, 1)}


def false_confidence(R):
    top = []
    for q in NEVER:
        hits = R.search(q, limit=1)
        top.append(round(float(hits[0].get("relevance", 0.0)), 3) if hits else 0.0)
    return {"max_top_relevance": max(top) if top else 0.0, "samples": top}


def main():
    if not C.ollama_up():
        print("Ollama not reachable - stress test needs real embeddings."); return 2
    C.OUT_DIR.mkdir(exist_ok=True)
    metrics_path = C.OUT_DIR / "stress_metrics.jsonl"
    report_path = C.OUT_DIR / "stress_report.md"
    open(metrics_path, "w").close()   # reset

    # realistic long-horizon dynamics: forgetting actually fires within the run
    # realistic tier/decay so promotion + forgetting actually exercise within the
    # run. max_long_facts / forget_after_dormant are applied via the dream-cycle
    # method calls below (enforce_long_tier_cap / prune_weak_facts), not the ctor.
    s, R, _, db = C.make_store("stress.db",
                               initial_resonance=4, decay_per_cycle=0.5,
                               short_tier_cycles=3, mid_tier_cycles=6)
    prompts = C.load("prompts")
    print(f"Stress test: {FACTS} facts over {EPOCHS} epochs x {EPOCH}, abstraction every "
          f"{ABSTRACT_EVERY}. model={C.REASON_MODEL} db={db}")

    # plant goldens at epoch 0
    goldens = build_goldens()
    for g in goldens:
        a, fid = C.add_fact(s, R, g["fact"], category="golden", entities=g["anchors"], session="gold")
        g["id"] = fid if a == "added" else fid   # reinforced returns existing id; goldens are unique anyway
    for g in goldens:
        if g["cohort"] == "pinned" and g["id"]:
            s.set_pinned(g["id"], True)
    planted = sum(1 for g in goldens if g["id"])
    print(f"planted {planted}/30 golden needles (ids unique={len({g['id'] for g in goldens})})")

    run_t0 = time.time()
    fact_i = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        ep_t0 = time.time()
        added = merged = 0
        for _ in range(EPOCH):
            content, ents = distractor(fact_i); fact_i += 1
            a, _fid = C.add_fact(s, R, content, category="biz", entities=ents, session=f"e{epoch}")
            added += (a == "added"); merged += (a != "added")
        ingest_s = time.time() - ep_t0

        # reinforce the REINFORCED cohort (simulates recall-driven use)
        cyc = s._current_memory_cycle()
        for g in goldens:
            if g["cohort"] == "reinforced" and g["id"]:
                s.adjust_resonance(g["id"], 3)
                try:
                    s._conn.execute("UPDATE semantic_facts SET last_confirmed_cycle=? WHERE id=?", (cyc, g["id"]))
                except Exception:
                    pass
        s._conn.commit()

        # dream cycle
        d_t0 = time.time()
        # Advance the LOGICAL memory clock so cycle-driven dormancy/prune grace
        # actually elapses (the provider does this each consolidation epoch).
        # Without it, time-based pruning never fires and row reduction is
        # dedup/merge-only.
        s.set_cycle_counts(memory_cycle=epoch)
        s.apply_cycle_decay()
        s.increment_tier_cycles()
        s.promote_facts()
        try:
            s.resolve_hrr_conflicts()
        except Exception as e:
            print(f"  (conflict scan note: {e})")
        s.prune_weak_facts(forget_after_cycles=10)
        capped = s.enforce_long_tier_cap(1000)
        abstracted = 0
        if epoch % ABSTRACT_EVERY == 0:
            try:
                ab0 = s._conn.execute("SELECT COUNT(*) FROM semantic_facts WHERE category='abstract'").fetchone()[0]
                s.perform_abstraction_pass(C.REASON_MODEL, C.OLLAMA,
                                           prompt=getattr(prompts, "DEFAULT_CONSOLIDATION_PROMPT", None),
                                           min_cluster_size=3, max_clusters=4)
                ab1 = s._conn.execute("SELECT COUNT(*) FROM semantic_facts WHERE category='abstract'").fetchone()[0]
                abstracted = ab1 - ab0
            except Exception as e:
                print(f"  (abstraction note: {e})")
        dream_s = time.time() - d_t0

        health = s.get_memory_health()
        snap = {
            "epoch": epoch, "cycle": s._current_memory_cycle(),
            "facts_ingested_total": fact_i, "epoch_added": added, "epoch_merged": merged,
            "ingest_facts_per_s": round(EPOCH / ingest_s, 1) if ingest_s else None,
            "ingest_s": round(ingest_s, 1), "dream_s": round(dream_s, 1),
            "db_bytes": os.path.getsize(db),
            "total_rows": s._conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0],
            "by_tier": health.get("by_tier"),
            "entities": health.get("total_entities"),
            "conflict_groups": health.get("active_conflict_groups", health.get("conflict_groups")),
            "long_cap_evicted": capped, "abstracted": abstracted,
            "recall_pinned": recall_metrics(R, goldens, "pinned"),
            "recall_reinforced": recall_metrics(R, goldens, "reinforced"),
            "recall_cold": recall_metrics(R, goldens, "cold"),
            "false_confidence": false_confidence(R),
            "elapsed_s": round(time.time() - run_t0, 1),
        }
        history.append(snap)
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snap) + "\n")
        write_report(report_path, history, db)
        print(f"[epoch {epoch}/{EPOCHS}] rows={snap['total_rows']} tiers={snap['by_tier']} "
              f"db={snap['db_bytes']//1024}KB pin@10={snap['recall_pinned'].get('recall@10')} "
              f"cold@10={snap['recall_cold'].get('recall@10')} "
              f"fab={snap['false_confidence']['max_top_relevance']} "
              f"ingest={snap['ingest_facts_per_s']}/s elapsed={snap['elapsed_s']}s")

    s.close()
    print(f"\nDONE in {round(time.time()-run_t0,1)}s. Report: {report_path}")
    return 0


def write_report(path, history, db):
    last = history[-1]
    pin = [h["recall_pinned"].get("recall@10") for h in history]
    cold = [h["recall_cold"].get("recall@10") for h in history]
    rein = [h["recall_reinforced"].get("recall@10") for h in history]
    fab = [h["false_confidence"]["max_top_relevance"] for h in history]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Long-Horizon Scale Stress Test - running report\n\n")
        f.write(f"**Updated**: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Model**: {C.REASON_MODEL} / {C.EMBED_MODEL}  \n")
        f.write(f"**Config**: target {FACTS} facts, {EPOCH}/epoch, abstraction every {ABSTRACT_EVERY}  \n")
        f.write(f"**Progress**: epoch {last['epoch']}/{EPOCHS} - {last['facts_ingested_total']} facts ingested, "
                f"{last['total_rows']} rows live, {round(last['db_bytes']/1e6,1)} MB, elapsed {last['elapsed_s']}s  \n\n")
        f.write("## Headline (latest epoch)\n\n")
        f.write(f"- PINNED recall@10: **{last['recall_pinned'].get('recall@10')}** (must stay 1.0) | "
                f"MRR {last['recall_pinned'].get('mrr')} | latency {last['recall_pinned'].get('avg_latency_ms')}ms\n")
        f.write(f"- REINFORCED recall@10: **{last['recall_reinforced'].get('recall@10')}**\n")
        f.write(f"- COLD (distinctive, unreinforced) recall@10: **{last['recall_cold'].get('recall@10')}** "
                f"(retained by design: novelty->long tier->decay-exempt; forgetting acts on low-salience "
                f"distractors, visible in aggregate pruning below, not here)\n")
        f.write(f"- False-confidence (max relevance for never-stored queries): **{last['false_confidence']['max_top_relevance']}** (lower=better)\n")
        f.write(f"- Tier distribution: {last['by_tier']} | long-cap evictions this epoch: {last['long_cap_evicted']}\n")
        f.write(f"- DB size: {round(last['db_bytes']/1e6,2)} MB for {last['total_rows']} rows\n\n")
        f.write("## Trajectories (recall@10 per epoch)\n\n")
        f.write(f"- pinned:     {pin}\n")
        f.write(f"- reinforced: {rein}\n")
        f.write(f"- cold:       {cold}\n")
        f.write(f"- false-conf: {fab}\n\n")
        f.write("## Per-epoch metrics\n\n")
        f.write("| ep | rows | by_tier | db MB | ingest/s | dream s | pin@10 | rein@10 | cold@10 | fab | conflicts |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for h in history:
            f.write(f"| {h['epoch']} | {h['total_rows']} | {h['by_tier']} | {round(h['db_bytes']/1e6,1)} | "
                    f"{h['ingest_facts_per_s']} | {h['dream_s']} | {h['recall_pinned'].get('recall@10')} | "
                    f"{h['recall_reinforced'].get('recall@10')} | {h['recall_cold'].get('recall@10')} | "
                    f"{h['false_confidence']['max_top_relevance']} | {h['conflict_groups']} |\n")


if __name__ == "__main__":
    sys.exit(main())
