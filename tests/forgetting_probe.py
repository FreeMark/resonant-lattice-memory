r"""forgetting_probe.py - chart the FADE CURVE (selective forgetting).

The 20k stress test could not show forgetting per-fact because its goldens were
DISTINCTIVE: under the default regime (init_resonance == promotion_threshold +
novelty boost) a distinctive fact passively promotes into the decay-EXEMPT long
tier and is retained forever. That is the salience-retention feature, not a bug.

To chart forgetting we use the system's documented "recall-required" regime
(initial_resonance < promotion_threshold, novelty OFF): a fact must be USED to
survive. We then watch four cohorts over many cycles:

  * COLD       - never reinforced  -> should decay, go dormant, be PRUNED (fade).
  * REINFORCED - used every cycle  -> climbs, promotes to long, persists.
  * PINNED     - protected         -> never decays, always recalled.
  * REVIVED    - cold, then reinforced from cycle REVIVE_AT -> demonstrates the
                 "buried but pluckable" comeback (dormant -> revived).

Two signals are charted per cohort per cycle, because they differ:
  * mean RESONANCE  -> the gradual strength curve,
  * normal recall@10 + present-in-DB -> the FORGETTING event. Recall is
    relevance-driven, so a weakening fact stays fully recallable UNTIL it is
    pruned: forgetting is a cliff at the prune cycle, not a slow recall decline.

A compact CONTRAST also shows the SAME distinctive cold fact RETAINED under the
default regime but FADED under recall-required - i.e. retention is regime/
salience dependent, which is exactly why the stress-test cold cohort held at 1.0.

Pure substrate + embeddings (no LLM); runs in ~1-2 min. Result + per-cycle table
in results/forgetting_report.md, metrics in results/forgetting_metrics.jsonl.
"""
import json
import os
import sys
import _common as C

CYCLES = int(os.environ.get("RL_FADE_CYCLES", "32"))
REVIVE_AT = int(os.environ.get("RL_FADE_REVIVE_AT", "20"))   # inside the buried-but-pluckable window
PER_COHORT = 6
FORGET_AFTER = 8          # dormant grace before deep-delete (widens the pluckable window)
BG_PER_CYCLE = 10         # light background churn so recall has competition


def reinforce(s, fid, cyc, bump=1.6):
    """Simulate a recall-driven reinforcement (what reinforce_on_recall does)."""
    s.adjust_resonance(fid, bump)
    try:
        s._conn.execute("UPDATE semantic_facts SET last_confirmed_cycle=? WHERE id=?", (cyc, fid))
    except Exception:
        pass


def plant_cohort(s, R, tag, n):
    out = []
    for k in range(n):
        code = f"{tag}-{k}"
        fact = f"Fade-probe record {code}: a distinctive note about subject {code} for tracking."
        query = f"What is fade-probe record {code} about?"
        _, fid = C.add_fact(s, R, fact, category="probe", entities=[code.lower()], session="fade")
        out.append({"code": code, "id": fid, "query": query})
    return out


def cohort_stats(s, R, cohort):
    present = 0
    res_sum = 0.0
    recalled = 0
    for g in cohort:
        r = s.get_fact(g["id"])
        if r:
            present += 1
            res_sum += float(r.get("resonance_count") or 0.0)
        hits = R.search(g["query"], limit=10)
        if g["id"] in [h.get("id") for h in hits]:
            recalled += 1
    n = len(cohort)
    return {"present": present, "present_frac": round(present / n, 3),
            "mean_res": round(res_sum / present, 2) if present else 0.0,
            "recall@10": round(recalled / n, 3)}


def main():
    if not C.ollama_up():
        print("Ollama not reachable (embeddings needed)."); return 2
    suite = C.Suite("Forgetting / Fade-Curve Probe", model="(recall-required regime; no LLM)")
    metrics_path = C.OUT_DIR / "forgetting_metrics.jsonl"
    C.OUT_DIR.mkdir(exist_ok=True)
    open(metrics_path, "w").close()

    # recall-required regime: a fact must be USED to be retained
    s, R, _, db = C.make_store("fade.db", initial_resonance=3, promotion_threshold=4,
                               decay_per_cycle=0.5, novelty_enabled=False,
                               short_tier_cycles=2, mid_tier_cycles=3)

    cold = plant_cohort(s, R, "COLD", PER_COHORT)
    rein = plant_cohort(s, R, "REIN", PER_COHORT)
    pin = plant_cohort(s, R, "PIN", PER_COHORT)
    revv = plant_cohort(s, R, "REVV", PER_COHORT)
    for g in pin:
        s.set_pinned(g["id"], True)

    bg_i = 0
    history = []
    cold_recall_drop = None      # first cycle COLD recall@10 < 1.0
    cold_pruned_cycle = None     # first cycle COLD fully pruned
    for c in range(1, CYCLES + 1):
        # Advance the LOGICAL memory clock (the provider does this every
        # consolidation epoch; the store's cycle-driven dormancy/prune grace is
        # measured against it). Without this, (cur - dormant_since) never grows
        # and time-based pruning never fires.
        s.set_cycle_counts(memory_cycle=c)
        # light background churn (gives prune something to do; competes in recall)
        for _ in range(BG_PER_CYCLE):
            C.add_fact(s, R, f"Routine ops log entry {bg_i}: standard background activity, nothing notable.",
                       category="bg", entities=[], session="bg"); bg_i += 1
        s.apply_cycle_decay()
        cyc = s._current_memory_cycle()
        for g in rein:
            reinforce(s, g["id"], cyc)
        if c >= REVIVE_AT:
            for g in revv:
                reinforce(s, g["id"], cyc)
        s._conn.commit()
        s.increment_tier_cycles()
        s.promote_facts()
        s.prune_weak_facts(forget_after_cycles=FORGET_AFTER)

        snap = {"cycle": c,
                "cold": cohort_stats(s, R, cold),
                "reinforced": cohort_stats(s, R, rein),
                "pinned": cohort_stats(s, R, pin),
                "revived": cohort_stats(s, R, revv)}
        history.append(snap)
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snap) + "\n")
        if cold_recall_drop is None and snap["cold"]["recall@10"] < 1.0:
            cold_recall_drop = c
        if cold_pruned_cycle is None and snap["cold"]["present"] == 0:
            cold_pruned_cycle = c

    # ---- assertions: the fade story ----
    final = history[-1]
    suite.report("cold first dropped from recall at cycle", cold_recall_drop)
    suite.report("cold fully pruned at cycle", cold_pruned_cycle)
    suite.report("cold final", final["cold"])
    suite.report("reinforced final", final["reinforced"])
    suite.report("pinned final", final["pinned"])
    suite.report("revived final", final["revived"])

    cold_res = [h["cold"]["mean_res"] for h in history if h["cold"]["present"]]
    monotone = all(cold_res[i] >= cold_res[i + 1] - 1e-9 for i in range(len(cold_res) - 1))
    suite.hard("COLD resonance declines monotonically while present", monotone)
    suite.hard("COLD is eventually forgotten (pruned + dropped from recall)",
               final["cold"]["present"] == 0 and final["cold"]["recall@10"] == 0.0, final["cold"])
    suite.hard("PINNED never fades (present + recall@10==1.0 every cycle)",
               all(h["pinned"]["present_frac"] == 1.0 and h["pinned"]["recall@10"] == 1.0 for h in history))
    suite.hard("REINFORCED persists (final recall@10==1.0, all present)",
               final["reinforced"]["recall@10"] == 1.0 and final["reinforced"]["present_frac"] == 1.0,
               final["reinforced"])
    # revival: was fading before REVIVE_AT, recovered by the end
    pre = history[REVIVE_AT - 2]["revived"]
    suite.hard("REVIVED recovers after reinforcement resumes (buried-but-pluckable)",
               final["revived"]["recall@10"] == 1.0 and final["revived"]["mean_res"] > pre["mean_res"],
               f"pre_res={pre['mean_res']} final_res={final['revived']['mean_res']}")

    # ---- contrast: same distinctive cold fact, default regime vs recall-required ----
    sd, Rd, _, _ = C.make_store("fade_default.db")  # default: init=promo, novelty ON
    _, did = C.add_fact(sd, Rd, "Fade-probe record DEF-X: a distinctive note about subject DEF-X.",
                        entities=["def-x"], session="d")
    for cc in range(1, CYCLES + 1):
        sd.set_cycle_counts(memory_cycle=cc)
        for _ in range(BG_PER_CYCLE):
            C.add_fact(sd, Rd, f"Routine ops log {bg_i}: background.", entities=[], session="bg"); bg_i += 1
        sd.apply_cycle_decay(); sd.increment_tier_cycles(); sd.promote_facts()
        sd.prune_weak_facts(forget_after_cycles=FORGET_AFTER)
    drow = sd.get_fact(did)
    suite.report("contrast: distinctive fact under DEFAULT regime",
                 "PRUNED" if not drow else f"res={drow['resonance_count']:.2f} tier={drow['tier']}")
    suite.hard("DEFAULT-regime distinctive fact is RETAINED (novelty->long->decay-exempt)",
               drow is not None and drow["tier"] == "long")
    sd.close()

    s.close()

    # ---- fade-curve report ----
    body = "Per-cycle mean resonance (present facts) | recall@10:\n\n"
    body += "| cyc | COLD res / r@10 | REIN res / r@10 | PIN res / r@10 | REVV res / r@10 |\n|---|---|---|---|---|\n"
    for h in history:
        body += (f"| {h['cycle']} | {h['cold']['mean_res']} / {h['cold']['recall@10']} "
                 f"| {h['reinforced']['mean_res']} / {h['reinforced']['recall@10']} "
                 f"| {h['pinned']['mean_res']} / {h['pinned']['recall@10']} "
                 f"| {h['revived']['mean_res']} / {h['revived']['recall@10']} |\n")
    headline = (f"- COLD dropped from recall at cycle **{cold_recall_drop}**, fully pruned at "
                f"**{cold_pruned_cycle}** (gradual resonance decay, then a recall cliff at prune).\n"
                f"- REINFORCED + PINNED held recall@10 = 1.0 throughout.\n"
                f"- REVIVED faded then recovered once reinforcement resumed at cycle {REVIVE_AT} "
                f"(buried-but-pluckable).\n"
                f"- Contrast: the SAME kind of distinctive fact is RETAINED under the default regime "
                f"(novelty->long tier->decay-exempt) - retention is regime/salience dependent.\n")
    return suite.finish("forgetting_report.md",
                        extra_sections={"Fade milestones": headline, "Fade curve (per cycle)": body})


if __name__ == "__main__":
    sys.exit(main())
