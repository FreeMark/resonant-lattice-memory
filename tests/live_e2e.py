#!/usr/bin/env python
"""live_e2e.py — comprehensive live end-to-end exercise of resonant_lattice.

Builds a REAL LatticeStore + LatticeRetriever (real nomic-embed-text embeddings),
turns ON the behaviour behind every default-OFF roadmap flag, and drives a
realistic multi-feature scenario through real cycles against a live Ollama,
validating at the SQLite substrate. Parameterised by reasoning model so it can be
run per model (granite4.1:8b, gemma4:12b, …) with cold-load warm-up.

Usage:
    python tests/live_e2e.py --model ibm/granite4.1:8b
    python tests/live_e2e.py --model gemma4:12b

Hard assertions cover model-INDEPENDENT invariants (deterministic extraction,
inference never writes, self-model isolation, substrate shape). LLM-dependent
yields (fact extraction, LLM triples, abstraction, narrative prose) are REPORTED,
not asserted, so the run is robust to model variance. Exit code 0 = all hard
invariants held.
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
import urllib.request

PLUGIN = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "resonant_lattice"))
sys.path.insert(0, PLUGIN)

# Windows consoles default to cp1252; the report uses Unicode arrows/bullets.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Running from localhost
OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


def _load(n):
    spec = importlib.util.spec_from_file_location(n, os.path.join(PLUGIN, n + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def generate(model, prompt, timeout=180, num_predict=None):
    opts = {"temperature": 0.2}
    if num_predict:
        opts["num_predict"] = num_predict
    payload = {"model": model, "prompt": prompt, "stream": False, "options": opts}
    req = urllib.request.Request(
        f"{OLLAMA}/api/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()).get("response", "")


def embed_dim():
    req = urllib.request.Request(
        f"{OLLAMA}/api/embeddings",
        data=json.dumps({"model": EMBED_MODEL, "prompt": "probe"}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return len(json.loads(r.read().decode()).get("embedding", []))


# Deterministic backbone facts (the relation graph used for model-independent
# assertions — extraction of these triples does not depend on the LLM).
PLANTED = [
    ("Maya works at Acme Robotics", ["maya", "acme robotics"]),
    ("Acme Robotics is located in Boston", ["acme robotics", "boston"]),
    ("Boston is located in Massachusetts", ["boston", "massachusetts"]),
    ("Maya prefers dark mode interfaces", ["maya"]),
    ("Maya likes minimal user interfaces", ["maya"]),
    ("Maya uses keyboard shortcuts heavily", ["maya"]),
]

# A raw conversation transcript for the LLM fact-extraction (consolidation) path.
TRANSCRIPT = (
    "USER: Hey, I'm Maya. I just started as a robotics engineer at Acme Robotics here in Boston.\n"
    "ASSISTANT: Congratulations Maya! How are you finding the new role?\n"
    "USER: Good! I really prefer working in dark mode and I lean on keyboard shortcuts a lot.\n"
    "ASSISTANT: Noted. Anything else I should remember about your setup?\n"
    "USER: My daughter Lily sometimes sits with me while I work.\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    model = args.model

    print(f"\n{'='*70}\nLIVE E2E — reason model: {model}\n{'='*70}")
    results = {}
    hard_fail = []

    def check(name, cond, detail=""):
        ok = bool(cond)
        results[name] = ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            hard_fail.append(name)

    # ── 0. Warm up the (possibly cold) reasoning model ───────────────────
    t0 = time.time()
    try:
        generate(model, "Reply with the single word: ready", timeout=300, num_predict=8)
        print(f"\n[warmup] {model} responded in {time.time()-t0:.1f}s (cold-load included)")
    except Exception as e:
        print(f"[warmup] FAILED: {e}")
        print("Cannot reach the model — aborting this run.")
        return 2

    store_mod = _load("store")
    config_schema = _load("config_schema")
    prompts = _load("prompts")
    dim = embed_dim()
    print(f"[setup] embedding dim = {dim} ({EMBED_MODEL})")

    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "live_e2e.db")
    # Small dwell so we can promote to 'long' within the scenario; aggressive
    # novelty so one-shot facts stick. Using canonical defaults where applicable.
    defaults = config_schema.DEFAULTS
    s = store_mod.LatticeStore(db_path=db, vector_dim=dim,
                               promotion_threshold=4,  # canonical
                               short_tier_cycles=1,
                               mid_tier_cycles=1,
                               initial_resonance=5)  # override for test
    retr_mod = _load("retrieval")
    R = retr_mod.LatticeRetriever(s, OLLAMA, EMBED_MODEL, min_similarity=0.30)

    def ingest(content, entities, category="general", use_llm_rel=True):
        emb = R._get_embedding(content)
        hrr_vec = None
        try:
            from store_common import hrr, _HRR_AVAILABLE
            if _HRR_AVAILABLE:
                hrr_vec = hrr.encode_fact(content, entities, dim=s.hrr_dim)
        except Exception:
            pass
        action, fid = s.add_or_reinforce_fact(content, emb, category, "live-sess",
                                              hrr_vector=hrr_vec, entities=entities)
        if fid > 0 and action == "added":
            s.extract_and_store_relations(
                fid, content, entities=entities, min_confidence=0.5,
                reason_model=model, ollama_endpoint=OLLAMA, use_llm=use_llm_rel)
        return action, fid

    timings = {}

    # ── 1. LLM fact extraction (consolidation core) ──────────────────────
    print("\n— P-ingest: LLM fact extraction from transcript —")
    t = time.time()
    extracted = []
    try:
        raw = generate(model, f"{prompts.DEFAULT_EXTRACTION_PROMPT}\n\nLOG:\n{TRANSCRIPT}\n\nJSON OUTPUT:")
        cleaned = s._clean_llm_json(raw)
        a, b = cleaned.find("["), cleaned.rfind("]")
        parsed = json.loads(cleaned[a:b+1] if a != -1 and b != -1 else cleaned)
        if isinstance(parsed, dict):
            parsed = parsed.get("facts", [parsed] if "content" in parsed else [])
        for f in parsed if isinstance(parsed, list) else []:
            if isinstance(f, dict) and isinstance(f.get("content"), str):
                ents = s._extract_entities(f["content"])
                ingest(f["content"], ents, f.get("category", "general"))
                extracted.append(f["content"])
    except Exception as e:
        print(f"   (extraction error: {e})")
    timings["extraction"] = time.time() - t
    print(f"   extracted {len(extracted)} facts in {timings['extraction']:.1f}s")
    for c in extracted:
        print(f"     • {c[:80]}")

    # ── 2. Plant the deterministic backbone (+ LLM triple pass) ──────────
    print("\n— P5a: planted facts + relation extraction (deterministic + LLM) —")
    t = time.time()
    for content, ents in PLANTED:
        ingest(content, ents)
    timings["relations_llm"] = time.time() - t
    rels = s.get_relations(limit=200)
    print(f"   fact_relations now holds {len(rels)} triples "
          f"(planted+LLM pass took {timings['relations_llm']:.1f}s)")
    for r in s.get_relations(subject="maya", limit=10):
        print(f"     • ({r['subject']}, {r['relation']}, {r['object']}) conf={r['confidence']}")
    check("P5a deterministic triple (maya, works_at, acme robotics)",
          any(r["relation"] == "works_at" and r["object"] == "acme robotics"
              for r in s.get_relations(subject="maya")))

    # ── 3. Relational recall (P5b) ───────────────────────────────────────
    print("\n— P5b: relational recall —")
    rr = s.relational_recall(subject="maya", relation="works_at")
    print(f"   relational_recall(maya, works_at, ?) → {[(x['object'], x['match']) for x in rr]}")
    check("P5b graph recall returns acme robotics",
          any(x["object"] == "acme robotics" and x["match"] == "graph" for x in rr))
    fq = s.relational_recall(query="where is Acme Robotics located?")
    print(f"   free query 'where is Acme Robotics located?' → "
          f"{[(x['subject'], x['relation'], x['object']) for x in fq]}")

    # ── 4. Transitive inference (P5c) + no-write invariant ───────────────
    print("\n— P5c: bounded transitive inference (+ no-write invariant) —")
    fr0 = s._conn.execute("SELECT COUNT(*) FROM fact_relations").fetchone()[0]
    sf0 = s._conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
    inf = s.infer_relations("maya", max_hops=3)
    for x in inf:
        print(f"   inferred: maya → {x['object']} (hops={x['hops']}, conf={x['confidence']}, "
              f"rel={x['relation']}) via {[e['relation'] for e in x['path']]}")
    check("P5c infers maya → boston", any(x["object"] == "boston" for x in inf))
    check("P5c infers maya → massachusetts (3 hops)",
          any(x["object"] == "massachusetts" and x["hops"] == 3 for x in inf))
    fr1 = s._conn.execute("SELECT COUNT(*) FROM fact_relations").fetchone()[0]
    sf1 = s._conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
    check("P5c inference wrote NOTHING", fr0 == fr1 and sf0 == sf1,
          f"fact_relations {fr0}->{fr1}, semantic_facts {sf0}->{sf1}")

    # ── 5. Self-model (P7) + isolation from ingest ───────────────────────
    print("\n— P7: deliberate self-model (+ ingest isolation) —")
    s.set_self_model("name", "Hermes", current_cycle=s._current_memory_cycle())
    s.set_self_model("relationship_with_user", "long-term collaborator (Maya)",
                     current_cycle=s._current_memory_cycle())
    id_before = s.get_self_model()
    ingest("the assistant is just a memory system and my name is irrelevant", [])
    id_after = s.get_self_model()
    print(f"   self-model: {[(r['key'], r['value']) for r in id_after]}")
    check("P7 self-model untouched by autonomous ingest", id_before == id_after)
    check("P7 curated name intact", (s.get_self_model("name") or {}).get("value") == "Hermes")

    # ── 6. Promote to long + abstraction (P4 reuse) + conflict (P6) ──────
    print("\n— Hebbian cycle: promote → abstraction → conflict scan —")
    for fid_row in s._conn.execute("SELECT id FROM semantic_facts").fetchall():
        s.adjust_resonance(fid_row["id"], 8)          # push resonance up
    for _ in range(3):                                 # advance dwell + promote
        s.increment_tier_cycles()
        s.promote_facts()
    tiers = {r["tier"]: r["cnt"] for r in s._conn.execute(
        "SELECT tier, COUNT(*) cnt FROM semantic_facts GROUP BY tier").fetchall()}
    print(f"   tiers after promotion: {tiers}")
    check("Hebbian promotion produced long-tier facts", tiers.get("long", 0) > 0, str(tiers))

    # ── Heavy flags path: gist_before_prune (P4) + relation_extract_llm (P5) ─
    # Explicitly exercise the default-OFF heavy LLM paths for cost measurement.
    # (live_e2e already forces use_llm=True on relations and calls abstraction/narrative)
    try:
        low_ids = [r[0] for r in s._conn.execute(
            "SELECT id FROM semantic_facts WHERE tier IN ('mid','long') AND category != 'abstract' LIMIT 4").fetchall()]
        for lid in low_ids:
            s.adjust_resonance(lid, -100)  # drive to 0 while preserving historical peak
        print(f"   [heavy] lowered {len(low_ids)} facts to 0 resonance to qualify for gist")
        t_g = time.time()
        gcount = s.consolidate_before_prune(
            model, OLLAMA,
            prompt=getattr(prompts, "DEFAULT_GIST_PROMPT", None),
            gist_floor=0.0, min_peak_resonance=4.0, max_clusters=2
        )
        timings["gist"] = time.time() - t_g
        print(f"   [heavy] consolidate_before_prune (gist_before_prune) → {gcount} gist(s) in {timings['gist']:.1f}s")
    except Exception as e:
        print(f"   (heavy gist path non-fatal: {e})")

    t = time.time()
    long_before = s._conn.execute("SELECT COUNT(*) FROM semantic_facts WHERE category='abstract'").fetchone()[0]
    try:
        s.perform_abstraction_pass(model, OLLAMA, prompt=prompts.DEFAULT_CONSOLIDATION_PROMPT,
                                   min_cluster_size=2, cluster_entity_overlap=0.3,
                                   cluster_hrr_similarity=0.6, max_clusters=3)
    except Exception as e:
        print(f"   (abstraction error: {e})")
    timings["abstraction"] = time.time() - t
    abstractions = [dict(r) for r in s._conn.execute(
        "SELECT content FROM semantic_facts WHERE category='abstract'").fetchall()]
    print(f"   abstraction pass ({timings['abstraction']:.1f}s) created "
          f"{len(abstractions)-long_before} new abstraction(s):")
    for a in abstractions:
        print(f"     • {a['content'][:90]}")

    # conflict scan (best-effort — heuristic may or may not fire on synthetic data)
    s.resolve_hrr_conflicts()
    pend = s.get_pending_conflicts(min_age_cycles=0)
    print(f"   pending conflict groups after scan: {len(pend)}")

    # ── 7. Narrative (P8) ────────────────────────────────────────────────
    print("\n— P8: session narrative —")
    for line in TRANSCRIPT.strip().split("\n"):
        role, content = line.split(": ", 1)
        s.add_episode("live-sess", role.lower(), content)
    t = time.time()
    sid = s.summarize_session(model, OLLAMA, "live-sess",
                              prompt=prompts.DEFAULT_NARRATIVE_PROMPT,
                              started_cycle=0, ended_cycle=1, created_cycle=1,
                              keep=30, min_episodes=2)
    timings["narrative"] = time.time() - t
    narr = s.get_recent_narrative(limit=1)
    print(f"   summarize_session → id={sid} ({timings['narrative']:.1f}s)")
    if narr:
        print(f"   NARRATIVE: {narr[0]['summary']}")
    check("P8 narrative stored with cycle stamps",
          bool(narr) and narr[0]["ended_cycle"] == 1)

    # ── 8. Substrate dump ────────────────────────────────────────────────
    print("\n— Substrate snapshot —")
    for tbl in ("semantic_facts", "fact_relations", "agent_identity",
                "session_summaries", "entities"):
        n = s._conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"   {tbl:18s} {n}")
    health = s.get_memory_health()
    print(f"   memory_health: facts={health['total_facts']} by_tier={health['by_tier']} "
          f"entities={health['total_entities']} conflicts={health['active_conflict_groups']}")
    s.close()

    print(f"\n— LLM timings ({model}) —")
    for k, v in timings.items():
        print(f"   {k:14s} {v:6.1f}s")

    print(f"\n{'='*70}")
    if hard_fail:
        print(f"RESULT [{model}]: RED — hard invariants failed: {hard_fail}")
        return 1
    print(f"RESULT [{model}]: GREEN — all {len(results)} hard invariants held")
    return 0


if __name__ == "__main__":
    sys.exit(main())
