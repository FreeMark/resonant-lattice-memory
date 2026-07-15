#!/usr/bin/env python3
"""Plan B Phase 0 diagnostic: measure WHERE LLM triples die on real grok facts.

Read-only (extraction is pure; writes nothing to the lattice). For a sample of real facts it
runs the exact production triple pipeline in stages and reports the drop at each:

  raw granite output  ->  after _resolve_arg normalization/validation  ->  after the
  relation_min_confidence gate

plus deterministic-path counts, entity-grounding rates, and the confidence distribution of the
normalized-but-gated LLM triples (so we can see if simply lowering the gate would rescue them).

Usage: rlm_triple_diag.py [--n 25] [--verbose]
"""
import sys, os, json, argparse, urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import build_provider  # noqa: E402


def raw_llm(store, content, model, endpoint, prompt, timeout=120.0):
    """Replicate _llm_extract_triples' raw call + parse, but return the UNNORMALIZED triples."""
    base = prompt or (
        "Extract explicit (subject, relation, object) triples STATED in the text below. "
        "Only facts literally present - never infer or add world knowledge. relation is a short "
        'snake_case verb phrase. Output ONLY a JSON array of objects with keys "subject", '
        '"relation", "object", or [] if none.')
    final = f"{base}\n\nTEXT:\n{content}\n\nJSON OUTPUT:"
    payload = {"model": model, "prompt": final, "stream": False, "options": {"temperature": 0.1}}
    req = urllib.request.Request(f"{endpoint}/api/generate",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = json.loads(r.read().decode("utf-8")).get("response", "[]")
        cleaned = store._clean_llm_json(raw)
        i, j = cleaned.find("["), cleaned.rfind("]")
        parsed = json.loads(cleaned[i:j + 1] if i != -1 and j != -1 else cleaned)
        return parsed if isinstance(parsed, list) else []
    except Exception as e:
        return {"_error": str(e)[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    prov = build_provider(session_id="triple-diag")
    store = prov._store
    from store_relations import _resolve_arg  # module-level normalizer

    model = prov._relation_model
    endpoint = prov._ollama_endpoint_relation
    prompt = getattr(prov, "_relation_prompt", None)
    gate = float(prov._relation_min_confidence)
    print(f"[diag] model={model} endpoint={endpoint} min_confidence={gate}")

    rows = store._conn.execute(
        "SELECT id, content, category FROM semantic_facts "
        "WHERE superseded_by IS NULL AND length(content) BETWEEN 40 AND 500 "
        "ORDER BY resonance_count DESC LIMIT ?", (args.n,)).fetchall()

    agg = Counter()
    conf_hist = Counter()       # rounded confidence of normalized LLM triples (pre-gate)
    drop_reason = Counter()     # why a raw LLM triple failed normalization
    grounded_raw = Counter()    # subj/obj grounding of raw triples
    per_fact = []

    for row in rows:
        fid, content = row["id"], row["content"]
        entities = store._extract_entities(content)
        ent_set = {e.lower() for e in entities if e}

        det = store.extract_triples(content, entities)                       # deterministic
        raw = raw_llm(store, content, model, endpoint, prompt)               # raw granite
        if isinstance(raw, dict):                                            # error
            agg["llm_errors"] += 1
            per_fact.append((fid, len(det), "ERR", 0, 0)); continue

        # Normalize each raw triple exactly as _llm_extract_triples does.
        norm = []
        for item in raw:
            if not isinstance(item, dict):
                drop_reason["not_a_dict"] += 1; continue
            rel_raw = item.get("relation")
            if not isinstance(rel_raw, str) or not rel_raw.strip():
                drop_reason["bad_relation"] += 1; continue
            subj, sg = _resolve_arg(str(item.get("subject", "")), ent_set, "subj")
            obj, og = _resolve_arg(str(item.get("object", "")), ent_set, "obj")
            if not subj or not obj:
                drop_reason["arg_dropped_by_resolve"] += 1; continue
            if subj == obj:
                drop_reason["subj_eq_obj"] += 1; continue
            conf = 0.7 + (0.1 if sg else 0.0) + (0.1 if og else 0.0)
            if not sg and not og:
                conf *= 0.6
            conf = round(min(conf, 1.0), 3)
            grounded_raw["both" if (sg and og) else "one" if (sg or og) else "none"] += 1
            conf_hist[conf] += 1
            norm.append((subj, rel_raw, obj, conf))

        llm_pass = [t for t in norm if t[3] >= gate]
        det_pass = [t for t in det if t.get("confidence", 0) >= gate]

        agg["facts"] += 1
        agg["raw_llm"] += len(raw)
        agg["norm_llm"] += len(norm)
        agg["llm_pass_gate"] += len(llm_pass)
        agg["det_total"] += len(det)
        agg["det_pass_gate"] += len(det_pass)
        per_fact.append((fid, len(det), len(raw), len(norm), len(llm_pass)))
        if args.verbose:
            print(f"\n#{fid} [{row['category']}] {content[:90]}")
            print(f"   entities={entities}")
            print(f"   raw_llm={len(raw)} norm={len(norm)} pass={len(llm_pass)} det={len(det)}")
            for s, r, o, c in norm:
                print(f"     ({s} | {r} | {o}) conf={c} {'PASS' if c>=gate else 'gated'}")

    print("\n=== AGGREGATE ===")
    for k in ("facts", "raw_llm", "norm_llm", "llm_pass_gate", "det_total", "det_pass_gate",
              "llm_errors"):
        print(f"  {k}: {agg[k]}")
    if agg["raw_llm"]:
        print(f"  survive normalization: {agg['norm_llm']}/{agg['raw_llm']} "
              f"({100*agg['norm_llm']//max(1,agg['raw_llm'])}%)")
    if agg["norm_llm"]:
        print(f"  survive gate: {agg['llm_pass_gate']}/{agg['norm_llm']} "
              f"({100*agg['llm_pass_gate']//max(1,agg['norm_llm'])}%)")
    print(f"  drop reasons (raw->norm): {dict(drop_reason)}")
    print(f"  grounding of normalized raw triples: {dict(grounded_raw)}")
    print(f"  confidence histogram (normalized LLM, pre-gate): "
          f"{dict(sorted(conf_hist.items()))}")
    print(f"  gate = {gate}; how many gated LLM triples are at conf in [0.4,{gate}): "
          f"{sum(v for c,v in conf_hist.items() if 0.4 <= c < gate)}")


if __name__ == "__main__":
    main()
