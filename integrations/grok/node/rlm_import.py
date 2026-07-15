#!/usr/bin/env python3
"""Import selected facts from an external lattice into THIS instance's lattice (transfer_knowledge).

Reads fact ids (--ids "1,2,3") from --lattice-db (opened READ-ONLY), re-embeds each with this
instance's embed model, and adds them to this instance's OWN lattice via add_or_reinforce_fact
(so it dedups/reinforces against what is already known). Each import is provenance-tagged
source_ref="import:<lattice>:<id>" so borrowed knowledge stays labeled as borrowed. Never writes
to the source lattice. Prints a JSON result.
"""
import sys, os, re, json, argparse, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import build_provider, HE_RLM  # noqa: E402
sys.path.insert(0, os.path.join(HE_RLM, "resonant_lattice"))
from store_common import sqlite3  # RLM's sqlite3 binding  # noqa: E402


def embed(text, endpoint, model):
    req = urllib.request.Request(
        f"{endpoint}/api/embeddings",
        data=json.dumps({"model": model, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["embedding"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lattice-db", required=True)
    ap.add_argument("--lattice-name", default="external")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--session", default="knowledge-import")
    ap.add_argument("--home", default=None)  # instance to import INTO (default: this instance)
    args = ap.parse_args()

    ids = [int(x) for x in re.split(r"[,\s]+", args.ids.strip()) if x.strip().isdigit()]
    if not ids:
        print(json.dumps({"ok": False, "error": "no valid fact ids"}))
        return
    if not os.path.exists(args.lattice_db):
        print(json.dumps({"ok": False, "error": "no such source lattice"}))
        return

    try:
        prov = build_provider(session_id=args.session, hermes_home=args.home)
        store = prov._store
        src = sqlite3.connect(f"file:{args.lattice_db}?mode=ro", uri=True)
        src.row_factory = sqlite3.Row
        imported = []
        for fid in ids:
            row = src.execute(
                "SELECT content, category FROM semantic_facts WHERE id=? AND tier != 'superseded'",
                (fid,)).fetchone()
            if not row:
                imported.append({"src_id": fid, "status": "not_found"})
                continue
            emb = embed(row["content"], prov._ollama_endpoint_embed, prov._embed_model)
            action, newid = store.add_or_reinforce_fact(
                row["content"], emb, row["category"] or "concept",
                source_session=args.session, source_ref=f"import:{args.lattice_name}:{fid}")
            imported.append({"src_id": fid, "status": action, "new_id": newid,
                             "content": row["content"][:70]})
        src.close()
        print(json.dumps({"ok": True, "count": len(imported), "imported": imported}))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:200]}))


if __name__ == "__main__":
    main()
