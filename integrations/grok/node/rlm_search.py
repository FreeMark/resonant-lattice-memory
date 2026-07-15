#!/usr/bin/env python3
"""Read-only hybrid search over any RLM lattice DB (self, or an external domain lattice).

Query on STDIN; --db <path> (default: this instance's own lattice), --k N. Opens the DB READ-ONLY
(mode=ro), embeds the query with the configured embed model, and runs the same vector (k-NN) + FTS5
hybrid the RLM recall path uses (cosine relevance + pin bump, resonance tiebreak). Prints a JSON
list of hits. NEVER writes to the target DB.
"""
import sys, os, re, json, struct, argparse, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import load_config, INSTANCE, HE_RLM  # noqa: E402
sys.path.insert(0, os.path.join(HE_RLM, "resonant_lattice"))  # flat intra-package imports
import sqlite_vec  # noqa: E402
from store_common import sqlite3  # RLM's extension-capable sqlite3 binding  # noqa: E402

DEFAULT_DB = os.path.join(INSTANCE, "resonant_lattice_memory.db")
MIN_SIM = 0.30


def embed(text, endpoint, model):
    req = urllib.request.Request(
        f"{endpoint}/api/embeddings",
        data=json.dumps({"model": model, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["embedding"]


def ro_connect(db_path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def search(db_path, query, k, cfg):
    conn = ro_connect(db_path)
    results = {}
    # 1. vector k-NN (drop hits below the recall floor so results are relevant, not just nearest)
    try:
        qv = embed(query, cfg.get("ollama_endpoint_embed", "http://localhost:11434"),
                   cfg.get("embed_model", "nomic-embed-text:latest"))
        vb = struct.pack(f"{len(qv)}f", *qv)
        rows = conn.execute(
            "SELECT f.id, f.content, f.category, f.resonance_count, f.pinned, v.distance "
            "FROM semantic_vec v JOIN semantic_facts f ON f.id = v.id "
            "WHERE v.embedding MATCH ? AND k = ? AND f.tier != 'superseded' ORDER BY v.distance",
            (vb, max(k * 2, k))).fetchall()
        for r in rows:
            if r["distance"] is None or r["distance"] > (1.0 - MIN_SIM):
                continue
            results[r["id"]] = dict(r)
    except Exception as e:
        sys.stderr.write(f"vec: {e}\n")
    # 2. FTS5 keyword pass (hybrid union; catches exact technical names that embed poorly)
    try:
        cq = re.sub(r'["*^()+\[\]{}]', '', query).strip()
        fts_sql = ("SELECT f.id, f.content, f.category, f.resonance_count, f.pinned, fts.rank "
                   "FROM semantic_fts fts JOIN semantic_facts f ON f.id = fts.rowid "
                   "WHERE semantic_fts MATCH ? AND f.tier != 'superseded' ORDER BY fts.rank LIMIT ?")
        rows = []
        if cq:
            rows = conn.execute(fts_sql, (f'"{cq}"', k)).fetchall()
            if not rows:
                toks = [t for t in cq.split() if len(t) > 1]
                if toks:
                    rows = conn.execute(fts_sql, (" AND ".join(f'"{t}"' for t in toks), k)).fetchall()
        for r in rows:
            results.setdefault(r["id"], dict(r))
    except Exception as e:
        sys.stderr.write(f"fts: {e}\n")
    conn.close()
    # 3. rank: cosine (or FTS baseline) + pin bump; resonance tiebreak
    scored = []
    for r in results.values():
        dist = r.get("distance")
        base = (1.0 - float(dist)) if dist is not None else MIN_SIM
        if r.get("pinned"):
            base += 0.10
        r["_score"] = base
        scored.append(r)
    scored.sort(key=lambda h: (h["_score"], float(h.get("resonance_count") or 0)), reverse=True)
    return [{"id": r["id"], "content": r["content"], "category": r["category"],
             "resonance": r["resonance_count"], "pinned": bool(r["pinned"]),
             "score": round(r["_score"], 4)} for r in scored[:k]]


def reinforce(db_path, ids, bump):
    """SELF lattice only: a small resonance bump for recalled facts (saturates at 50), mirroring
    the RLM recall-reinforcement so recalled facts strengthen and can promote. Opens a SEPARATE
    writable connection; external lattices are never passed here and stay strictly read-only."""
    if not ids or bump <= 0:
        return 0
    conn = sqlite3.connect(db_path)
    try:
        ph = ",".join("?" * len(ids))
        cur = conn.execute(
            f"UPDATE semantic_facts SET resonance_count = MIN(resonance_count + ?, 50.0) "
            f"WHERE id IN ({ph})", (bump, *ids))
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--k", type=int, default=8)
    # precompute callers (the prefetch worker) recall WITHOUT reinforcing; the consumption side
    # (rlm_prefetch serving the block) reinforces via rlm_reinforce.py, so resonance tracks USE.
    ap.add_argument("--no-reinforce", action="store_true")
    args = ap.parse_args()
    query = sys.stdin.buffer.read().decode("utf-8", "replace").strip()
    if not query:
        print(json.dumps({"ok": False, "error": "empty query"}))
        return
    if not os.path.exists(args.db):
        print(json.dumps({"ok": False, "error": "no such lattice"}))
        return
    try:
        cfg = load_config()
        hits = search(args.db, query, args.k, cfg)
        # reinforce-on-read: SELF lattice only (never external), gated by config
        reinforced = 0
        is_self = os.path.abspath(args.db) == os.path.abspath(DEFAULT_DB)
        if is_self and hits and not args.no_reinforce \
                and cfg.get("reinforce_on_recall") and float(cfg.get("recall_bump", 0)) > 0:
            reinforced = reinforce(args.db, [h["id"] for h in hits], float(cfg.get("recall_bump")))
        print(json.dumps({"ok": True, "count": len(hits), "reinforced": reinforced, "hits": hits}))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:200]}))


if __name__ == "__main__":
    main()
