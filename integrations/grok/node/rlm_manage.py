#!/usr/bin/env python3
"""Prune or unpin a fact in the RLM lattice on behalf of the grok agent.

The inverses of rlm_write:
  --op forget  -> remove the fact entirely (inverse of rlm_remember)
  --op unpin   -> clear a fact's [PRIORITY] pin but keep the fact (inverse of rlm_pin)

Target by --id, or by the fact's content on STDIN. Content is matched EXACTLY against the
projected text (whitespace-normalized, bullet/[PRIORITY] decoration stripped). If the content
is ambiguous or has no exact match, the tool returns candidate ids and does NOT act -- a
destructive op is never performed on a fuzzy guess. Prints a JSON result line.
"""
import sys, os, re, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import build_provider  # noqa: E402


def norm(s):
    """Normalize to the projection's single-line form; drop a leading bullet + [PRIORITY] marker."""
    s = re.sub(r"\s+", " ", (s or "")).strip()
    s = re.sub(r"^-\s+", "", s)
    s = re.sub(r"^\*\*\[PRIORITY\]\*\*\s*", "", s)
    return s.strip()


def candidates(rows, nq, k=5):
    """Rank live facts by word overlap with the query (helps grok retarget by id). No side effects."""
    qtok = set(nq.lower().split())
    if not qtok:
        return []
    scored = []
    for fid, content, pinned in rows:
        ctok = set(norm(content).lower().split())
        if not ctok:
            continue
        overlap = len(qtok & ctok) / len(qtok)
        if overlap > 0:
            scored.append((overlap, fid, content))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [{"id": fid, "content": content[:120]} for _, fid, content in scored[:k]]


def resolve(store, fid, content):
    """Return (target_dict, None) on a unique hit, or (None, err_dict) with candidates/reason."""
    if fid:
        row = store._conn.execute(
            "SELECT id, content, pinned FROM semantic_facts WHERE id=? AND superseded_by IS NULL",
            (fid,)).fetchone()
        if row:
            return {"id": row[0], "content": row[1], "pinned": bool(row[2])}, None
        return None, {"error": f"no live fact with id {fid}"}
    if not content:
        return None, {"error": "need id or content to target a fact"}

    rows = store._conn.execute(
        "SELECT id, content, pinned FROM semantic_facts WHERE superseded_by IS NULL").fetchall()
    nq = norm(content)
    exact = [r for r in rows if norm(r[1]) == nq]
    if len(exact) == 1:
        r = exact[0]
        return {"id": r[0], "content": r[1], "pinned": bool(r[2])}, None
    if len(exact) > 1:
        return None, {"error": "multiple exact matches; retarget by id",
                      "candidates": [{"id": r[0], "content": r[1][:120]} for r in exact]}
    cands = candidates(rows, nq)
    if cands:
        return None, {"error": "no exact match; retarget by id from candidates", "candidates": cands}
    return None, {"error": "no matching fact found"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["forget", "unpin"], required=True)
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--session", default="grok-manage")
    args = ap.parse_args()

    content = ""
    if not args.id:
        content = sys.stdin.buffer.read().decode("utf-8", "replace").strip()

    try:
        prov = build_provider(session_id=args.session)
        store = prov._store
        target, err = resolve(store, args.id, content)
        if err:
            print(json.dumps({"ok": False, **err}))
            return
        fid = target["id"]
        if args.op == "forget":
            store.remove_fact(fid)
            print(json.dumps({"ok": True, "op": "forget", "id": fid,
                              "was_pinned": target["pinned"], "content": target["content"][:80]}))
        else:  # unpin
            if not target["pinned"]:
                print(json.dumps({"ok": True, "op": "unpin", "id": fid, "already": True,
                                  "content": target["content"][:80]}))
                return
            store.set_pinned(fid, False)
            print(json.dumps({"ok": True, "op": "unpin", "id": fid,
                              "content": target["content"][:80]}))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:200]}))


if __name__ == "__main__":
    main()
