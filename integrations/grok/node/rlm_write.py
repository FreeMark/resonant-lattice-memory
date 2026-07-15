#!/usr/bin/env python3
"""Write a single fact into the RLM lattice on behalf of the grok agent (durable memory write).

Called by the rlm-memory MCP server. Content comes on STDIN (safe for arbitrary text); category
and --pin are args. Uses the same add_or_reinforce_fact path the ingest uses, so the write is
embedded, deduped/reinforced against existing facts, and provenance-tagged as a grok direct write
(source_ref grok:direct-write) so recall can tell it apart from transcript-grounded facts.
Prints a JSON result line.
"""
import sys, os, re, json, argparse, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import build_provider  # noqa: E402


def norm_cat(category):
    """Open-ended label -> safe token. Categories are samples, not a closed enum."""
    return re.sub(r"[^a-z0-9_]", "", re.sub(r"[\s-]+", "_", (category or "").strip().lower())) or "concept"


def embed(text, endpoint, model):
    req = urllib.request.Request(
        f"{endpoint}/api/embeddings",
        data=json.dumps({"model": model, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["embedding"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="concept")
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--session", default="grok-direct-write")
    args = ap.parse_args()

    # read raw bytes and decode UTF-8 explicitly, so arbitrary Unicode survives regardless of
    # the SSH session locale (a C/POSIX locale would otherwise make sys.stdin.read() choke).
    content = sys.stdin.buffer.read().decode("utf-8", "replace").strip()
    if not content:
        print(json.dumps({"ok": False, "error": "empty content"}))
        return
    cat = norm_cat(args.category)

    try:
        prov = build_provider(session_id=args.session)
        store = prov._store
        emb = embed(content, prov._ollama_endpoint_embed, prov._embed_model)
        action, fid = store.add_or_reinforce_fact(
            content, emb, cat, source_session=args.session, source_ref="grok:direct-write")
        pinned = False
        if args.pin and fid and fid > 0:
            store.set_pinned(fid, True)
            pinned = True
        print(json.dumps({"ok": True, "id": fid, "action": action,
                          "pinned": pinned, "category": cat}))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:200]}))


if __name__ == "__main__":
    main()
