#!/usr/bin/env python3
"""Pin authority facts into the RLM lattice as [PRIORITY].

Pinned facts are never forgotten and surface with authority presentation. add_or_reinforce_fact
dedups against existing lattice facts (so re-running reinforces rather than duplicates), then
set_pinned marks it. Embeddings via the same embed endpoint the lattice uses.

EDIT the AUTHORITY list below with your own hard rules / locked decisions before running.
"""
import sys, os, json, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import build_provider  # noqa: E402

# --- EDIT: your authority/guardrail facts (content, category). Examples below. ---
AUTHORITY = [
    ("Never modify the production databases or profiles; work only on the designated sandbox.", "constraint"),
    ("Propose source-code changes to the operator; do not deploy unsolicited patches to the host.", "constraint"),
    ("Pin the platform/tool version stated by the operator; do not upgrade it unprompted.", "constraint"),
]


def embed(text, endpoint, model):
    req = urllib.request.Request(
        f"{endpoint}/api/embeddings",
        data=json.dumps({"model": model, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["embedding"]


def main():
    prov = build_provider(session_id="authority-seed")
    store = prov._store
    endpoint, model = prov._ollama_endpoint_embed, prov._embed_model
    n = 0
    for content, cat in AUTHORITY:
        emb = embed(content, endpoint, model)
        action, fid = store.add_or_reinforce_fact(content, emb, cat, "authority-seed")
        if fid and fid > 0:
            store.set_pinned(fid, True)
            n += 1
            print(f"[pin] id={fid:<4} {action:<24} {content[:60]}")
        else:
            print(f"[skip] {action} {content[:60]}")
    print(f"\n[pin] pinned {n}/{len(AUTHORITY)} authority facts")


if __name__ == "__main__":
    main()
