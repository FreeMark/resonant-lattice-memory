#!/usr/bin/env python3
"""Read or CAREFULLY update the grok agent's self-model (identity), for its OWN lattice.

  --op get [--key KEY]  : read one entry (KEY) or the whole self-model (read-only)
  --op set --key KEY    : upsert one entry; the VALUE is read from STDIN (utf-8)

The self-model is a curated, authoritative record of who the agent is -- surfaced
deterministically in the projection, not reconstructed from fuzzy recall. To keep identity
core from mutating on mood, SET is allowlist-gated: only keys in config
self_model_writable_keys are writable (seed keys like 'name' can be left off the list so they
stay locked). An off-list key is rejected with the allowed set, and nothing is written.
Prints a JSON result line.
"""
import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import build_provider  # noqa: E402

_DEFAULT_WRITABLE = ["role", "relationship_with_user", "mandate",
                     "current_focus", "values", "communication_style"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["get", "set"], required=True)
    ap.add_argument("--key", default=None)
    ap.add_argument("--session", default="grok-self-model")
    args = ap.parse_args()
    try:
        prov = build_provider(session_id=args.session)
        store = prov._store
        if args.op == "get":
            model = store.get_self_model(args.key.strip().lower() if args.key else None)
            print(json.dumps({"ok": True, "op": "get",
                              "key": args.key, "model": model}, default=str))
            return
        # set
        key = (args.key or "").strip().lower()
        if not key:
            print(json.dumps({"ok": False, "error": "set requires --key"}))
            return
        allowed = [str(k).strip().lower()
                   for k in (prov._config.get("self_model_writable_keys") or _DEFAULT_WRITABLE)]
        if key not in allowed:
            print(json.dumps({"ok": False,
                              "error": f"key '{key}' is not writable (identity core is locked)",
                              "allowed": allowed}))
            return
        value = sys.stdin.buffer.read().decode("utf-8", "replace").strip()
        if not value:
            print(json.dumps({"ok": False, "error": "empty value on stdin"}))
            return
        row = store.set_self_model(key, value, current_cycle=prov._memory_cycle)
        if not row:
            print(json.dumps({"ok": False, "error": "invalid key/value for set_self_model"}))
            return
        print(json.dumps({"ok": True, "op": "set", "key": row["key"],
                          "value": row["value"][:120], "updated_cycle": row["updated_cycle"]},
                         default=str))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:200]}))


if __name__ == "__main__":
    main()
