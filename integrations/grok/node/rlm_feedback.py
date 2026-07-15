#!/usr/bin/env python3
"""Hebbian feedback for the grok agent's OWN lattice (the soft usefulness lever).

  --fb helpful    : raise a fact's resonance (a gentle, deliberate 'this was useful' nudge)
  --fb unhelpful  : lower it (a 'this was wrong/stale' signal that fades the fact toward dormancy)

Feedback is RESONANCE-ONLY: it never touches the pin bit, so a pinned fact keeps its
[PRIORITY] protection even after an 'unhelpful' (the ding lowers resonance but pin still
exempts it from decay -- reported as a note). Deltas come from config
(feedback_helpful_delta / feedback_unhelpful_delta); the positive side is clamped to the
same 50.0 ceiling as recall-reinforcement so a hot fact can't run away. Target by --id.
Prints a JSON result line. Writes only to the agent's own lattice.
"""
import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import build_provider  # noqa: E402

RESONANCE_CEILING = 50.0  # matches store.reinforce_on_recall's saturation cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--fb", choices=["helpful", "unhelpful"], required=True)
    ap.add_argument("--session", default="grok-feedback")
    args = ap.parse_args()
    try:
        prov = build_provider(session_id=args.session)
        store = prov._store
        cur = store.get_fact(args.id)
        if not cur or cur.get("superseded_by") is not None:
            print(json.dumps({"ok": False, "error": f"no live fact with id {args.id}"}))
            return
        before = float(cur.get("resonance_count") or 0.0)
        pinned = bool(cur.get("pinned"))
        h = float(prov._config.get("feedback_helpful_delta", 1.0))
        u = float(prov._config.get("feedback_unhelpful_delta", -3.0))
        delta = h if args.fb == "helpful" else u
        # Clamp the positive side so feedback can't push a fact past the recall ceiling;
        # the negative side floors at 0 inside adjust_resonance.
        if delta > 0:
            delta = min(delta, max(0.0, RESONANCE_CEILING - before))
        store.adjust_resonance(args.id, delta)
        after = store.get_fact(args.id)
        new_res = float((after or {}).get("resonance_count") or 0.0)
        out = {"ok": True, "id": args.id, "feedback": args.fb,
               "delta": round(delta, 3), "resonance_before": round(before, 3),
               "resonance_after": round(new_res, 3),
               "content": (cur.get("content") or "")[:80]}
        if pinned:
            out["note"] = ("fact is [PRIORITY] pinned -- resonance adjusted, but the pin "
                           "still protects it from fading")
        print(json.dumps(out, default=str))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:200]}))


if __name__ == "__main__":
    main()
