"""eval_metrics.py — the three Phase-1 harness metrics over per-turn replay results.

Pure functions (no deps) so they're trivially unit-testable and reusable by the preset runner.
A "turn result" is a dict produced by eval_replay for each scored turn:
    {"expected": [key,...], "prefetched": [key,...], "poison": [key,...], "tool_calls": [{...}]}

The metrics map 1:1 to the project's success test (see resonant-lattice north-star plan):
  * right_time_recall  — A1: did the right memory surface UNPROMPTED at the moment it mattered?
  * poison_hit_rate    — A6 guardrail: a stale/contested/irrelevant memory must NOT surface.
  * tool_hallucination_rate — A1 headline (scaffolded now; Phase 3 makes the system DRIVE it down).
"""


def _safe_div(a, b):
    return (a / b) if b else 0.0


def right_time_recall(turn_results):
    """Right-time recall (A1). Over every turn that ASSERTS an expectation, did the expected
    memory appear in that turn's prefetch block? Reports micro recall (pooled over all
    (turn, expected-key) pairs) plus the fraction of expectation-turns FULLY satisfied — the
    cleaner "did the right thing surface at the right moment" number.

    Precision is intentionally not reported here: not every prefetched item is relevance-annotated,
    so a classic precision would be ill-defined. The precision side of the bias (A6 — don't surface
    the wrong thing) is covered by poison_hit_rate over explicitly-forbidden keys."""
    tp = fn = 0
    turns_with_expectation = 0
    turns_fully_satisfied = 0
    for t in turn_results:
        # An expect_top fact must also SURFACE (rank is scored separately by relevance_ordering),
        # so fold both into the recall expectation.
        exp = set(t.get("expected") or []) | set(t.get("expect_top") or [])
        if not exp:
            continue
        turns_with_expectation += 1
        pre = set(t.get("prefetched") or [])
        hit = exp & pre
        tp += len(hit)
        fn += len(exp - pre)
        if hit == exp:
            turns_fully_satisfied += 1
    return {
        "recall": _safe_div(tp, tp + fn),
        "turns_with_expectation": turns_with_expectation,
        "turns_fully_satisfied": turns_fully_satisfied,
        "turn_satisfaction_rate": _safe_div(turns_fully_satisfied, turns_with_expectation),
        "expected_hits": tp,
        "expected_misses": fn,
    }


def poison_hit_rate(turn_results):
    """Guardrail (A6 — a poison hit hurts infinitely more than a miss). How often a memory the
    turn marked ``poison`` (stale / superseded / contested / irrelevant) leaked into the prefetch.
    Reports the per-turn leak rate over turns that declared poison + the raw leaked-item count."""
    poison_turns = turns_with_leak = leaked_items = 0
    for t in turn_results:
        poison = set(t.get("poison") or [])
        if not poison:
            continue
        poison_turns += 1
        leaked = poison & set(t.get("prefetched") or [])
        if leaked:
            turns_with_leak += 1
        leaked_items += len(leaked)
    return {
        "poison_turns": poison_turns,
        "turns_with_leak": turns_with_leak,
        "leak_rate": _safe_div(turns_with_leak, poison_turns),
        "leaked_items": leaked_items,
    }


def relevance_ordering(turn_results):
    """Contextual-relevance ORDERING (the user's 'inherent relevance tier'). Over turns that declare
    ``expect_top`` (the contextually-correct fact(s), most-relevant first), did the system RANK them
    where they belong — not merely surface them? ``top1_accuracy`` = expect_top[0] is the #1
    prefetched item; ``in_topk_rate`` = every expect_top key appears in the prefetch block. This is
    the number the slider page tunes so that, of many valid facts, the situationally-correct one wins
    (3" build → 3-4S, not 6S; racing → props-out, not the props-in default)."""
    turns = top1 = topk = 0
    for t in turn_results:
        et = t.get("expect_top") or []
        if not et:
            continue
        turns += 1
        pre = t.get("prefetched") or []
        if pre and pre[0] == et[0]:
            top1 += 1
        if set(et) <= set(pre):
            topk += 1
    return {
        "turns_with_expect_top": turns,
        "top1": top1, "in_topk": topk,
        "top1_accuracy": _safe_div(top1, turns),
        "in_topk_rate": _safe_div(topk, turns),
    }


def tool_hallucination_rate(turn_results):
    """Tool-use hallucination (A1 headline; Phase 3 drives it down). For now this measures the
    corpus's annotated tool-call correctness — the fraction of tool calls flagged ``correct=False``.
    Until the Phase-3 grounding loop exists this is a BASELINE the system doesn't yet influence;
    once procedural memory grounds tool use, the same number should fall toward zero."""
    total = wrong = 0
    for t in turn_results:
        for tc in (t.get("tool_calls") or []):
            total += 1
            if not tc.get("correct", True):
                wrong += 1
    return {"tool_calls": total, "hallucinated": wrong, "rate": _safe_div(wrong, total)}


def summarize(turn_results):
    """All three metrics for one replay, plus the turn count — the row the preset runner compares."""
    return {
        "turns": len(turn_results),
        "right_time_recall": right_time_recall(turn_results),
        "relevance_ordering": relevance_ordering(turn_results),
        "poison": poison_hit_rate(turn_results),
        "tool": tool_hallucination_rate(turn_results),
    }
