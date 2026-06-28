"""eval_corpus.py — the replayable session corpus: schema, loader, validator, + a small example.

A corpus is a list of SESSIONS; each session is a list of TURNS. A turn is a dict:

  user:          (str)  the user prompt this turn — drives prefetch + right-time-recall scoring.
  facts:         [{"key","content","category"?,"entities"?}]  facts this turn introduces or
                 reinforces. Injected DIRECTLY (deterministic; no LLM extraction needed to tune the
                 dynamics). ``key`` is a stable handle the author assigns so expectations reference
                 facts by identity, independent of the fact ids the store assigns at insert.
  expect_recall: [key]   facts that SHOULD appear in this turn's prefetch (right-time-recall GT).
  poison:        [key]   facts that must NOT appear (stale/contested/irrelevant) — the A6 guardrail.
  tool_calls:    [{"name","args"?,"correct":bool}]  for the tool-hallucination metric (Phase 3).
  dream:         (bool)  force a dream cycle after this turn (else cadence-driven).   [optional]

Authoring rule: give distinct facts distinct content — the store dedups by semantic similarity, so
two keys with near-identical content would collapse to one fact and confuse the key↔id mapping.
"""

import json

_TURN_KEYS = {"user", "facts", "expect_recall", "expect_top", "poison", "tool_calls", "dream", "note"}
_FACT_KEYS = {"key", "content", "category", "entities"}


def validate_corpus(corpus):
    """Raise ValueError on a malformed corpus; return the corpus unchanged on success. Checks shape,
    required fields, and that every expect_recall / poison key is INTRODUCED by some fact in the
    corpus (a typo'd expectation would otherwise silently score as a permanent miss)."""
    if not isinstance(corpus, list):
        raise ValueError("corpus must be a list of sessions")
    introduced = set()
    for s_i, session in enumerate(corpus):
        if not isinstance(session, list):
            raise ValueError(f"session {s_i} must be a list of turns")
        for t_i, turn in enumerate(session):
            where = f"session {s_i} turn {t_i}"
            if not isinstance(turn, dict):
                raise ValueError(f"{where}: turn must be a dict")
            extra = set(turn) - _TURN_KEYS
            if extra:
                raise ValueError(f"{where}: unknown turn keys {sorted(extra)}")
            if not isinstance(turn.get("user", ""), str):
                raise ValueError(f"{where}: 'user' must be a string")
            for f in (turn.get("facts") or []):
                if not isinstance(f, dict) or not f.get("key") or not f.get("content"):
                    raise ValueError(f"{where}: each fact needs non-empty 'key' and 'content'")
                if set(f) - _FACT_KEYS:
                    raise ValueError(f"{where}: unknown fact keys {sorted(set(f) - _FACT_KEYS)}")
                introduced.add(f["key"])
    # second pass: every referenced key must be introduced somewhere
    for s_i, session in enumerate(corpus):
        for t_i, turn in enumerate(session):
            for ref in ((turn.get("expect_recall") or []) + (turn.get("expect_top") or [])
                        + (turn.get("poison") or [])):
                if ref not in introduced:
                    raise ValueError(
                        f"session {s_i} turn {t_i}: references key '{ref}' that no fact introduces")
    return corpus


def load_corpus(path):
    """Load a corpus from JSON (a single list) or JSONL (one session per line). Validated."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read().strip()
    if not text:
        return []
    if text[0] == "[":
        corpus = json.loads(text)
    else:  # JSONL: each line is a session (list of turns)
        corpus = [json.loads(line) for line in text.splitlines() if line.strip()]
    return validate_corpus(corpus)


def save_corpus(path, corpus):
    validate_corpus(corpus)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(corpus, fh, indent=2)


def all_texts(corpus):
    """Every text the replay will embed (fact contents + user prompts) — for cache pre-warming."""
    texts = []
    for session in corpus:
        for turn in session:
            if turn.get("user"):
                texts.append(turn["user"])
            for f in (turn.get("facts") or []):
                texts.append(f["content"])
    return texts


def example_corpus():
    """A tiny built-in corpus exercising recall-over-time (the A1 'three weeks later' shape):
    session 1 lays down facts; later sessions — after intervening decay/consolidation cycles — must
    still surface the right one UNPROMPTED on a semantically related prompt."""
    return [
        # ── Session 1: lay down durable facts ────────────────────────────────────
        [
            {"user": "hey, a few things about me",
             "facts": [
                 {"key": "maya", "content": "the user's daughter is named Maya and she turns 7 in June",
                  "entities": ["Maya"]},
                 {"key": "props_in", "content": "the user sets up racing quadcopters with props rotating inward by default",
                  "entities": ["quadcopter"]},
                 {"key": "db_port", "content": "the production database runs on port 5432",
                  "entities": ["database"]},
             ]},
        ],
        # ── Session 2: reinforce one, introduce another ──────────────────────────
        [
            {"user": "remind me what port the prod db is on",
             "facts": [], "expect_recall": ["db_port"]},
            {"user": "i also fly a 5-inch freestyle quad now",
             "facts": [{"key": "freestyle", "content": "the user flies a 5-inch freestyle quadcopter",
                        "entities": ["quadcopter"]}]},
        ],
        # ── Session 3: later — the right memory should surface unprompted ─────────
        [
            {"user": "what should I get my kid for her birthday?",
             "facts": [], "expect_recall": ["maya"]},
            {"user": "help me set up a new racing quad",
             "facts": [], "expect_recall": ["props_in"]},
        ],
    ]
