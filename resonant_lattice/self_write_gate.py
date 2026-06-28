"""
self_write_gate.py — Phase E self-write policy boundary.

Explicit, auditable denylist of phrases that unambiguously mark the AGENT
describing its OWN configuration / infrastructure / identity (model name,
context size, IP/endpoint, training, "as an AI…"). When such chatter is
autonomously consolidated it lands in semantic_facts as a fake "user fact",
polluting the user model and inflating self-referential resonance.

This is a POLICY, not a heuristic to be clever about — explicit lists, simple
substring matching, no fuzzy scoring. It is intentionally CONSERVATIVE: it
should UNDER-block (miss a novel paraphrase) rather than ever drop a legitimate
fact about the USER's own infrastructure or about the user's project. NOTE:
this user develops a memory system, so facts like "the user refactored the
consolidation epoch" or "the model training data was collected in 2024" are
REAL facts and must NOT match. Bare topic words ("consolidation", "resonance",
"HRR", "ollama", a model name) are therefore deliberately absent below.

Two-tier rule (see is_self_referential_infra):
  1) any _SELF_INFRA_PHRASES substring  → block (unambiguous on its own), OR
  2) an AI-self _SELF_SUBJECTS term AND an _INFRA_TERMS descriptor co-occur.
Neither subject nor infra term blocks alone — only their co-occurrence does,
which is what makes "I am running granite4.1:8b with a 128k context window"
self-infra while "the user runs Ollama on port 11434" stays a user fact.

Pure (str only); the provider's _is_self_referential_infra method delegates here.
"""

_SELF_INFRA_PHRASES = (
    # First-person AI identity — the agent describing what it *is*.
    "as an ai", "as a language model", "as an ai language model",
    "i am an ai", "i'm an ai", "i am a language model", "i'm a language model",
    "i am an assistant", "i am the assistant", "i am claude",
    "large language model",
    # First-person "my <own machinery>" — unambiguous self-config.
    "my context window", "my context length", "my context size",
    "my maximum context", "my token limit", "my knowledge cutoff",
    "my training data", "my training cutoff", "my system prompt",
    "my ip address", "my embedding model", "my reasoning model",
    "my underlying model", "my model name", "i was trained",
)
# AI-self SUBJECTS — kept tight to the agent ITSELF, never a model-in-general
# ("the language model"/"the ai model" are excluded: an AI developer discusses
# those generically). Only fire in conjunction with an _INFRA_TERMS descriptor.
_SELF_SUBJECTS = (
    "i am running", "i'm running", "i am powered", "i'm powered",
    "the assistant", "the ai assistant", "this assistant",
)
# Runtime/config DESCRIPTORS. Deliberately NOT bare nouns like "ollama" or a
# model name (those appear in legitimate user/project facts); these describe a
# runtime's own characteristics, so paired with an AI-self subject they mark
# self-infra chatter.
_INFRA_TERMS = (
    "context window", "context length", "context size", "token limit",
    "knowledge cutoff", "training data", "training cutoff", "system prompt",
    "ip address", "powered by", "running on", "model name",
    "embedding model", "reasoning model", "underlying model",
)


def is_self_referential_infra(content: str) -> bool:
    """True if `content` describes the AGENT'S OWN config/infra/identity.

    Conservative two-tier rule over explicit, auditable lists:
      1) any _SELF_INFRA_PHRASES substring is unambiguous on its own, OR
      2) an AI-self _SELF_SUBJECTS term co-occurs with an _INFRA_TERMS
         descriptor (neither fires alone).
    UNDER-blocks on purpose — it will miss a novel paraphrase rather than risk
    dropping a legitimate fact about the USER's infrastructure or project.
    """
    if not content:
        return False
    c = content.lower()
    if any(phrase in c for phrase in _SELF_INFRA_PHRASES):
        return True
    if any(s in c for s in _SELF_SUBJECTS) and any(t in c for t in _INFRA_TERMS):
        return True
    return False


# ---------------------------------------------------------------------------
# Task/process directives mined from the GOAL PROMPT.
#
# The autonomous goal loop re-injects the full goal text ("[Continuing toward
# your standing goal] Goal: …") as a USER turn every cycle; sync_turn logs it
# verbatim, so the consolidation extractor sees the task INSTRUCTIONS repeated
# in every episode window and mines them as durable "user preferences" (e.g.
# "end with 'Synthesis complete. Stopping.'", "record only … principles"). The
# repetition inflates their apparent importance, and the mid- + session-end
# passes store near-duplicates. These are response FORMAT / task PROCEDURE, not
# external domain knowledge — drop them before they reach the store.
#
# Same discipline as above: explicit, auditable substrings; UNDER-block. Bare
# topic words (memory, synthesis, search, lattice_store, consolidation) are
# ABSENT on purpose — this user builds a memory system, so "the lattice_store
# tool has a search action" is a REAL fact and must NOT match. Only the
# instruction-shaped phrasings below match.
_TASK_META_PHRASES = (
    # completion sentinels the goal templates mandate
    "synthesis complete. stopping", "quiz complete. stopping",
    "goal complete — covered", "goal complete - covered",
    # output-format directives
    "end your final message", "on its own line, with nothing else",
    "with nothing else after", "the completion judge",
    # the recording meta-instruction (the ironic self-record)
    "do not record the goal", "do not record your own progress",
    "do not record the assistant", "record only genuinely new",
    "only genuinely new cross-cutting principles",
    # research / synthesis procedure directives
    "work only from memory", "do not search the web or extract",
    "cover each subtopic once", "do not restate what you already",
    "recall memory separately for each", "do not rely on the facts already in",
    "cite the fact ids",
)


def is_task_process_meta(content: str) -> bool:
    """True if `content` is a TASK/PROCESS directive lifted from the goal prompt
    (how to FORMAT or RUN the synthesis/quiz/research task) rather than external
    domain knowledge.

    Conservative: explicit instruction-shaped substrings only; bare topic words
    are absent so a real fact about the user's memory project is never dropped.
    """
    if not content:
        return False
    c = content.lower()
    return any(p in c for p in _TASK_META_PHRASES)
