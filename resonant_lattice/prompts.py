"""
prompts.py — default LLM prompt strings for the Resonant Lattice provider.

Text only. These are the *defaults* used when the corresponding config key is
absent; a config value overrides them (see LatticeMemoryProvider.__init__).
Literals are kept in implicit-concatenation form so their concatenated values
are byte-identical to the originals.
"""

DEFAULT_PROCEDURAL_PROMPT = (
    "You are a procedural memory engine for an AI agent. Below are recent "
    "records of the agent calling one tool, each tagged SUCCESS or FAILURE "
    "with its arguments and outcome.\n\n"
    "Synthesize concise, REUSABLE procedural rules to help the agent use "
    "this tool better next time:\n"
    "- Argument/context patterns that tend to SUCCEED\n"
    "- Argument/context patterns that tend to FAIL (especially valuable)\n"
    "- Preconditions, gotchas, effective usage patterns\n\n"
    "Rules:\n"
    "- Each rule must GENERALIZE across the records, never restate one call\n"
    "- Write each rule as a standalone, actionable fact\n"
    "- If any FAILUREs occurred, include at least one failure-avoidance rule\n"
    "- Output ONLY a valid JSON array; each object has keys \"content\" and "
    "\"category\" (use \"procedural\")\n"
    "- Output 1-4 rules, or [] if nothing generalizable can be learned"
)

DEFAULT_EXTRACTION_PROMPT = (
    "Analyze this dialogue log and extract only NEW, durable facts, user "
    "preferences, or goal states that are EXPLICITLY supported by the log.\n"
    "GROUNDING RULES (critical — violating them corrupts memory):\n"
    "- Do NOT invent or infer specifics (names, numbers, dates, IDs, paths, "
    "settings, versions) that are not literally present in the log. If a "
    "detail is not in the log, leave it out.\n"
    "- For EVERY fact, include \"source_quote\": the shortest snippet copied "
    "VERBATIM from the log (character-for-character, no paraphrase) that "
    "supports the fact.\n"
    "- If the supporting turn came from a tool or web fetch that carried a "
    "URL or identifier, also include \"source_ref\" with that URL/ref; "
    "otherwise omit source_ref or set it to null.\n"
    "- If a candidate fact has no exact supporting snippet in the log, DROP "
    "it rather than guessing.\n"
    "Output ONLY a valid JSON array of objects with keys: \"content\", "
    "\"category\", \"source_quote\", and optional \"source_ref\". "
    "If nothing new is learned, output an empty array []."
)

DEFAULT_CONSOLIDATION_PROMPT = (
    "You are an expert memory abstraction engine for an AI agent.\n\n"
    "Given the following group of related long-term facts, synthesize 1-2 higher-level "
    "abstractions that capture the general pattern WITHOUT erasing what makes each fact "
    "true in its OWN situation.\n\n"
    "Rules:\n"
    "- Abstraction is CONTEXTUALIZATION, not erasure. When the facts give different answers "
    "in different conditions, state the DEFAULT and PRESERVE the exceptions as scoped "
    "conditions, e.g. 'X by default; Y when <condition>'. The conditions that make each fact "
    "correct in its situation are the POINT — keep them, don't discard them as 'detail'.\n"
    "- Do NOT collapse facts that hold in distinct contexts into one detail-free claim, and "
    "NEVER invent a generalization the facts do not support.\n"
    "- Concise but meaningful; one principle per object\n"
    "- Output ONLY a valid JSON array (no extra text)\n"
    "- Each object must have keys: \"content\" and \"category\" (use \"abstract\")"
)

DEFAULT_NARRATIVE_PROMPT = (
    "Summarise the session below as ONE short paragraph of durable "
    "autobiographical memory for an AI agent — what the user and assistant worked "
    "on and decided together, the kind of throughline worth remembering at the "
    "start of the next session.\n\n"
    "Rules:\n"
    "- Frame it as a remembered summary, NOT a transcript or a list of turns\n"
    "- Keep the throughline; drop turn-by-turn detail and exact wording\n"
    "- NEVER invent anything not present in the log\n"
    "- One paragraph, a few sentences at most\n"
    "- Output ONLY the paragraph, with no preamble or headings"
)

DEFAULT_RELATION_PROMPT = (
    "Extract the explicit (subject, relation, object) triples STATED in the text "
    "below. Capture only relationships LITERALLY present — never infer, chain, or "
    "add outside world knowledge (that is done elsewhere, deliberately, and never "
    "stored as fact).\n\n"
    "Rules:\n"
    "- relation is a short snake_case verb phrase (e.g. works_at, lives_in, prefers)\n"
    "- subject and object are the concrete entities/terms named in the text\n"
    "- Do NOT emit a triple unless both subject and object appear in the text\n"
    "- Output ONLY a valid JSON array of objects with keys \"subject\", "
    "\"relation\", \"object\", or [] if no explicit relationship is stated"
)

DEFAULT_GIST_PROMPT = (
    "You are a memory consolidation engine for an AI agent. The facts below are "
    "FADING from memory (their resonance has decayed toward zero) but they "
    "mattered once. Before they are forgotten entirely, write ONE concise GIST "
    "that compresses the NARRATIVE while KEEPING the load-bearing specifics — so "
    "the meaning survives without losing the values an agent may need to act on.\n\n"
    "Rules:\n"
    "- PRESERVE exact numbers, money amounts, IDs, and dates VERBATIM. For money / "
    "compliance / spec facts the specific value IS the meaning — rounding 4050 "
    "cents to 'about $40' or dropping an invoice ID is WORSE than forgetting. "
    "Compress the surrounding prose, not the hard values.\n"
    "- Capture the common theme AND list the concrete values it covers, e.g. "
    "'Acme March invoices: 4050, 9900, 12500 cents for hosting/support/egress'.\n"
    "- Frame it as a remembered summary/generalization, NOT a verbatim transcript\n"
    "- NEVER invent specifics that are not present in the facts below\n"
    "- Output ONLY a valid JSON array with a SINGLE object with keys \"content\" "
    "and \"category\" (use \"gist\"), or [] if there is no shared meaning worth keeping"
)
