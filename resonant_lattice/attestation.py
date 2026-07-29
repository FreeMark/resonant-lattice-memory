"""
attestation.py - source_quote attestation (two-channel grounding verifier).

Phase D+. Pure (re + difflib + a passed entity list); no framework or heavy
dependencies, so it is importable standalone (the test suite loads it directly).

A single similarity score (embedding cosine, HRR) CANNOT separate a trivial
typo from a critical one: both 'teh'->'the' and '11434'->'11435' are a one-edit
change, and embeddings deliberately blur surface specifics (numbers/IDs barely
move them). The importance of a token is orthogonal to its string/semantic
distance - so we decompose the quote and route each token-class to the metric
that is actually discriminative for it:

  • PROSE channel (fuzzy, typo-forgiving): difflib over a WINDOW of the
    transcript (~len(quote)) centred on the best-matching region, plus a
    longest-contiguous coverage backstop. Windowing keeps the score - and the
    quote_match_threshold knob - meaningful regardless of transcript length: a
    whole-transcript ratio is 2M/(len(quote)+len(transcript)) and collapses
    toward 0 on a long log, so the knob would otherwise be inert. Tolerates
    typos, whitespace and casing.
  • SPECIFICS channel (exact, unforgiving): the load-bearing tokens. Numbers /
    IDs / ports / versions (digit-cores) are checked by SET MEMBERSHIP against
    the transcript's number tokens - NOT as a substring of one concatenated
    digit blob, which would let a fabricated number slip through by spanning
    unrelated tokens (e.g. "3014" inside the blob for "...4.1 30b ... 14 ...
    34"). Named entities (reusing the system's own extractor) must appear
    verbatim; that channel inherits the extractor's high-precision/low-recall
    bias, so a fabricated token it does not recognise as an entity is never
    checked here and falls through to the prose channel. This is what catches
    '11434'->'11435' and a fabricated id/name.

Verdict from the COMBINATION (this is the whole point):
  trivial typo   → prose OK + specifics OK            → 'attested'
  no specifics   → prose OK, nothing hard to pin      → 'soft'
  critical typo  → a hard specific is absent          → 'specific_mismatch' (DROP)
  fabrication    → prose not anchored, no specific hit → 'unattested' (keep+flag)
Runs on the async consolidation cycle, so latency is cheap.
"""

import re
import difflib
from typing import List

_QUOTE_NUM_TOKEN_RE = re.compile(r"\d[\d.,:_\-/]*\d|\d")


def _normalize_for_match(text: str) -> str:
    """Lowercase + collapse whitespace + straighten quotes for fuzzy comparison."""
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text.lower()).strip()


def _digit_core(token: str) -> str:
    """Strip everything but digits - compare numeric identifiers exactly."""
    return re.sub(r"\D", "", token)


def _attest_source_quote(quote: str, transcript: str, entities: List[str],
                         ratio_threshold: float = 0.82) -> str:
    """Grounding verdict for `quote` against `transcript`. See module note above.

    Returns one of 'attested' | 'soft' | 'specific_mismatch' | 'unattested'.
    The SPECIFICS channel is checked first: a fabricated/changed numeric token or
    named entity short-circuits to 'specific_mismatch' (the caller DROPS the fact)
    regardless of how convincing the surrounding prose is - that is exactly the
    'important typo' case a fuzzy score would wave through.
    """
    if not quote or not transcript:
        return "unattested"
    nt = _normalize_for_match(transcript)

    # --- Specifics channel (exact, unforgiving) ---
    # FIX 1: compare number tokens by SET MEMBERSHIP of their digit-cores, not as
    # substrings of one concatenated digit blob. A blob-substring test lets a
    # fabricated number pass when it merely spans unrelated tokens (e.g. "3014" is
    # a substring of the blob "41301434" for "...4.1 30b ... 14 ... 34").
    # Two views of the transcript's number tokens (by digit-core): the full set
    # (including single digits) and the >=2-digit set. Multi-digit cores are the
    # load-bearing identifiers; single digits are handled asymmetrically below.
    nt_num_cores_all = {
        c for c in (_digit_core(t) for t in _QUOTE_NUM_TOKEN_RE.findall(transcript)) if c
    }
    nt_num_cores = {c for c in nt_num_cores_all if len(c) >= 2}
    has_hard = False
    for tok in _QUOTE_NUM_TOKEN_RE.findall(quote):
        core = _digit_core(tok)
        if not core:
            continue
        if len(core) >= 2:
            has_hard = True
            if core not in nt_num_cores:
                return "specific_mismatch"   # fabricated/changed number → DROP
        elif core in nt_num_cores_all:
            # FIX 3: a lone digit (e.g. "purchase 3") is ASYMMETRIC. When it IS
            # present in the transcript it counts as a confirming specific, so a
            # faithfully-lifted single-number quote can reach 'attested' instead
            # of merely 'soft'. But its ABSENCE never triggers a drop: lone
            # digits are ubiquitous (list markers "1.", "1-2 rules") and
            # word<->digit normalization ("three"<->"3") would otherwise cause
            # false 'specific_mismatch' drops of legitimate facts. Catching a
            # changed lone digit reliably is impossible over a transcript, so we
            # keep+flag ('soft') rather than over-block.
            has_hard = True
    for ent in (entities or []):
        if len(ent) < 3:
            continue  # too short to verify reliably
        has_hard = True
        # The entity channel inherits the extractor's high-precision/low-recall
        # bias: a fabricated token the extractor doesn't recognise as an entity is
        # never passed in here, so it isn't checked and falls through to prose.
        if _normalize_for_match(ent) not in nt:
            return "specific_mismatch"   # fabricated/changed named entity → DROP

    # --- Prose channel (fuzzy, typo-forgiving) ---
    nq = _normalize_for_match(quote)
    # autojunk=False: on a long transcript difflib's auto-junk heuristic marks
    # common chars as "popular" and excludes them from matching, fragmenting the
    # longest match and corrupting coverage. Disable it so alignment is reliable.
    sm = difflib.SequenceMatcher(None, nq, nt, autojunk=False)
    m = sm.find_longest_match(0, len(nq), 0, len(nt))
    coverage = m.size / max(len(nq), 1)
    # FIX 2: score the fuzzy ratio over a WINDOW of the transcript ~len(quote)
    # around the best match - NOT the whole transcript. A whole-transcript ratio
    # collapses toward 0 on a long log, making ratio_threshold inert (coverage
    # silently did all the work). Windowing keeps the score - and the
    # quote_match_threshold knob - meaningful at any transcript length.
    qstart = max(0, m.b - m.a)                 # transcript offset the quote aligns to
    pad = max(4, len(nq) // 8)                 # small slack for boundary drift
    window = nt[max(0, qstart - pad): qstart + len(nq) + pad]
    windowed_ratio = (
        difflib.SequenceMatcher(None, nq, window).ratio() if window else 0.0
    )
    prose_ok = windowed_ratio >= ratio_threshold or coverage >= 0.6

    if prose_ok:
        return "attested" if has_hard else "soft"
    return "unattested"   # prose not anchored, but nothing hard contradicted → keep+flag


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT ↔ QUOTE numeric conflict - the link _attest_source_quote does not check.
#
# _attest_source_quote verifies the QUOTE against the TRANSCRIPT. Nothing verifies the
# FACT's own content against its quote, so a fact can carry numbers contradicting the
# very snippet stored as its evidence and still be 'attested'. Measured on a live
# 6,983-fact clinical corpus:
#
#   content: "Normal canine body temperature range is 99.5 to 102.5 degrees F"
#   quote  : "normal body temperature for a dog is between 99.8-102.8 degrees F"
#   status : attested
#
# Both are plausible published ranges, so no reader catches it either.
#
# WHY THIS IS NARROW ON PURPOSE. "A content number absent from the quote" is far too
# broad - on that corpus it fires on 17% of reference-range facts, and reading them,
# nearly all are legitimate: the quote was captured too narrowly, or the model DERIVED
# a value correctly (0.080 mg/L -> 80 ppb; 1.24 g/L -> 124 mg/dL; a 40x objective ->
# 400x magnification). Shipping that as a drop would destroy ~83 good facts to catch 1.
#
# A derivation restates a value in other units and lands ORDERS OF MAGNITUDE from
# anything in the quote. A CONTRADICTION states the same measurement with a different
# value and lands right beside a quote number without matching it. So a conflict
# requires a NEAR MISS, excluding rounding - the dominant benign case (58% for 57.9%,
# 37.1% for 37.06%, "approximately 40%" for 39.7%).
#
# Measured at tolerance 0.02 over 6,640 quoted facts: 10 flags (0.15%), including the
# real contradiction. Precision is roughly a third, which is why the caller MARKS
# rather than drops, and why the check is OFF unless a tolerance is configured.
# ─────────────────────────────────────────────────────────────────────────────

_DEC_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d\d\d\b)")
_DASH_MAP = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
# Bibliographic numbers can never appear in a source quote, so counting them as missing
# clinical values inflates the signal: '(as_of 2021)' is an annotation this system adds,
# '(Laboklin 2024)' is a citation, 'Table 3' a locator. Stripped by SHAPE, not by
# removing every 4-digit number - a reference range may legitimately read "2000 mL".
_CITATION_RES = (
    re.compile(r"\(\s*as[_ ]of[^)]*\)", re.I),
    re.compile(r"\((?=[^)]*\b(?:19|20)\d\d\b)[^)]*\)"),
    re.compile(r"\b(?:table|figure|fig\.?|chapter|section|appendix|part)\s+\d+\b", re.I),
)


def _decimal_numbers(text: str, strip_citations: bool = False) -> set:
    """Decimal-AWARE value set: 2.9 and 5.3 stay separate values.

    Deliberately NOT _digit_core: stripping non-digits turns '2.9-5.3' into '2953' and
    '3.5-6.2' into '3562' - unequal by luck rather than design - while '2.9 to 5.3' and
    '2.9-5.3' come out unequal despite being the same interval.
    """
    s = (text or "").translate(_DASH_MAP)
    s = _THOUSANDS_RE.sub("", s)
    if strip_citations:
        for rx in _CITATION_RES:
            s = rx.sub(" ", s)
    out = set()
    for tok in _DEC_NUM_RE.findall(s):
        try:
            out.add(float(tok))
        except ValueError:
            pass
    return out


def _is_rounding(a: float, b: float) -> bool:
    """True when one value is the other restated with fewer decimal places.

    Checked in BOTH directions, and only for decimal-place rounding, so 99.8 -> 99.5 and
    302 -> 300 are NOT exempted: those are different measurements, not shorter ones.
    """
    for d in range(0, 4):
        if round(b, d) == a or round(a, d) == b:
            return True
    return False


def content_quote_numeric_conflict(content: str, quote: str,
                                   tolerance: float = 0.02) -> List[tuple]:
    """[(content_value, nearest_quote_value, relative_distance), …]; empty if none.

    A conflict is a number in ``content`` ABSENT from ``quote`` while ``quote`` holds a
    DIFFERENT number within ``tolerance`` relative distance that is not merely a
    rounding of it. ``tolerance <= 0`` disables the check and returns [].
    """
    if tolerance <= 0 or not content or not quote:
        return []
    cn = _decimal_numbers(content, strip_citations=True)
    qn = _decimal_numbers(quote)
    if not cn or not qn:
        return []
    out = []
    for c in sorted(cn - qn):
        best = None
        for q in qn:
            denom = max(abs(c), abs(q))
            if q == c or denom == 0:
                continue
            d = abs(c - q) / denom
            if d <= tolerance and not _is_rounding(c, q) and (best is None or d < best[1]):
                best = (q, d)
        if best is not None:
            out.append((c, best[0], round(best[1], 4)))
    return out
