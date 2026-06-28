"""
entity_extractor.py — Enhanced Entity Extraction for Resonant Lattice Memory
==============================================================================
A two-layer system (spaCy NER + a 14-pattern confidence-scored regex):

  Layer 1 (spaCy NER — optional, best quality):
    Uses en_core_web_sm for named entity recognition.
    Install: pip install spacy && python -m spacy download en_core_web_sm
    Falls back gracefully if spaCy or the model is unavailable.

  Layer 2 (Enhanced regex — always runs):
    14 targeted patterns covering the gaps the original missed:
      • Single-word proper nouns and ALL-CAPS acronyms
      • Technical identifiers (model names, GPU model strings)
      • Version strings and quantized model tags
      • IP addresses and network endpoints
      • Measurements with units (GB, MHz, ms, etc.)
      • File paths and dotfile references
      • Numeric identifiers (port numbers, context lengths)
      • AKA / also-known-as aliases (original, kept)
      • Quoted terms (original, kept + improved)
      • Capitalized multi-word phrases (original, kept + improved)

  Post-processing:
    • Deduplication (case-insensitive)
    • Stopword filtering (removes common English words caught as capitalized)
    • Length filtering (2–80 chars)
    • Lowercasing for consistent entity graph keys

INTEGRATION: store.py imports `extract_entities` from this module at load time
(see the try/except around `from entity_extractor import extract_entities`).
"""

import re
import sys
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Stopword list — capitalized words that are NOT entities
# These appear at sentence starts or in common phrases and create noise.
# ─────────────────────────────────────────────────────────────────────────────
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "it", "its", "be", "been",
    "was", "are", "were", "has", "have", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "shall",
    "not", "no", "so", "as", "up", "out", "this", "that", "these", "those",
    "then", "than", "also", "just", "now", "here", "there", "when", "what",
    "which", "who", "how", "all", "any", "each", "both", "few", "more",
    "most", "other", "some", "such", "into", "about", "after", "before",
    "because", "while", "during", "through", "between", "very", "too",
    "can", "new", "old", "good", "well", "much", "only", "even", "still",
    # Technical non-entities
    "use", "run", "set", "get", "add", "put", "let", "see", "yes", "ok",
    "note", "type", "file", "data", "code", "test", "true", "false", "none",
    # Common short acronyms that are rarely standalone entities
    "id", "vs", "no", "ok", "yes", "ui", "ux", "os", "db",
})

# spaCy entity label types we care about (exclude pure numerics and temporals)
_SPACY_LABELS = frozenset({
    "PERSON", "ORG", "GPE", "LOC", "FAC",   # Who/where
    "PRODUCT", "WORK_OF_ART", "LANGUAGE",    # What (things)
    "NORP", "EVENT",                          # Groups/events
    "QUANTITY", "MONEY",                      # Amounts with context
})


# ─────────────────────────────────────────────────────────────────────────────
# Compiled regex patterns (compiled once at import time)
# ─────────────────────────────────────────────────────────────────────────────
_PATTERNS = [

    # 1. Capitalized multi-word phrases: "Charlie Brown", "Rural Indiana Lab"
    #    Original pattern — kept, slightly tightened to require 2+ words
    (re.compile(r'\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})+)\b'), 1),

    # 2. Mixed-case proper words: "iPhone", "GitHub", "HuggingFace"
    #    Catches camelCase product names that look like proper nouns
    (re.compile(r'\b([A-Z][a-z]+[A-Z]\w*)\b'), 1),

    # 3. ALL-CAPS acronyms (2–6 chars): RTX, GPU, FPV, API, RFM, LLM, UI
    #    Filter: not a stopword, not a common English word
    (re.compile(r'\b([A-Z]{2,6})\b'), 1),

    # 4. Technical model identifiers: granite-4.1-30b, nomic-embed-text,
    #    granite-16k, qwen3.5:0.8b, llama-server, faster-whisper
    #    Pattern: letter(s) + hyphen/dot + alphanumeric suffix(es)
    (re.compile(
        r'\b([A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+){1,6})\b'
    ), 1),

    # 5. GPU / hardware model strings: "RTX 3090 Ti", "GTX 1660", "A100 SXM"
    #    Specific to GPU-style naming: letters + space + 3-4 digits + optional suffix
    (re.compile(r'\b([A-Z]{2,4}\s+\d{3,4}(?:\s+(?:Ti|Super|XT|XTX|SXM|OC))?)\b'), 1),

    # 6. Quantized model tags: Q4_K_XL, Q5_0, Q8_0, UD-Q4_K_XL
    #    These are the storage/quantization suffixes in GGUF model names
    (re.compile(r'\b([A-Z0-9]{1,4}[-_]Q\d_[A-Z0-9_]+)\b'), 1),

    # 7. File paths and dotfiles: ~/.hermes/memory.db, /etc/config, ./build/bin
    #    Paths are often referenced in technical memory about the user's setup
    (re.compile(r'((?:~|\.{1,2})?/[\w./\-_]{4,60})'), 1),

    # 8. Python/package identifiers: sqlite_vec, pysqlite3, numpy, sentence_transformers
    #    Pattern: snake_case names that are clearly library/module identifiers
    (re.compile(r'\b([a-z][a-z0-9]*(?:_[a-z][a-z0-9]+){1,4})\b'), 1),

    # 9. Network endpoints: 192.0.2.10:11434, localhost:8174
    #    Captures IP:port and hostname:port patterns
    (re.compile(
        r'\b((?:\d{1,3}\.){3}\d{1,3}(?::\d{2,5})?'
        r'|localhost(?::\d{2,5})?)\b'
    ), 1),

    # 10. Measurements with units: 24 GB, 6 GB VRAM, 69420 context, 300 ms
    #     Common hardware/performance specs in technical memory
    (re.compile(
        r'\b(\d+(?:\.\d+)?\s*'
        r'(?:GB|MB|KB|TB|GiB|MiB|KiB'        # storage
        r'|GHz|MHz|Hz'                          # frequency
        r'|ms|ns|μs|us|seconds?|minutes?'      # time
        r'|VRAM|RAM|TFLOPS|FLOPS'              # compute
        r'|cores?|threads?|tokens?'            # compute units
        r'|fps|FPS|RPM|Mbps|Gbps))\b'         # rates
    ), 1),

    # 11. Quoted terms — double quotes: "Resonant Lattice", "hello world"
    #     Original pattern, kept. Intentionally quoted = probably named.
    (re.compile(r'"([^"]{2,60})"'), 1),

    # 12. Quoted terms — single quotes: 'charlie', 'the thing'
    #     Filter in post-processing to avoid possessives (it's, don't)
    (re.compile(r"(?<!\w)'([^']{2,60})'(?!\w)"), 1),

    # 13. AKA / also-known-as aliases
    #     Original pattern, kept. Captures both terms in the alias relationship.
    (re.compile(
        r'(\w+(?:\s+\w+){0,4})\s+(?:aka|also\s+known\s+as|a\.k\.a\.?)\s+'
        r'(\w+(?:\s+\w+){0,4})',
        re.IGNORECASE
    ), 2),  # 2 = extract both groups

    # 14. Numeric-only identifiers: port 8174, context 69420
    #     These are often meaningful as named configuration values.
    #     Only capture when adjacent to a labelling word.
    (re.compile(
        r'\b((?:port|context|limit|timeout|dim|size|count|id|session)\s+\d{2,6})\b',
        re.IGNORECASE
    ), 1),
]


# ─────────────────────────────────────────────────────────────────────────────
# Per-pattern confidence + kind (parallel to _PATTERNS, 1-based index → meta).
#   base:  prior confidence in [0,1] that a match of this pattern is a real entity
#   kind:  "normal"  — use base directly
#          "hyphen"  — pattern 4: hyphen/dot identifiers (noisy) → reject/boost heuristic
#          "snake"   — pattern 8: snake_case identifiers (noisy) → reject/boost heuristic
# Quoted/AKA/proper-noun/NER patterns are trusted; pattern 4 & 8 are the noise
# sources (well-being, state-of-the-art, foo_bar) and earn their keep only via
# the vocab booster or a digit/case signal.
# ─────────────────────────────────────────────────────────────────────────────
_PATTERN_META = {
    1:  (0.80, "normal"),   # Capitalized multi-word phrases
    2:  (0.80, "normal"),   # Mixed-case (iPhone, GitHub)
    3:  (0.60, "normal"),   # ALL-CAPS acronym (vocab-boostable)
    4:  (0.30, "hyphen"),   # hyphen/dot identifiers — NOISY
    5:  (0.70, "normal"),   # GPU/hardware strings
    6:  (0.60, "normal"),   # Quantized tags
    7:  (0.55, "normal"),   # File paths
    8:  (0.30, "snake"),    # snake_case — NOISY
    9:  (0.80, "normal"),   # Network endpoints
    10: (0.60, "normal"),   # Measurements
    11: (0.90, "normal"),   # Double-quoted
    12: (0.85, "normal"),   # Single-quoted
    13: (0.90, "normal"),   # AKA aliases
    14: (0.60, "normal"),   # Numeric identifiers
}

_SPACY_CONFIDENCE = 1.0       # NER hits are trusted highest
_MIN_CONFIDENCE = 0.50        # candidates below this are dropped


# ─────────────────────────────────────────────────────────────────────────────
# Import-free technical vocabulary (confidence booster, NOT a gate).
# Harvested from the stdlib module list and INSTALLED distribution names without
# importing any of them — so it reflects the real environment, adds zero runtime
# dependencies, and keeps extraction general-purpose (unknown tokens still count;
# known tech names just score higher). Built lazily + cached on first use so the
# (possibly duplicate) module load does not pay for it twice.
# ─────────────────────────────────────────────────────────────────────────────
_TECH_VOCAB: Optional[frozenset] = None


def _build_tech_vocab() -> frozenset:
    vocab: set = set()
    try:
        vocab |= {m.lower() for m in sys.stdlib_module_names}  # Python 3.10+
    except Exception:
        pass
    try:
        import importlib.metadata as _md
        for dist in _md.distributions():
            try:
                nm = (dist.metadata["Name"] or "").strip().lower()
            except Exception:
                nm = ""
            if nm:
                vocab.add(nm)
                vocab.add(nm.replace("-", "_"))
                vocab.add(nm.replace("_", "-"))
    except Exception:
        pass
    return frozenset(vocab)


def _get_tech_vocab() -> frozenset:
    global _TECH_VOCAB
    if _TECH_VOCAB is None:
        _TECH_VOCAB = _build_tech_vocab()
    return _TECH_VOCAB


def _in_vocab(token: str) -> bool:
    """True if the token (or a hyphen/underscore-normalized form) is a known
    stdlib module or installed package name."""
    t = token.lower()
    vocab = _get_tech_vocab()
    if t in vocab:
        return True
    if t.replace("-", "_") in vocab or t.replace("_", "-") in vocab:
        return True
    # First path-ish segment (e.g. "numpy.linalg" → "numpy")
    head = re.split(r'[.\-_:]', t, 1)[0]
    return bool(head) and head in vocab


def _score_noisy_candidate(candidate: str, base: float) -> float:
    """Confidence for a hyphen/dot/snake candidate (patterns 4 & 8).

    Boost to high confidence when there's a real signal:
      - the token (or its head) is a known module/package name (vocab), or
      - it contains a digit (model/version strings: granite-4.1-30b, granite-16k), or
      - it has internal uppercase (GitHub-Actions, faster-Whisper).
    Otherwise it is almost certainly an ordinary English compound
    (well-being, state-of-the-art, long-term) → push below threshold so it's dropped.
    """
    if _in_vocab(candidate):
        return 0.85
    if any(ch.isdigit() for ch in candidate):
        return 0.75
    if any(ch.isupper() for ch in candidate):
        return 0.65
    return 0.20  # below _MIN_CONFIDENCE → rejected


class EntityExtractor:
    """
    Two-layer entity extractor: spaCy NER (if available) + enhanced regex.

    Usage:
        extractor = EntityExtractor()          # module-level singleton
        entities = extractor.extract("...")    # returns List[str]

    Thread-safe: the extract() method holds no mutable state and can be called
    from multiple threads simultaneously without locks.
    """

    def __init__(self) -> None:
        self._nlp = None
        self._spacy_available = False
        self._spacy_attempted = False   # lazy: load on first extract(), not at import

    def _ensure_spacy(self) -> None:
        """Load spaCy + en_core_web_sm on first use (idempotent, fails silently).

        Lazy on purpose: the host loader may import this module twice (dotted +
        bare namespace). Deferring the model load to first extract() means the
        duplicate module — which never has extract() called on it — never pays
        the ~50-100ms model-load cost or holds a second copy in RAM.
        """
        if self._spacy_attempted:
            return
        self._spacy_attempted = True
        try:
            import spacy  # type: ignore[import]
            try:
                self._nlp = spacy.load("en_core_web_sm")
                self._spacy_available = True
                logger.info(
                    "EntityExtractor: spaCy en_core_web_sm loaded — NER active (best quality)."
                )
            except OSError:
                logger.warning(
                    "EntityExtractor: spaCy installed but en_core_web_sm not found. "
                    "Run: python -m spacy download en_core_web_sm\n"
                    "Falling back to enhanced regex extraction."
                )
        except ImportError:
            logger.info(
                "EntityExtractor: spaCy not installed — enhanced regex only. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )

    def extract(self, text: str) -> List[str]:
        """Extract named entities from text.

        Two layers (spaCy NER + 14-pattern regex) feed a confidence model.
        Each candidate gets a confidence in [0,1]; candidates below
        _MIN_CONFIDENCE are dropped. The noisy identifier patterns (4 hyphen/dot,
        8 snake_case) only survive when a real signal is present (known
        module/package name, a digit, or internal uppercase) — so ordinary
        English compounds like "well-being" or "state-of-the-art" no longer
        pollute the entity graph, while "granite-4.1-30b" and "sqlite_vec" do.

        Stays general-purpose: unknown tokens still count; the tech vocab only
        *raises* confidence, never gates. Returns a deduplicated, lowercased
        list ordered by confidence (highest first).
        """
        if not text or not text.strip():
            return []

        self._ensure_spacy()

        best: dict = {}   # lower-name → confidence (keep the max)

        def _consider(name: str, confidence: float) -> None:
            name = name.strip().strip("'\".,;:!?()")
            if not name:
                return
            lower = name.lower()
            if len(lower) < 2 or len(lower) > 80:
                return
            if lower in _STOPWORDS:
                return
            if confidence < _MIN_CONFIDENCE:
                return
            if confidence > best.get(lower, 0.0):
                best[lower] = confidence

        # ── Layer 1: spaCy NER (trusted highest) ──────────────────────
        if self._spacy_available and self._nlp is not None:
            try:
                doc = self._nlp(text[:100_000])
                for ent in doc.ents:
                    if ent.label_ in _SPACY_LABELS:
                        _consider(ent.text, _SPACY_CONFIDENCE)
            except Exception as e:
                logger.debug("spaCy NER failed on this text: %s", e)

        # ── Layer 2: Enhanced regex with per-pattern confidence ───────
        for idx, (pattern, n_groups) in enumerate(_PATTERNS, start=1):
            base, kind = _PATTERN_META.get(idx, (0.5, "normal"))
            for match in pattern.finditer(text):
                if n_groups == 2:
                    # AKA pattern — both sides are explicitly named → trust base
                    _consider(match.group(1), base)
                    _consider(match.group(2), base)
                    continue

                candidate = match.group(1)

                if kind == "normal":
                    # ALL-CAPS acronym (pattern 3): common word in caps → drop;
                    # known acronym in vocab → boost.
                    if idx == 3:
                        if candidate.lower() in _STOPWORDS:
                            continue
                        conf = 0.85 if _in_vocab(candidate) else base
                        _consider(candidate, conf)
                    else:
                        _consider(candidate, base)
                else:
                    # Noisy identifier patterns (4 hyphen/dot, 8 snake_case)
                    if kind == "snake" and "_" in candidate and len(candidate) < 6:
                        continue  # tiny snake_case is usually a loop var (i_j, x_y)
                    _consider(candidate, _score_noisy_candidate(candidate, base))

        # Highest-confidence first, lowercased for consistent entity-graph keys
        return [name for name, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)]

    @property
    def mode(self) -> str:
        """Returns 'spacy+regex' or 'regex-only' — useful for logging.

        Note: reflects state only after the first extract() (lazy spaCy load).
        """
        return "spacy+regex" if self._spacy_available else "regex-only"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton — imported by store.py
# Created once at module load time; shared across all LatticeStore instances.
# spaCy + the tech vocab load lazily on the first extract() call.
# ─────────────────────────────────────────────────────────────────────────────
_extractor = EntityExtractor()


def extract_entities(text: str) -> List[str]:
    """
    Public interface — module-level function wrapping the singleton.
    Drop-in replacement for LatticeStore._extract_entities().

    Args:
        text: Raw fact content string.

    Returns:
        Deduplicated, lowercased list of entity strings.
    """
    return _extractor.extract(text)
