"""
Holographic Reduced Representations (HRR) with phase encoding.

HRRs are a vector symbolic architecture for encoding compositional structure
into fixed-width distributed representations. This module uses *phase vectors*:
each concept is a vector of angles in [0, 2π). The algebraic operations are:

  bind   — circular convolution (phase addition)  — associates two concepts
  unbind — circular correlation (phase subtraction) — retrieves a bound value
  bundle — superposition (circular mean)           — merges multiple concepts

Phase encoding is numerically stable, avoids the magnitude collapse of
traditional complex-number HRRs, and maps cleanly to cosine similarity.

Atoms are generated deterministically from SHA-256 so representations are
identical across processes, machines, and language versions.

References:
  Plate (1995) — Holographic Reduced Representations
  Gayler (2004) — Vector Symbolic Architectures answer Jackendoff's challenges

ENHANCEMENTS (rich encoding, v2):
  - _tokenize() shared helper
  - encode_text_rich() — unigram BoW + positional binding + NON-commutative
    rolled bigrams (recommended for encode_fact / conflict / abstraction)
  - encode_fact() uses encode_text_rich() internally

All original functions (encode_atom, bind, unbind, bundle, similarity,
encode_text, phases_to_bytes, bytes_to_phases, snr_estimate) remain unchanged
for backward compatibility. Existing stored HRR vectors continue to work.
"""

import hashlib
import logging
import struct
import math
from typing import List

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

logger = logging.getLogger(__name__)

_TWO_PI = 2.0 * math.pi


def _require_numpy() -> None:
    if not _HAS_NUMPY:
        raise RuntimeError("numpy is required for holographic operations")


_ATOM_CACHE: dict = {}          # (word, dim) -> bytes (immutable float64 payload)
_ATOM_CACHE_MAX = 4096          # ~32 MB worst case at dim=1024


def encode_atom(word: str, dim: int = 1024) -> "np.ndarray":
    """Deterministic phase vector via SHA-256 counter blocks (memoized).

    Uses hashlib (not numpy RNG) for cross-platform reproducibility.

    Algorithm:
    - Generate enough SHA-256 blocks by hashing f"{word}:{i}" for i=0,1,2,...
    - Concatenate digests, interpret as uint16 values via struct.unpack
    - Scale to [0, 2π): phases = values * (2π / 65536)
    - Truncate to dim elements
    - Returns np.float64 array of shape (dim,)

    Results are cached as immutable bytes keyed by (word, dim); each call
    returns a fresh mutable copy, so callers may safely mutate the result.
    """
    _require_numpy()

    key = (word, dim)
    cached = _ATOM_CACHE.get(key)
    if cached is not None:
        return np.frombuffer(cached, dtype=np.float64).copy()

    # Each SHA-256 digest is 32 bytes = 16 uint16 values.
    values_per_block = 16
    blocks_needed = math.ceil(dim / values_per_block)

    uint16_values: list[int] = []
    for i in range(blocks_needed):
        digest = hashlib.sha256(f"{word}:{i}".encode()).digest()
        uint16_values.extend(struct.unpack("<16H", digest))

    phases = np.array(uint16_values[:dim], dtype=np.float64) * (_TWO_PI / 65536.0)

    if len(_ATOM_CACHE) >= _ATOM_CACHE_MAX:
        _ATOM_CACHE.clear()   # cheap wholesale reset; working sets are tiny
    _ATOM_CACHE[key] = phases.tobytes()
    return phases


def bind(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    """Circular convolution = element-wise phase addition.

    Binding associates two concepts into a single composite vector.
    The result is dissimilar to both inputs (quasi-orthogonal).
    """
    _require_numpy()
    return (a + b) % _TWO_PI


def unbind(memory: "np.ndarray", key: "np.ndarray") -> "np.ndarray":
    """Circular correlation = element-wise phase subtraction.

    Unbinding retrieves the value associated with a key from a memory vector.
    unbind(bind(a, b), a) ≈ b  (up to superposition noise)
    """
    _require_numpy()
    return (memory - key) % _TWO_PI


def bundle(*vectors: "np.ndarray", dim: int = 1024) -> "np.ndarray":
    """Superposition via circular mean of complex exponentials.

    Bundling merges multiple vectors into one that is similar to each input.
    The result can hold O(sqrt(dim)) items before similarity degrades.

    Empty input returns a deterministic empty-atom vector rather than a numpy
    scalar (np.angle(np.sum([])) collapses to shape ()), which would break every
    downstream consumer (similarity, phases_to_bytes, the dim-length guard).
    """
    _require_numpy()
    if not vectors:
        return encode_atom("__hrr_empty__", dim)
    complex_sum = np.sum([np.exp(1j * v) for v in vectors], axis=0)
    return np.angle(complex_sum) % _TWO_PI


def similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    """Phase cosine similarity. Range [-1, 1].

    Returns 1.0 for identical vectors, near 0.0 for random (unrelated) vectors,
    and -1.0 for perfectly anti-correlated vectors.
    """
    _require_numpy()
    return float(np.mean(np.cos(a - b)))


def hrr_lift(phases: "np.ndarray") -> "np.ndarray":
    """Lift a phase vector to the ``(cos φ, sin φ)/√dim`` real vector whose COSINE equals the
    HRR phase-similarity (E4 4a). Since ``mean(cos(a−b)) = (1/dim)·Σ[cos a·cos b + sin a·sin b]
    = dot(lift(a), lift(b))`` and ``‖lift‖ = 1``, the lifted vector is L2-unit and its cosine
    IS ``similarity(a, b)``.

    This is the client-side bridge that makes HRR recall homomorphic with ZERO new crypto: the
    Tier-1 blind store's existing cosine inner-product (``he_crypto.BlindRecallPRE`` over the
    encrypted HRR lifts in ``semantic_he_hrr``) computes HRR similarity directly on the encrypted
    ``2*dim`` lift — no ``cos`` / ``mod 2π`` / ``bundle`` under HE (all stay plaintext here, per
    roadmap principle 3.3). The store sees only the encrypted lift; the phase angle is not
    recoverable from it without the key. Proven on the node vs ``similarity`` (err ~1e-12).
    """
    _require_numpy()
    ph = np.asarray(phases, dtype=np.float64).ravel()
    n = ph.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate([np.cos(ph), np.sin(ph)]) / math.sqrt(n)


def encode_text(text: str, dim: int = 1024) -> "np.ndarray":
    """Bag-of-words: bundle of atom vectors for each token.

    Now uses the shared _tokenize() helper (consistent with rich variants).
    Drops single-character tokens after punctuation stripping.
    """
    _require_numpy()
    tokens = _tokenize(text)
    if not tokens:
        return encode_atom("__hrr_empty__", dim)
    atom_vectors = [encode_atom(token, dim) for token in tokens]
    return bundle(*atom_vectors)

def _tokenize(text: str) -> List[str]:
    """Shared tokenizer for all encode_text_* variants.
 
    Lowercases, splits on whitespace, strips leading/trailing punctuation
    from each token, and drops empty/single-char results.
 
    Returns a list of clean token strings. Empty input returns [].
    """
    tokens = [
        token.strip(".,!?;:\"'()[]{}-—/\\")
        for token in text.lower().split()
    ]
    return [t for t in tokens if len(t) > 1]
    


def encode_text_rich(text: str, dim: int = 1024) -> "np.ndarray":
    """Rich text encoding: unigram BoW + positional binding + bigram binding.
 
    The combination of all three layers makes the resulting vector:
      - Tolerant of word-for-word synonyms (unigram overlap still high)
      - Sensitive to word-order changes (positional binding diverges)
      - Sensitive to adjacent-word substitutions (bigram binding diverges)
 
    Use this instead of encode_text() wherever order and phrase structure
    matter — specifically in encode_fact() for conflict detection and
    abstraction clustering. encode_text() is preserved for backward
    compatibility and for cases where BoW recall is the priority.
 
    Returns np.float64 array of shape (dim,).
    """
    _require_numpy()
    tokens = _tokenize(text)
    if not tokens:
        return encode_atom("__hrr_empty__", dim)
 
    all_components = []
 
    # Layer 1: Unigram BoW (original behavior — highest recall)
    for token in tokens:
        all_components.append(encode_atom(token, dim))
 
    # Layer 2: Positional binding (word-order sensitivity)
    for i, token in enumerate(tokens):
        pos_atom = encode_atom(f"__pos_{i}__", dim)
        all_components.append(bind(encode_atom(token, dim), pos_atom))
 
    # Layer 3: Bigrams (local phrase sensitivity, NON-commutative)
    # roll(second, 1) breaks the commutativity of phase-addition binding:
    # bind(A, roll(B,1)) != bind(B, roll(A,1)), so "dark themes" and
    # "themes dark" now diverge in this layer too (same trick the old
    # encode_text_ngram used). Encoding change ⇒ rich-v2.
    for i in range(len(tokens) - 1):
        bigram = bind(
            encode_atom(tokens[i], dim),
            np.roll(encode_atom(tokens[i + 1], dim), 1),
        )
        all_components.append(bigram)

    # No SNR warning is emitted here on purpose: conflict detection re-encodes
    # every fact each dream cycle, so a per-encode log would be spammy. Callers
    # that care about bundle capacity can call snr_estimate(dim,
    # len(all_components)) themselves (~dim/4 components is the safe ceiling).
    return bundle(*all_components)



def encode_fact(content: str, entities: list, dim: int = 1024) -> "np.ndarray":
    """Structured HRR encoding with rich content representation.
 
    Encodes a fact as a bundle of:
      1. bind(encode_text_rich(content), ROLE_CONTENT)
            ← upgraded from encode_text(): now positional + bigram aware
      2. bind(encode_atom(entity), ROLE_ENTITY)  for each entity
 
    The rich content encoding means:
      - "user prefers dark themes" and "user prefers light themes" now
        produce meaningfully different content vectors (bigram layer:
        (prefers, dark) vs (prefers, light) diverges).
      - Rephrased facts with same meaning stay similar (BoW layer).
      - Word-order changes (e.g. passive voice) are detectable (positional).
 
    Role vectors are reserved atoms: "__hrr_role_content__", "__hrr_role_entity__"
    These are identical to the original — no DB migration needed.
 
    Enables algebraic extraction (unchanged from original):
        unbind(fact_vec, bind(entity_atom, ROLE_ENTITY)) ≈ content_vector
    """
    _require_numpy()
 
    role_content = encode_atom("__hrr_role_content__", dim)
    role_entity = encode_atom("__hrr_role_entity__", dim)
 
    components = [
        # UPGRADED: was encode_text(content, dim)
        bind(encode_text_rich(content, dim), role_content)
    ]
 
    for entity in entities:
        components.append(
            bind(encode_atom(entity.lower(), dim), role_entity)
        )
 
    return bundle(*components)


def encode_triple(subject: str, relation: str, object_: str, dim: int = 1024) -> "np.ndarray":
    """Role-filler HRR encoding of a (subject, relation, object) triple.

    Builds T = bundle( bind(subj, ROLE_SUBJECT),
                       bind(rel,  ROLE_RELATION),
                       bind(obj,  ROLE_OBJECT) ).

    Because binding is invertible, the stored triple is *queryable* by role —
    this is what Phase 5b uses for fuzzy relational recall:

        unbind(T, ROLE_OBJECT)  ≈ encode_atom(object)     # "(subj, rel, ?)"
        unbind(T, ROLE_SUBJECT) ≈ encode_atom(subject)    # "(?, rel, obj)"

    The three role atoms are reserved deterministic vectors (distinct from the
    encode_fact role atoms), so triple vectors never collide with the fact-level
    content/entity encoding. Args are lowercased for stable atom identity, matching
    the entity-graph normalization. Three superposed bound pairs are well within
    the dim/4 SNR ceiling at dim>=1024.
    """
    _require_numpy()
    role_subject = encode_atom("__hrr_role_subject__", dim)
    role_relation = encode_atom("__hrr_role_relation__", dim)
    role_object = encode_atom("__hrr_role_object__", dim)
    return bundle(
        bind(encode_atom(subject.lower(), dim), role_subject),
        bind(encode_atom(relation.lower(), dim), role_relation),
        bind(encode_atom(object_.lower(), dim), role_object),
    )


def encode_triple_query(subject: str = None, relation: str = None,
                        object_: str = None, dim: int = 1024) -> "np.ndarray":
    """Partial-binding PROBE for relational recall (Phase 5b).

    Bundles only the PROVIDED slots, each bound to its role atom — the same role
    atoms encode_triple uses. Comparing this probe to stored triple vectors with
    similarity() ranks them by how many of the known bindings they contain:

        (mark, lives_in, ?)  → probe = bundle(bind(mark,ROLE_S), bind(lives_in,ROLE_R))
        sim(probe, triple)  ≈ 0.69  when the triple matches BOTH known slots
                            ≈ 0.34  when it matches ONE
                            ≈ 0.00  when it matches NEITHER

    So an exact graph hit ranks highest and partial-structure matches degrade
    gracefully (the fuzzy fallback when no triple satisfies every known slot).
    Returns None if no slot is provided (nothing to probe with).
    """
    _require_numpy()
    components = []
    if subject:
        components.append(bind(encode_atom(subject.lower(), dim),
                               encode_atom("__hrr_role_subject__", dim)))
    if relation:
        components.append(bind(encode_atom(relation.lower(), dim),
                               encode_atom("__hrr_role_relation__", dim)))
    if object_:
        components.append(bind(encode_atom(object_.lower(), dim),
                               encode_atom("__hrr_role_object__", dim)))
    if not components:
        return None
    return bundle(*components)


def phases_to_bytes(phases: "np.ndarray") -> bytes:
    """Serialize phase vector to bytes. float64 tobytes — 8 KB at dim=1024."""
    _require_numpy()
    return phases.tobytes()


def bytes_to_phases(data: bytes) -> "np.ndarray":
    """Deserialize bytes back to phase vector. Inverse of phases_to_bytes.

    The .copy() call is required because frombuffer returns a read-only view
    backed by the bytes object; callers expect a mutable array.
    """
    _require_numpy()
    return np.frombuffer(data, dtype=np.float64).copy()


def snr_estimate(dim: int, n_items: int) -> float:
    """Signal-to-noise ratio estimate for holographic storage.

    SNR = sqrt(dim / n_items) when n_items > 0, else inf.

    The SNR falls below 2.0 when n_items > dim / 4, meaning retrieval
    errors become likely. Logs a warning when this threshold is crossed.
    """
    _require_numpy()

    if n_items <= 0:
        return float("inf")

    snr = math.sqrt(dim / n_items)

    if snr < 2.0:
        logger.warning(
            "HRR storage near capacity: SNR=%.2f (dim=%d, n_items=%d). "
            "Retrieval accuracy may degrade. Consider increasing dim or reducing stored items.",
            snr,
            dim,
            n_items,
        )

    return snr
