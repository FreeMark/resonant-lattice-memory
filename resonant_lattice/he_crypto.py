"""he_crypto.py — CKKS engine for the homomorphic blind store (Tier 1, E2).

Blind vector recall: embeddings are stored as CKKS ciphertext; the (untrusted)
store side computes cosine similarity homomorphically — `EvalInnerProduct` over
L2-normalized vectors — and NEVER decrypts. Only the client side, holding the
secret key (unwrapped from the keystore under the master passphrase), decrypts the
resulting scalar scores to rank them.

Design (validated against openfhe 1.5.1 on the target node):
  * `Serialize(obj, BINARY)` -> ``bytes`` (store directly as a SQLite BLOB);
    `Deserialize<T>String(blob, BINARY)` -> obj.
  * The eval-key store (mult + sum/rotation keys) is a GLOBAL/static map, so
    key *generation* happens once at setup (a one-shot process); runtime processes
    only *deserialize* keys. Generating and deserializing eval keys in the SAME
    process collides on the keyTag — hence the setup/runtime split below.
  * Cosine = dot product of unit vectors. Depth = 1 multiply + a rotation-sum, so
    no bootstrapping; small parameters; embarrassingly parallel across facts.

Trust boundary (enforced structurally):
  * EVAL side (store): public + eval keys only -> can encrypt and score, CANNOT
    decrypt (``_sk is None``).
  * CLIENT side (agent): additionally holds the secret key -> can decrypt scores.

Heavy dep (``openfhe``) is imported lazily-guarded: importing this module never
fails; the functions that need OpenFHE raise a clear error if it is absent, so
non-blind deployments keep the "no new heavy deps" guarantee.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import openfhe as _ofhe
    _OPENFHE_AVAILABLE = True
except Exception as e:  # pragma: no cover - exercised only without openfhe
    logger.debug("openfhe not available: %s. Blind store disabled until installed.", e)
    _ofhe = None
    _OPENFHE_AVAILABLE = False

try:
    import numpy as _np
    _NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover
    _np = None
    _NUMPY_AVAILABLE = False


# ── Versioned CKKS parameters (bump version, never edit in place) ───────────────
HE_PARAMS_VERSION = 1
_MULT_DEPTH = 1          # one elementwise multiply; the rotation-sum adds no depth
_SCALE_BITS = 50         # CKKS scaling modulus size
_DEFAULT_DIM = 768       # nomic-embed-text dimension

# ── E3 scheme-switching (homomorphic argmax) parameters ─────────────────────────
# Depth budget for CKKS<->TFHE argmin/argmax: the OpenFHE scheme-switching recipe needs
# 9 (CKKS) + 3 (switch) + 1, plus ceil(log2(numValues)) for the argmin rounds, plus the
# 2 leveled mults we spend packing per-fact inner products into one score vector.
_ARGMAX_BASE_DEPTH = 9 + 3 + 1
_ARGMAX_PACK_DEPTH = 2          # EvalInnerProduct result -> mask+scale into slot i
_FHEW_LOGQ = 25                 # FHEW ciphertext modulus size for the comparison
# Scores must sit inside the FHEW comparison's safe range or large values wrap and the
# argmax is wrong (validated on the node: cosines in [-1,1] scaled by 0.5 rank correctly).
_TOPK_SAFE_SCALE = 0.5

# ── Pure-CKKS comparison argmax (E3 serializable alternative) parameters ─────────
# Blind argmax WITHOUT FHEW scheme switching: approximate max / |·| as Chebyshev
# polynomials (Cheon et al., HE comparison) so the circuit uses ONLY CKKS mult+rotation
# keys, which DO serialize -> the eval-only store can run it across a process split
# (the FHEW BlindArgmax canNOT — its switching keys do not serialize; see 0a / its
# docstring). Proof params (HEStd_NotSet, leveled/no-bootstrap); production = E3 §3b.
_CMP_DEPTH = 32          # leveled depth for the comparison circuit (no bootstrapping)
_CMP_RING = 1 << 16      # ring dim for HEStd_NotSet fast-PoC ONLY (real levels auto-select)
# E3 §3b: production security level. Node-measured (2026-06-19) — HEStd_128_classic
# auto-selects ring 2^17 for depth 32; ~31 s/argmax at N=8 (one-time setup ~3.7 s) vs ~14 s
# at HEStd_NotSet/ring2^16. "Cycles-not-seconds" absorbs it. Pass "HEStd_NotSet" for fast tests.
_CMP_SECURITY_DEFAULT = "HEStd_128_classic"
_CMP_ABS_DEGREE = 59     # Chebyshev degree for |a-b| in max(a,b)=½(a+b)+½|a-b|
_CMP_ONEHOT_DEGREE = 27  # Chebyshev degree for the exp() one-hot indicator
_CMP_ONEHOT_ALPHA = 15.0 # exp sharpness: onehot_i ≈ exp(α·(s_i − max))
_CMP_PAD = -2.0          # sentinel for unused (non-power-of-two) slots: below the cosine range

# ── E5 blind dream-cycle maintenance (encrypted resonance) parameters ────────────
# Homomorphic decay (scalar mult, depth 1) + threshold comparison (promotion/eviction) via
# the same Chebyshev sign-approx as the argmax. Resonance is scaled to ~[0,1] client-side, so
# the comparison sits in the bounded interval [-_MAINT_RANGE, _MAINT_RANGE]. Node-measured
# (2026-06-19): decay ~4ms, one comparison ~3.5s/batch at HEStd_128_classic.
_MAINT_DEPTH = 14        # decay(1) + one Chebyshev step (~deg/2 levels) with headroom
_MAINT_STEP_DEGREE = 119 # Chebyshev degree for step(x): sharp enough outside the transition band
_MAINT_RANGE = 1.0       # comparison interval [-1, 1] (resonance scaled to ~[0,1])
_MAINT_NOTSET_RING = 1 << 15  # ring for the HEStd_NotSet fast-PoC path (real levels auto-select)
# DECAY-ONLY light depth (provider blind-maintenance path; user sign-off 2026-06-19). The
# provider dream cycle does blind DECAY + CLIENT-assisted settle (E5 5b) — NOT the homomorphic
# ge_threshold (Chebyshev step, needs the deep _MAINT_DEPTH ctx). Node-spike (2026-06-19): an
# IN-PLACE compounding decay survives exactly `depth` cycles (depth-1 ⇒ 1 decay before the ct
# exhausts), but DECAY-FROM-ORIGIN (one EvalMult of the preserved original by factor**elapsed,
# using the PUBLIC logical cycle clock) runs UNBOUNDED at depth 1. So the blind-maint keyset is
# generated at depth 1 (~0.8MB mult key, vs ~63MB at depth 14) and the maintainer decays from
# origin — lightest keystore AND unlimited autonomous cycles (the untrusted-store north star).
# Bump only if store-side autonomous ge_threshold flagging is later wired (would need depth ~14).
# ⚠️ The decay-FROM-ORIGIN maintainer this depth-1 keyset assumes is NOT yet implemented. The only
# decay path today — retrieval.BlindMaintainer.decay_all — does IN-PLACE compounding and needs the
# deep _MAINT_DEPTH; read its DEPTH CONTRACT docstring before wiring this keyset to any decay path.
_MAINT_BLIND_DEPTH = 1

# The serialized-key blob keys persisted in the keystore (secret is wrapped separately).
KEY_BLOBS = ("ctx", "pub", "em", "ea")


class HEUnavailableError(RuntimeError):
    """Raised when a blind-store operation is requested but openfhe/numpy is missing."""


class SecretRequiredError(RuntimeError):
    """Raised when decryption is attempted on an eval-only (store-side) context."""


def he_available() -> bool:
    return _OPENFHE_AVAILABLE and _NUMPY_AVAILABLE


def _require() -> None:
    if not _OPENFHE_AVAILABLE:
        raise HEUnavailableError(
            "openfhe is required for the blind store. Install it "
            "(`pip install openfhe`) or use encryption_mode in (none, at_rest)."
        )
    if not _NUMPY_AVAILABLE:
        raise HEUnavailableError("numpy is required for blind-store vector encoding.")


def _next_pow2_at_least(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _normalize_pad(vec: List[float], dim: int, batch: int) -> List[float]:
    """L2-normalize to unit length (so dot product == cosine), then zero-pad to batch.

    Padding with zeros is inert for the inner product, so summing all `batch` slots
    yields exactly the dot product over the `dim` real components.
    """
    v = _np.asarray(vec, dtype=float).ravel()
    if v.shape[0] != dim:
        # Be forgiving: truncate/pad to the configured dim before normalizing.
        v = v[:dim] if v.shape[0] > dim else _np.pad(v, (0, dim - v.shape[0]))
    norm = float(_np.linalg.norm(v))
    if norm > 0:
        v = v / norm
    if batch > v.shape[0]:
        v = _np.pad(v, (0, batch - v.shape[0]))
    return v[:batch].tolist()


def _build_context(batch: int):
    p = _ofhe.CCParamsCKKSRNS()
    p.SetMultiplicativeDepth(_MULT_DEPTH)
    p.SetScalingModSize(_SCALE_BITS)
    p.SetBatchSize(batch)
    cc = _ofhe.GenCryptoContext(p)
    for feat in (_ofhe.PKE, _ofhe.KEYSWITCH, _ofhe.LEVELEDSHE, _ofhe.ADVANCEDSHE):
        cc.Enable(feat)
    return cc


class BlindCrypto:
    """CKKS engine bound to one keypair. Construct via the classmethods, not directly.

    Holds the OpenFHE CryptoContext (with eval keys loaded into the global store) and
    the public key for encryption. ``_sk`` is None on the store side.
    """

    def __init__(self, cc, dim: int, batch: int):
        self._cc = cc
        self._dim = dim
        self._batch = batch
        self._pub = None
        self._sk = None

    # ── construction ────────────────────────────────────────────────────────────
    @classmethod
    def generate(cls, dim: int = _DEFAULT_DIM) -> Tuple["BlindCrypto", Dict[str, bytes], bytes]:
        """SETUP-ONLY (one-shot process): fresh keygen.

        Returns ``(client_instance, key_blobs, secret_blob)`` where:
          * ``key_blobs`` = {ctx, pub, em, ea} serialized bytes — NON-secret, persisted
            in the keystore as-is.
          * ``secret_blob`` = the serialized secret key bytes — the caller MUST wrap it
            (AES-GCM under a master-derived key) before persisting.
        Do not call other instances' deserialize paths in this same process (the eval
        key store is global/static and would collide).
        """
        _require()
        batch = _next_pow2_at_least(dim)
        cc = _build_context(batch)
        kp = cc.KeyGen()
        cc.EvalMultKeyGen(kp.secretKey)
        cc.EvalSumKeyGen(kp.secretKey)
        key_blobs = {
            "ctx": _ofhe.Serialize(cc, _ofhe.BINARY),
            "pub": _ofhe.Serialize(kp.publicKey, _ofhe.BINARY),
            "em": _ofhe.SerializeEvalMultKeyString(_ofhe.BINARY, ""),
            "ea": _ofhe.SerializeEvalAutomorphismKeyString(_ofhe.BINARY, ""),
        }
        secret_blob = _ofhe.Serialize(kp.secretKey, _ofhe.BINARY)
        inst = cls(cc, dim, batch)
        inst._pub = kp.publicKey
        inst._sk = kp.secretKey
        meta = {"version": HE_PARAMS_VERSION, "dim": dim, "batch": batch}
        inst._meta = meta
        return inst, key_blobs, secret_blob

    @classmethod
    def load_eval(cls, key_blobs: Dict[str, bytes], dim: int, batch: int) -> "BlindCrypto":
        """STORE side (fresh process): reconstruct context + eval keys + public key.

        No secret key — this instance can encrypt and score but never decrypt.
        """
        _require()
        cc = _ofhe.DeserializeCryptoContextString(key_blobs["ctx"], _ofhe.BINARY)
        _ofhe.DeserializeEvalMultKeyString(key_blobs["em"], _ofhe.BINARY)
        _ofhe.DeserializeEvalAutomorphismKeyString(key_blobs["ea"], _ofhe.BINARY)
        inst = cls(cc, dim, batch)
        inst._pub = _ofhe.DeserializePublicKeyString(key_blobs["pub"], _ofhe.BINARY)
        inst._sk = None
        return inst

    @classmethod
    def load_client(cls, key_blobs: Dict[str, bytes], dim: int, batch: int,
                    secret_blob: bytes) -> "BlindCrypto":
        """CLIENT side (fresh process): eval reconstruction + the unwrapped secret key."""
        inst = cls.load_eval(key_blobs, dim, batch)
        inst._sk = _ofhe.DeserializePrivateKeyString(secret_blob, _ofhe.BINARY)
        return inst

    # ── properties ───────────────────────────────────────────────────────────────
    @property
    def can_decrypt(self) -> bool:
        return self._sk is not None

    @property
    def batch(self) -> int:
        return self._batch

    # ── operations ────────────────────────────────────────────────────────────────
    def encrypt_unit_vector(self, vec: List[float]) -> bytes:
        """Client/eval side: normalize+pad, encrypt under the public key, serialize."""
        _require()
        packed = self._cc.MakeCKKSPackedPlaintext(_normalize_pad(vec, self._dim, self._batch))
        ct = self._cc.Encrypt(self._pub, packed)
        return _ofhe.Serialize(ct, _ofhe.BINARY)

    def _deser_ct(self, blob: bytes):
        return _ofhe.DeserializeCiphertextString(blob, _ofhe.BINARY)

    def cosine_score(self, query_ct_blob: bytes, stored_ct_blob: bytes):
        """STORE side: homomorphic cosine (inner product of unit vectors).

        Returns the encrypted scalar Ciphertext OBJECT (kept in-memory; serializing
        each score would move ~hundreds of KB per fact between store and client — we
        only serialize when the store is a genuinely separate process). Uses ONLY
        public + eval keys; never touches a secret key.
        """
        _require()
        q = self._deser_ct(query_ct_blob)
        s = self._deser_ct(stored_ct_blob)
        return self._cc.EvalInnerProduct(q, s, self._batch)

    def serialize_score(self, score_ct) -> bytes:
        """Serialize an encrypted score Ciphertext (store -> client transfer)."""
        _require()
        return _ofhe.Serialize(score_ct, _ofhe.BINARY)

    def decrypt_score(self, score_ct) -> float:
        """CLIENT side: decrypt an encrypted scalar score to a float cosine value."""
        _require()
        if self._sk is None:
            raise SecretRequiredError("this is an eval-only (store-side) context; cannot decrypt")
        # Accept either a live Ciphertext object or a serialized blob.
        ct = self._deser_ct(score_ct) if isinstance(score_ct, (bytes, bytearray)) else score_ct
        pt = self._cc.Decrypt(ct, self._sk)
        pt.SetLength(1)
        return float(pt.GetRealPackedValue()[0])


class BlindArgmax:
    """E3: homomorphic blind argmax over encrypted embeddings via CKKS<->TFHE scheme
    switching. The store computes WHICH stored vector is most similar to the query — a
    one-hot indicator — using public + eval/switching keys ONLY (``EvalMaxSchemeSwitching``
    takes the public key, never the secret). So neither the per-fact scores nor their full
    ranking leave the store; only the single winning index does. This removes the
    count/score leak of E2's client-side ranking (roadmap §7.3 option b).

    Validated end-to-end on the node 2026-06-18 (N=8, dim=16): packed encrypted inner
    products -> ``EvalMaxSchemeSwitching`` -> one-hot == plaintext argmax, ~2.7 s.

    Scope (this engine): blind **top-1** argmax in a single scheme-switching context,
    **co-located only**. 0a finding (node-proven 2026-06-19): the FHEW / scheme-switching
    keys do NOT serialize in openfhe-python 1.5.1 (no ``LWEPrivateKey``/``BinFHEContext``
    ``Serialize`` overload; a deserialized eval-only context segfaults in
    ``EvalMaxSchemeSwitching``) and 1.5.1 is the latest wheel — so this engine canNOT run
    on an untrusted store via serialized keys. For the splittable blind-argmax path use
    ``BlindArgmaxCKKS`` (pure-CKKS comparison, all keys serialize). This class stays as the
    faster co-located reference. Other remaining E3 plumbing: top-k>1 (each argmax exhausts
    the depth budget, so subsequent rounds need inter-round bootstrapping), arbitrary
    (non-power-of-two) fact counts via negative-infinity padding, and a production security
    level (this proof uses ``HEStd_NotSet`` + ``TOY`` FHEW for speed).
    """

    def __init__(self, cc, dim: int, num_slots: int, num_values: int):
        self._cc = cc
        self._dim = dim
        self._slots = num_slots
        self._n = num_values
        self._pub = None
        self._sk = None

    @classmethod
    def generate(cls, dim: int = _DEFAULT_DIM, num_facts: int = 8) -> "BlindArgmax":
        """SETUP/CLIENT (one process): build the scheme-switching context + all keys.

        ``num_facts`` is rounded up to a power of two (the argmin slot count). The
        instance holds the secret (client side); the store side would deserialize the
        eval/switching keys only (serialized split is remaining E3 plumbing)."""
        _require()
        n = _next_pow2_at_least(max(2, num_facts))
        slots = _next_pow2_at_least(max(dim, n))
        depth = _ARGMAX_PACK_DEPTH + _ARGMAX_BASE_DEPTH + (n.bit_length() - 1)  # +ceil(log2 n)
        p = _ofhe.CCParamsCKKSRNS()
        p.SetMultiplicativeDepth(depth)
        p.SetScalingModSize(_SCALE_BITS)
        p.SetFirstModSize(60)
        p.SetScalingTechnique(_ofhe.FIXEDMANUAL)
        p.SetSecurityLevel(_ofhe.HEStd_NotSet)   # proof-of-concept; production = a real level
        p.SetRingDim(8192)
        p.SetBatchSize(slots)
        p.SetKeySwitchTechnique(_ofhe.HYBRID)
        cc = _ofhe.GenCryptoContext(p)
        for feat in (_ofhe.PKE, _ofhe.KEYSWITCH, _ofhe.LEVELEDSHE, _ofhe.ADVANCEDSHE,
                     _ofhe.SCHEMESWITCH):
            cc.Enable(feat)
        kp = cc.KeyGen()
        cc.EvalMultKeyGen(kp.secretKey)
        cc.EvalSumKeyGen(kp.secretKey)
        cc.EvalRotateKeyGen(kp.secretKey, [-i for i in range(1, n)])   # pack score -> slot i
        sp = _ofhe.SchSwchParams()
        sp.SetSecurityLevelCKKS(_ofhe.HEStd_NotSet)
        sp.SetSecurityLevelFHEW(_ofhe.TOY)
        sp.SetCtxtModSizeFHEWLargePrec(_FHEW_LOGQ)
        sp.SetNumSlotsCKKS(slots)
        sp.SetNumValues(n)
        sp.SetComputeArgmin(True)
        sp.SetOneHotEncoding(True)
        lwesk = cc.EvalSchemeSwitchingSetup(sp)
        cc.EvalSchemeSwitchingKeyGen(kp, lwesk)
        cc.EvalCKKStoFHEWPrecompute(_TOPK_SAFE_SCALE)
        inst = cls(cc, dim, slots, n)
        inst._pub = kp.publicKey
        inst._sk = kp.secretKey
        return inst

    @property
    def num_values(self) -> int:
        return self._n

    def encrypt_vector(self, vec: List[float]):
        """Encrypt an L2-normalized, zero-padded embedding under the public key."""
        _require()
        packed = self._cc.MakeCKKSPackedPlaintext(
            _normalize_pad(vec, self._dim, self._slots), 1, 0, None, self._slots)
        return self._cc.Encrypt(self._pub, packed)

    def argmax(self, query_ct, fact_cts: List) -> "object":
        """STORE side: one-hot argmax of cosine(query, fact_i) over the fact ciphertexts.

        Packs each fact's inner product into slot i of one score vector, scales into the
        FHEW-safe range, then scheme-switches to the homomorphic argmax. Uses ONLY public
        + eval/switching keys — never the secret. Requires exactly ``num_values`` facts
        (caller pads to the power-of-two count; richer padding is remaining plumbing)."""
        _require()
        if len(fact_cts) != self._n:
            raise ValueError(
                f"BlindArgmax expects exactly num_values={self._n} fact ciphertexts "
                f"(pad to the power-of-two count), got {len(fact_cts)}")
        mask_scale = self._cc.MakeCKKSPackedPlaintext(
            [_TOPK_SAFE_SCALE] + [0.0] * (self._slots - 1), 1, 0, None, self._slots)
        packed = None
        for i, fct in enumerate(fact_cts):
            s = self._cc.EvalInnerProduct(query_ct, fct, self._slots)   # cosine -> slot 0
            s0 = self._cc.EvalMult(s, mask_scale)                       # isolate slot 0 + scale
            si = s0 if i == 0 else self._cc.EvalRotate(s0, -i)          # move to slot i
            packed = si if packed is None else self._cc.EvalAdd(packed, si)
        res = self._cc.EvalMaxSchemeSwitching(packed, self._pub, self._n, self._slots)
        return res[1]   # one-hot argmax ciphertext (res[0] is the max value)

    def decrypt_onehot(self, onehot_ct) -> List[float]:
        """CLIENT side: decrypt the one-hot argmax to a list of num_values floats."""
        _require()
        if self._sk is None:
            raise SecretRequiredError("this is an eval-only (store-side) context; cannot decrypt")
        pt = self._cc.Decrypt(onehot_ct, self._sk)
        pt.SetLength(self._n)
        return list(pt.GetRealPackedValue())


class BlindPRE:
    """E6: proxy re-encryption runtime path — the three-key "use but can't read" model
    (roadmap §4.1 / §7.1). Proven on the node 2026-06-18.

    The store holds a query RESULT ciphertext under the STORAGE key. With a one-time
    re-encryption key ``rk_storage->agent`` (generated at setup, while the master is
    present), the store re-encrypts that result to the AGENT use-key — so the agent
    decrypts ONLY results the store re-encrypts for a query it ran. Applied to the raw DB,
    the agent key decrypts NOTHING (proven: OpenFHE rejects the decode). The master/user
    re-derives everything for god-mode. This property is impossible in Tier 0 — a runtime
    key that can use the data can read all of it.

    The honest seam (tension #2): a hijacked agent can still ask the store to re-encrypt
    results query-by-query; that is bounded by POLICY (``blind_policy``: scope/rate caps +
    a re-encryption audit log), not by the math.

    Scope (this engine): the proven primitive in one process. Serialized key blobs for a
    real store/agent split and unifying this PRE context with the E2 scoring context (so
    actual cosine-score cts are what gets re-encrypted) are remaining E6 plumbing."""

    def __init__(self, cc, batch: int):
        self._cc = cc
        self._batch = batch

    @classmethod
    def generate(cls, batch: int = 8) -> "BlindPRE":
        """Build a PRE-enabled CKKS context (one process; serialized split deferred)."""
        _require()
        batch = _next_pow2_at_least(max(2, batch))
        p = _ofhe.CCParamsCKKSRNS()
        p.SetMultiplicativeDepth(1)
        p.SetScalingModSize(_SCALE_BITS)
        p.SetBatchSize(batch)
        p.SetPREMode(_ofhe.INDCPA)
        cc = _ofhe.GenCryptoContext(p)
        for feat in (_ofhe.PKE, _ofhe.KEYSWITCH, _ofhe.LEVELEDSHE, _ofhe.PRE):
            cc.Enable(feat)
        return cls(cc, batch)

    def keygen(self):
        """A fresh keypair — used for both the storage (master) key and the agent use-key."""
        _require()
        return self._cc.KeyGen()

    def rekey(self, from_secret, to_public):
        """rk_from->to : the re-encryption key (needs the FROM secret, generated at setup)."""
        _require()
        return self._cc.ReKeyGen(from_secret, to_public)

    def encrypt(self, values: List[float], public_key):
        _require()
        v = list(values)[:self._batch]
        v = v + [0.0] * (self._batch - len(v))
        return self._cc.Encrypt(public_key, self._cc.MakeCKKSPackedPlaintext(v, 1, 0, None, self._batch))

    def reencrypt(self, ct, rk):
        """STORE side: re-encrypt a result ciphertext to the agent use-key under ``rk``."""
        _require()
        return self._cc.ReEncrypt(ct, rk)

    def decrypt(self, ct, secret_key, length: int = 1) -> List[float]:
        """Decrypt with whichever secret key (agent for re-encrypted results, master for
        god-mode). Raises if the key can't read the ciphertext (e.g. the agent key on the
        raw DB) — that rejection IS the security property."""
        _require()
        pt = self._cc.Decrypt(ct, secret_key)
        pt.SetLength(length)
        return list(pt.GetRealPackedValue())


class ThresholdAudit:
    """E6: threshold / multiparty user-audit path (roadmap §7.1). Proven on the node
    2026-06-18 (2-of-2). Distributed decryption needs ALL shares fused — no single party
    (store or agent) decrypts alone; the user, holding the shares, reconstructs anything
    (god-mode audit / export / rotation). This is the path RESERVED for the user, distinct
    from the runtime PRE path (which is for the agent). Generalizes to (t,n); the proven
    core is 2-of-2. Serialized share distribution is remaining plumbing."""

    def __init__(self, cc, batch: int):
        self._cc = cc
        self._batch = batch

    @classmethod
    def generate(cls, batch: int = 8) -> "ThresholdAudit":
        _require()
        batch = _next_pow2_at_least(max(2, batch))
        p = _ofhe.CCParamsCKKSRNS()
        p.SetMultiplicativeDepth(1)
        p.SetScalingModSize(_SCALE_BITS)
        p.SetBatchSize(batch)
        cc = _ofhe.GenCryptoContext(p)
        for feat in (_ofhe.PKE, _ofhe.KEYSWITCH, _ofhe.LEVELEDSHE, _ofhe.MULTIPARTY):
            cc.Enable(feat)
        return cls(cc, batch)

    def first_party(self):
        """Party 1's keypair (the lead)."""
        _require()
        return self._cc.KeyGen()

    def join(self, prev_public):
        """Party N joins, chaining onto the previous public key -> the joint public key."""
        _require()
        return self._cc.MultipartyKeyGen(prev_public)

    def encrypt(self, values: List[float], joint_public):
        _require()
        v = list(values)[:self._batch]
        v = v + [0.0] * (self._batch - len(v))
        return self._cc.Encrypt(joint_public, self._cc.MakeCKKSPackedPlaintext(v, 1, 0, None, self._batch))

    def partial_lead(self, ct, secret_key):
        _require()
        return self._cc.MultipartyDecryptLead([ct], secret_key)[0]

    def partial_main(self, ct, secret_key):
        _require()
        return self._cc.MultipartyDecryptMain([ct], secret_key)[0]

    def fuse(self, partials: List, length: int = 1) -> List[float]:
        """Fuse decryption shares. ALL shares are required — fusing a subset raises (the
        rejection that proves no single party can decrypt)."""
        _require()
        pt = self._cc.MultipartyDecryptFusion(list(partials))
        pt.SetLength(length)
        return list(pt.GetRealPackedValue())


class BlindRecallPRE:
    """E2+E6 unified: the deployable blind-RECALL runtime in ONE serializable CKKS context
    (roadmap 0a decision — purpose-built contexts). The untrusted store homomorphically
    scores cosine similarity (E2) AND re-encrypts the encrypted score to the agent's
    use-key via PRE (E6 §4.1 / item 6b) — all with public + eval keys + the rekey, NEVER a
    secret. The agent decrypts ONLY results the store re-encrypts for it (its use-key on the
    RAW store ct is rejected by OpenFHE — the "use but can't read" property); the user
    re-derives the master for god-mode. Because the score that gets re-encrypted is the
    actual cosine ct, this folds E6's "unify PRE with the scoring context" (6b).

    Node-proven 2026-06-19 across a 3-process serialized split: an eval-only store scores +
    re-encrypts with NO secret; the agent decrypts the result (err ~3e-5, argmax correct);
    the agent key on the raw store ct is REJECTED; the master reads it (god-mode). Every key
    here serializes (``ctx``/``pub``/``em``/``ea`` like E2, plus ``rk`` via the generic
    ``EvalKey`` serializer), unlike the FHEW path.

    Three roles via the load_* classmethods (each a fresh process; never keygen+deserialize
    in the same process — the eval-key store is global/static):
      * STORE  (``load_eval``)   — ctx + storage pub + eval keys + rekey; scores + reencrypts; no secret.
      * AGENT  (``load_client``) — store role + the agent use-key; decrypts re-encrypted results only.
      * USER   (``load_user``)   — store role + the master secret; god-mode decrypt of any ct.
    """

    _BLOBS = ("ctx", "pub", "em", "ea", "rk")

    def __init__(self, cc, dim: int, batch: int):
        self._cc = cc
        self._dim = dim
        self._batch = batch
        self._pub = None      # storage public key (encrypt)
        self._rk = None       # rk_storage->agent (re-encrypt)
        self._sk = None       # agent use-key (load_client) or master secret (load_user/generate)

    # ── construction ────────────────────────────────────────────────────────────
    @classmethod
    def generate(cls, dim: int = _DEFAULT_DIM):
        """SETUP-ONLY (one-shot process): build the recall+PRE context and all keys.

        Returns ``(user_instance, key_blobs, secret_blobs)`` where:
          * ``key_blobs`` = {ctx, pub, em, ea, rk} serialized bytes — NON-secret, persist as-is.
          * ``secret_blobs`` = {"master": bytes, "agent": bytes} — the caller MUST wrap each
            (``crypto_keys.wrap_he_secret`` under a master-derived key) before persisting.
        The returned instance carries the master secret (the user/setup view)."""
        _require()
        batch = _next_pow2_at_least(dim)
        p = _ofhe.CCParamsCKKSRNS()
        p.SetMultiplicativeDepth(_MULT_DEPTH + 1)   # cosine = 1 mult; +1 headroom
        p.SetScalingModSize(_SCALE_BITS)
        p.SetBatchSize(batch)
        p.SetPREMode(_ofhe.INDCPA)                  # default security level = HEStd_128_classic
        cc = _ofhe.GenCryptoContext(p)
        for feat in (_ofhe.PKE, _ofhe.KEYSWITCH, _ofhe.LEVELEDSHE, _ofhe.ADVANCEDSHE, _ofhe.PRE):
            cc.Enable(feat)
        kpS = cc.KeyGen()                           # storage / master keypair
        kpA = cc.KeyGen()                           # agent use-key keypair
        cc.EvalMultKeyGen(kpS.secretKey)
        cc.EvalSumKeyGen(kpS.secretKey)             # for EvalInnerProduct under the storage key
        rk = cc.ReKeyGen(kpS.secretKey, kpA.publicKey)   # rk_storage->agent (needs storage secret)
        # Serialize the eval keys BY THIS CONTEXT'S KEY TAG (not ""), so MULTIPLE keysets can be
        # generated in ONE setup process (recall@embed-dim + HRR@2·hrr-dim — Option A multi-keyset
        # keystore) with ISOLATED blobs: the "" selector grabs the whole GLOBAL eval-key store,
        # which by then holds every prior context's keys. All eval keys here register under the
        # storage keypair's tag (EvalMult/EvalSumKeyGen(kpS.secretKey); kpA is only ReKeyGen'd).
        # Node-proven 2026-06-19: by-tag yields clean per-context blobs AND live instances keep
        # working (no clearing). Backward-compatible — for a single-context process by-tag == "".
        _tag = kpS.secretKey.GetKeyTag()
        key_blobs = {
            "ctx": _ofhe.Serialize(cc, _ofhe.BINARY),
            "pub": _ofhe.Serialize(kpS.publicKey, _ofhe.BINARY),
            "em": _ofhe.SerializeEvalMultKeyString(_ofhe.BINARY, _tag),
            "ea": _ofhe.SerializeEvalAutomorphismKeyString(_ofhe.BINARY, _tag),
            "rk": _ofhe.Serialize(rk, _ofhe.BINARY),
        }
        secret_blobs = {
            "master": _ofhe.Serialize(kpS.secretKey, _ofhe.BINARY),
            "agent": _ofhe.Serialize(kpA.secretKey, _ofhe.BINARY),
        }
        inst = cls(cc, dim, batch)
        inst._pub = kpS.publicKey
        inst._rk = rk
        inst._sk = kpS.secretKey
        return inst, key_blobs, secret_blobs

    @classmethod
    def load_eval(cls, key_blobs: Dict[str, bytes], dim: int, batch: int) -> "BlindRecallPRE":
        """STORE side (fresh process): ctx + eval keys + storage pub + rekey. No secret —
        can encrypt, score, and re-encrypt, but never decrypt."""
        _require()
        cc = _ofhe.DeserializeCryptoContextString(key_blobs["ctx"], _ofhe.BINARY)
        _ofhe.DeserializeEvalMultKeyString(key_blobs["em"], _ofhe.BINARY)
        _ofhe.DeserializeEvalAutomorphismKeyString(key_blobs["ea"], _ofhe.BINARY)
        inst = cls(cc, dim, batch)
        inst._pub = _ofhe.DeserializePublicKeyString(key_blobs["pub"], _ofhe.BINARY)
        inst._rk = _ofhe.DeserializeEvalKeyString(key_blobs["rk"], _ofhe.BINARY)
        inst._sk = None
        return inst

    @classmethod
    def load_client(cls, key_blobs: Dict[str, bytes], dim: int, batch: int,
                    agent_secret_blob: bytes) -> "BlindRecallPRE":
        """AGENT side: store role + the agent use-key (decrypts re-encrypted results only)."""
        inst = cls.load_eval(key_blobs, dim, batch)
        inst._sk = _ofhe.DeserializePrivateKeyString(agent_secret_blob, _ofhe.BINARY)
        return inst

    @classmethod
    def load_user(cls, key_blobs: Dict[str, bytes], dim: int, batch: int,
                  master_secret_blob: bytes) -> "BlindRecallPRE":
        """USER side: store role + the master secret (god-mode decrypt of any ciphertext)."""
        inst = cls.load_eval(key_blobs, dim, batch)
        inst._sk = _ofhe.DeserializePrivateKeyString(master_secret_blob, _ofhe.BINARY)
        return inst

    # ── properties ───────────────────────────────────────────────────────────────
    @property
    def can_decrypt(self) -> bool:
        return self._sk is not None

    @property
    def can_reencrypt(self) -> bool:
        return self._rk is not None

    @property
    def batch(self) -> int:
        return self._batch

    # ── operations ────────────────────────────────────────────────────────────────
    def encrypt_unit_vector(self, vec: List[float]) -> bytes:
        """Client/eval side: normalize+pad, encrypt under the storage public key, serialize.
        Duck-typed identical to ``BlindCrypto`` so ``BlindRetriever`` can use either."""
        _require()
        packed = self._cc.MakeCKKSPackedPlaintext(_normalize_pad(vec, self._dim, self._batch))
        return _ofhe.Serialize(self._cc.Encrypt(self._pub, packed), _ofhe.BINARY)

    def _deser_ct(self, blob):
        return _ofhe.DeserializeCiphertextString(blob, _ofhe.BINARY)

    def cosine_score(self, query_ct_blob: bytes, stored_ct_blob: bytes):
        """STORE side: homomorphic cosine (inner product of unit vectors) -> score Ciphertext
        OBJECT. Public + eval keys only; never a secret."""
        _require()
        q = self._deser_ct(query_ct_blob)
        s = self._deser_ct(stored_ct_blob)
        return self._cc.EvalInnerProduct(q, s, self._batch)

    def reencrypt_score(self, score_ct) -> bytes:
        """STORE side: PRE re-encrypt the score ct to the agent use-key, serialize for the
        store->agent transfer. Accepts a live Ciphertext or a serialized blob."""
        _require()
        if self._rk is None:
            raise SecretRequiredError("no rekey on this context; cannot re-encrypt")
        ct = self._deser_ct(score_ct) if isinstance(score_ct, (bytes, bytearray)) else score_ct
        return _ofhe.Serialize(self._cc.ReEncrypt(ct, self._rk), _ofhe.BINARY)

    def decrypt_score(self, score_ct) -> float:
        """AGENT (a re-encrypted result) or USER (any ct): decrypt the scalar cosine score.
        The agent's use-key on a RAW (non-re-encrypted) store ct is rejected by OpenFHE —
        that rejection IS the 'use but can't read' property."""
        _require()
        if self._sk is None:
            raise SecretRequiredError("this is an eval-only (store-side) context; cannot decrypt")
        ct = self._deser_ct(score_ct) if isinstance(score_ct, (bytes, bytearray)) else score_ct
        pt = self._cc.Decrypt(ct, self._sk)
        pt.SetLength(1)
        return float(pt.GetRealPackedValue()[0])


class BlindArgmaxCKKS:
    """E3 serializable alternative to the FHEW ``BlindArgmax``: blind argmax via PURE-CKKS
    polynomial comparison — no scheme switching. ``max(a,b)=½(a+b)+½·|a−b|`` with |·| a
    Chebyshev approximation (Cheon et al., HE comparison), a cyclic rotate-max-reduce to the
    global max broadcast across slots, then a steep ``exp()`` one-hot indicator. Uses ONLY
    CKKS mult + rotation keys, which SERIALIZE — so unlike the FHEW ``BlindArgmax`` (whose
    switching keys do not serialize and segfault on a deserialized eval-only store) this
    engine genuinely SPLITS: the untrusted store runs argmax with public + eval keys only
    and never decrypts. Node-proven 2026-06-19 across a 3-process serialized split
    (eval-only store one-hot == plaintext argmax).

    Cost/accuracy (honest): heavier per op than FHEW (deep leveled circuit, ~seconds) but
    splittable; approximate, so near-ties can flip (mitigate with a steeper indicator /
    higher degree / a client-side rank fallback). The proof params use ``HEStd_NotSet`` for
    speed; production security + bootstrapping (bootstrap keys also serialize, so the split
    holds) are E3 hardening (roadmap §3b). Operates on an encrypted SCORE vector (the N
    cosines packed in N slots); packing query/fact inner products into that vector in this
    same context is the recall-integration step (0c / E3 hardening — node-spike first)."""

    _BLOBS = ("ctx", "pub", "em", "ea")

    def __init__(self, cc, n: int, batch: int):
        self._cc = cc
        self._n = n
        self._batch = batch
        self._pub = None
        self._sk = None

    @classmethod
    def generate(cls, num_facts: int = 8, security: Optional[str] = None):
        """SETUP/CLIENT (one process): build the comparison context + mult/rotation keys.

        ``security`` is an OpenFHE level name (default ``HEStd_128_classic``, production); pass
        ``"HEStd_NotSet"`` for the fast PoC params. At a real level OpenFHE auto-selects the
        ring dim for depth ``_CMP_DEPTH`` (2^17 at 128-bit, ~31 s/argmax for N=8 — §3b); at
        ``HEStd_NotSet`` we fix the small ring ``_CMP_RING`` for speed. Returns
        ``(client_instance, key_blobs, secret_blob)`` mirroring ``BlindCrypto`` — key_blobs
        {ctx,pub,em,ea} are NON-secret; secret_blob is wrapped by the caller. The store side
        reconstructs via ``load_eval`` (no secret)."""
        _require()
        n = max(2, int(num_facts))
        batch = _next_pow2_at_least(n)
        shifts = [1 << i for i in range(batch.bit_length() - 1)]   # 1,2,4,...,batch/2
        sec_name = security or _CMP_SECURITY_DEFAULT
        sec = getattr(_ofhe, sec_name)
        p = _ofhe.CCParamsCKKSRNS()
        p.SetMultiplicativeDepth(_CMP_DEPTH)
        p.SetScalingModSize(_SCALE_BITS)
        p.SetFirstModSize(60)
        p.SetScalingTechnique(_ofhe.FLEXIBLEAUTO)
        p.SetSecurityLevel(sec)
        if sec_name == "HEStd_NotSet":
            p.SetRingDim(_CMP_RING)                 # PoC: no auto-selection at NotSet -> fix a small ring
        # else: let OpenFHE pick the ring to satisfy the security level + the depth budget
        p.SetBatchSize(batch)
        cc = _ofhe.GenCryptoContext(p)
        for feat in (_ofhe.PKE, _ofhe.KEYSWITCH, _ofhe.LEVELEDSHE, _ofhe.ADVANCEDSHE):
            cc.Enable(feat)
        kp = cc.KeyGen()
        cc.EvalMultKeyGen(kp.secretKey)
        cc.EvalRotateKeyGen(kp.secretKey, shifts)
        key_blobs = {
            "ctx": _ofhe.Serialize(cc, _ofhe.BINARY),
            "pub": _ofhe.Serialize(kp.publicKey, _ofhe.BINARY),
            "em": _ofhe.SerializeEvalMultKeyString(_ofhe.BINARY, ""),
            "ea": _ofhe.SerializeEvalAutomorphismKeyString(_ofhe.BINARY, ""),
        }
        secret_blob = _ofhe.Serialize(kp.secretKey, _ofhe.BINARY)
        inst = cls(cc, n, batch)
        inst._pub = kp.publicKey
        inst._sk = kp.secretKey
        return inst, key_blobs, secret_blob

    @classmethod
    def load_eval(cls, key_blobs: Dict[str, bytes], n: int, batch: int) -> "BlindArgmaxCKKS":
        """STORE side (fresh process): ctx + mult + rotation keys + pub. No secret."""
        _require()
        cc = _ofhe.DeserializeCryptoContextString(key_blobs["ctx"], _ofhe.BINARY)
        _ofhe.DeserializeEvalMultKeyString(key_blobs["em"], _ofhe.BINARY)
        _ofhe.DeserializeEvalAutomorphismKeyString(key_blobs["ea"], _ofhe.BINARY)
        inst = cls(cc, n, batch)
        inst._pub = _ofhe.DeserializePublicKeyString(key_blobs["pub"], _ofhe.BINARY)
        inst._sk = None
        return inst

    @classmethod
    def load_client(cls, key_blobs: Dict[str, bytes], n: int, batch: int,
                    secret_blob: bytes) -> "BlindArgmaxCKKS":
        """CLIENT side: eval reconstruction + the unwrapped secret key (decrypts the one-hot)."""
        inst = cls.load_eval(key_blobs, n, batch)
        inst._sk = _ofhe.DeserializePrivateKeyString(secret_blob, _ofhe.BINARY)
        return inst

    @property
    def num_values(self) -> int:
        return self._n

    @property
    def batch(self) -> int:
        return self._batch

    def _interval(self) -> Tuple[float, float]:
        """Score value range. Pow2 N (no padding) keeps the tight [-1,1] cosine range proven
        in the spike; non-pow2 widens to include the ``_CMP_PAD`` sentinel."""
        lo = _CMP_PAD if self._batch > self._n else -1.0
        return lo, 1.0

    def encrypt_scores(self, scores: List[float]):
        """Encrypt the N cosine scores into a length-``batch`` plaintext, padding unused
        (non-power-of-two) slots with a below-range sentinel so they never win the max."""
        _require()
        v = list(scores)[:self._n]
        v = v + [_CMP_PAD] * (self._batch - len(v))
        return self._cc.Encrypt(self._pub, self._cc.MakeCKKSPackedPlaintext(v, 1, 0, None, self._batch))

    def argmax(self, score_ct):
        """STORE side: one-hot argmax over the encrypted score vector — public + eval keys
        ONLY, never the secret. Cyclic Chebyshev-|·| max-reduce to the global max, then an
        ``exp()`` one-hot indicator on (score − max)."""
        _require()
        lo, hi = self._interval()
        absf = lambda x: abs(x)
        a_lo, a_hi = lo - hi, hi - lo            # range of (a − b)
        def hmax(a, b):
            ab = self._cc.EvalChebyshevFunction(absf, self._cc.EvalSub(a, b), a_lo, a_hi, _CMP_ABS_DEGREE)
            return self._cc.EvalAdd(self._cc.EvalMult(self._cc.EvalAdd(a, b), 0.5),
                                    self._cc.EvalMult(ab, 0.5))
        x = score_ct
        shift = 1
        while shift < self._batch:
            x = hmax(x, self._cc.EvalRotate(x, shift))
            shift <<= 1
        d = self._cc.EvalSub(score_ct, x)        # 0 at the argmax, < 0 elsewhere
        alpha = _CMP_ONEHOT_ALPHA
        expf = lambda t: math.exp(alpha * t)
        return self._cc.EvalChebyshevFunction(expf, d, lo - hi, 0.1, _CMP_ONEHOT_DEGREE)

    def decrypt_onehot(self, onehot_ct) -> List[float]:
        """CLIENT side: decrypt the one-hot argmax to a list of ``num_values`` floats."""
        _require()
        if self._sk is None:
            raise SecretRequiredError("this is an eval-only (store-side) context; cannot decrypt")
        pt = self._cc.Decrypt(onehot_ct, self._sk)
        pt.SetLength(self._n)
        return list(pt.GetRealPackedValue())


class BlindMaintenance:
    """E5: blind dream-cycle maintenance on encrypted SCALARS (resonance), pure-CKKS so the
    keys serialize (like BlindArgmaxCKKS / BlindRecallPRE). The untrusted store runs the
    Hebbian maintenance on ciphertext it cannot read, with public + eval keys ONLY:

      * DECAY — ``resonance *= factor`` (a plaintext scalar multiply): the per-cycle forgetting.
      * threshold COMPARE — ``step(resonance − threshold)`` -> an encrypted 0/1 indicator via
        the Chebyshev sign-approx: promotion (resonance ≥ promote) / eviction (resonance <
        prune), and conflict similarity bands (low ≤ sim ≤ high = two compares on a score ct).

    Resonance is scaled to ~[0,1] client-side (``resonance / max_resonance``) so the comparison
    sits in a bounded interval. The comparison has a TRANSITION BAND: a value within ~epsilon of
    the threshold may classify either way (the exact-boundary case decrypts to ~0.5). That is
    fine for a soft, every-cycle signal — a near-threshold fact merely promotes/evicts a cycle
    early or late, and resolves as resonance moves. Node-measured: decay ~4ms, one comparison
    ~3.5s/batch at HEStd_128_classic; the cost lands on the dream cycle, never the hot recall
    path — exactly where "cycles-not-seconds" pays off (roadmap §4/§9).

    Scope (E5 5a): the homomorphic decay + threshold-comparison primitives, serializable across
    the store/client split. ACTING on an encrypted indicator (a blind tier flip / conditional
    eviction) and which maintenance ops run fully blind vs need per-cycle client assistance is
    5b (a design decision; some steps — e.g. structural pruning — may stay client-assisted)."""

    _BLOBS = ("ctx", "pub", "em")

    def __init__(self, cc, batch: int):
        self._cc = cc
        self._batch = batch
        self._pub = None
        self._sk = None

    @classmethod
    def generate(cls, batch: int = 8, security: Optional[str] = None,
                 depth: Optional[int] = None):
        """SETUP/CLIENT: build the maintenance context + mult key. ``security`` defaults to
        HEStd_128_classic (production); pass ``"HEStd_NotSet"`` for the fast small-ring PoC.
        ``depth`` defaults to ``_MAINT_DEPTH`` (14 — supports the homomorphic ``ge_threshold``
        Chebyshev step); pass ``_MAINT_BLIND_DEPTH`` (1) for the DECAY-ONLY light keyset (the
        provider blind-maint path, ~0.8MB vs ~63MB). A depth-1 context can ``decay``/``encrypt``/
        ``decrypt`` but NOT ``ge_threshold`` (which exhausts depth-1 immediately); the maintainer
        decays FROM ORIGIN (one mult by factor**elapsed) so depth 1 lasts unbounded cycles.
        Returns ``(client_instance, key_blobs, secret_blob)``; the store reconstructs via
        ``load_eval`` (no secret)."""
        _require()
        b = _next_pow2_at_least(max(2, batch))
        sec_name = security or _CMP_SECURITY_DEFAULT
        mult_depth = _MAINT_DEPTH if depth is None else int(depth)
        p = _ofhe.CCParamsCKKSRNS()
        p.SetMultiplicativeDepth(mult_depth)
        p.SetScalingModSize(_SCALE_BITS)
        p.SetFirstModSize(60)
        p.SetScalingTechnique(_ofhe.FLEXIBLEAUTO)
        p.SetSecurityLevel(getattr(_ofhe, sec_name))
        if sec_name == "HEStd_NotSet":
            p.SetRingDim(_MAINT_NOTSET_RING)
        p.SetBatchSize(b)
        cc = _ofhe.GenCryptoContext(p)
        for feat in (_ofhe.PKE, _ofhe.KEYSWITCH, _ofhe.LEVELEDSHE, _ofhe.ADVANCEDSHE):
            cc.Enable(feat)
        kp = cc.KeyGen()
        cc.EvalMultKeyGen(kp.secretKey)
        # By-tag (not "") so this keyset's mult key is ISOLATED when generated alongside the
        # recall/HRR keysets in one setup process (see BlindRecallPRE.generate). Single-context: ==  "".
        key_blobs = {
            "ctx": _ofhe.Serialize(cc, _ofhe.BINARY),
            "pub": _ofhe.Serialize(kp.publicKey, _ofhe.BINARY),
            "em": _ofhe.SerializeEvalMultKeyString(_ofhe.BINARY, kp.secretKey.GetKeyTag()),
        }
        secret_blob = _ofhe.Serialize(kp.secretKey, _ofhe.BINARY)
        inst = cls(cc, b)
        inst._pub = kp.publicKey
        inst._sk = kp.secretKey
        return inst, key_blobs, secret_blob

    @classmethod
    def load_eval(cls, key_blobs: Dict[str, bytes], batch: int) -> "BlindMaintenance":
        """STORE side (fresh process): ctx + mult key + pub. No secret — decays and compares
        but never decrypts."""
        _require()
        cc = _ofhe.DeserializeCryptoContextString(key_blobs["ctx"], _ofhe.BINARY)
        _ofhe.DeserializeEvalMultKeyString(key_blobs["em"], _ofhe.BINARY)
        inst = cls(cc, batch)
        inst._pub = _ofhe.DeserializePublicKeyString(key_blobs["pub"], _ofhe.BINARY)
        inst._sk = None
        return inst

    @classmethod
    def load_client(cls, key_blobs: Dict[str, bytes], batch: int,
                    secret_blob: bytes) -> "BlindMaintenance":
        """CLIENT side: eval reconstruction + the unwrapped secret key (decrypts results)."""
        inst = cls.load_eval(key_blobs, batch)
        inst._sk = _ofhe.DeserializePrivateKeyString(secret_blob, _ofhe.BINARY)
        return inst

    @property
    def batch(self) -> int:
        return self._batch

    def encrypt_scalars(self, values: List[float]):
        """Encrypt a batch of per-fact resonance scalars (scaled to ~[0,1]) into one ct."""
        _require()
        v = list(values)[:self._batch]
        v = v + [0.0] * (self._batch - len(v))
        return self._cc.Encrypt(self._pub, self._cc.MakeCKKSPackedPlaintext(v, 1, 0, None, self._batch))

    def serialize_ct(self, ct) -> bytes:
        """Serialize a resonance ciphertext to bytes for storage (e.g. semantic_he_meta)."""
        _require()
        return _ofhe.Serialize(ct, _ofhe.BINARY)

    def _as_ct(self, ct):
        """Accept a live Ciphertext or a serialized blob -> a live Ciphertext."""
        return (_ofhe.DeserializeCiphertextString(ct, _ofhe.BINARY)
                if isinstance(ct, (bytes, bytearray)) else ct)

    def decay(self, ct, factor: float):
        """STORE side: per-cycle decay — multiply every encrypted resonance by a plaintext
        ``factor`` (depth 1). Public + eval keys only. Accepts a live ct or a stored blob."""
        _require()
        return self._cc.EvalMult(self._as_ct(ct), float(factor))

    def ge_threshold(self, ct, threshold: float):
        """STORE side: encrypted 0/1 indicator of ``scaled_resonance >= threshold`` via the
        Chebyshev step (public + eval keys only; NEVER the secret). Values must be scaled to
        ~[0,1]; classification is exact outside the transition band around ``threshold``."""
        _require()
        thr_pt = self._cc.MakeCKKSPackedPlaintext(
            [float(threshold)] * self._batch, 1, 0, None, self._batch)
        diff = self._cc.EvalSub(self._as_ct(ct), thr_pt)
        stepf = lambda x: 1.0 if x >= 0 else 0.0
        return self._cc.EvalChebyshevFunction(stepf, diff, -_MAINT_RANGE, _MAINT_RANGE,
                                              _MAINT_STEP_DEGREE)

    def decrypt_scalars(self, ct, n: Optional[int] = None) -> List[float]:
        """CLIENT side: decrypt a batch ct (live or stored blob) to a list of floats (decayed
        resonance, or the ~0/1 promotion/eviction indicators)."""
        _require()
        if self._sk is None:
            raise SecretRequiredError("this is an eval-only (store-side) context; cannot decrypt")
        pt = self._cc.Decrypt(self._as_ct(ct), self._sk)
        length = n if n is not None else self._batch
        pt.SetLength(length)
        return list(pt.GetRealPackedValue()[:length])
