"""crypto_keys.py — passphrase-derived key hierarchy for the encrypted-at-rest tier (E0).

Dependency-free of LatticeStore and every store_* mixin ON PURPOSE (a leaf, like
store_common.py): the store imports it, never the reverse, so it can be loaded by the
test harness without dragging in the SQLite stack.

What it does (Tier 0 / encrypted-at-rest):

    passphrase --Argon2id(salt, versioned params)--> MASTER secret (bytearray, zeroized)
              --HKDF-SHA256(info="...rest-db-key...")--> 32-byte RAW DB KEY  -> SQLCipher PRAGMA key
              --HKDF-SHA256(info="...key-check...")---> 16-byte KEY-CHECK    -> stored, for fast
                                                                                wrong-passphrase detection

Forward-compatible with Tier 1 (HE): the master is a KEK from which named subkeys are
derived by HKDF, so the future HE keys (store public/eval key, agent use-key, the
re-encryption key) become siblings of the rest-db-key under the *same* master secret.
E0 derives only the rest-db-key today.

KEYSTORE SIDECAR (`<db>.keys`, JSON) holds ONLY non-secret material: KDF version, Argon2id
params, the random salt, and the key-check value (an HKDF output of the master under a
distinct info label — it reveals neither the master nor the DB key, and only enables the
same offline passphrase-guessing an attacker could already mount against the ciphertext,
which Argon2id is there to make expensive). The master secret and the DB key are NEVER
written to disk; they are re-derived from the passphrase on demand.

Secrets live in `bytearray`s so they can be wiped (`secure_zero`) and best-effort page-locked
(`try_mlock`) — true zeroization is not guaranteed under CPython, so this is defence-in-depth,
not a guarantee. The honest threat boundary is in ENCRYPTION_ROADMAP.md §2/§7.

Heavy/optional deps are imported lazily-guarded (mirrors store_common): importing this module
never fails; only the functions that actually need Argon2id raise if `argon2-cffi` is missing,
so non-encrypted deployments keep the "no new heavy deps" guarantee.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    from argon2.low_level import hash_secret_raw as _argon2_hash_raw, Type as _Argon2Type
    _ARGON2_AVAILABLE = True
except Exception as e:  # pragma: no cover - exercised only without argon2-cffi
    logger.debug("argon2-cffi not available: %s. Encrypted-at-rest disabled until installed.", e)
    _argon2_hash_raw = None
    _Argon2Type = None
    _ARGON2_AVAILABLE = False

# AES-256-GCM (for wrapping the Tier-1 HE secret key under a master subkey). The Python
# stdlib has no symmetric cipher, so an AEAD genuinely needs a dependency; `cryptography`
# is the standard provider. Lazy-guarded like argon2 and gated to the blind (Tier-1) path,
# so the default + Tier-0 deployments never require it (principle 3.6).
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _AESGCM
    _AEAD_AVAILABLE = True
except Exception as e:  # pragma: no cover - exercised only without `cryptography`
    logger.debug("cryptography not available: %s. Blind-store secret wrap disabled until installed.", e)
    _AESGCM = None
    _AEAD_AVAILABLE = False


# ── Versioned constants (bump the version, never edit params in place) ──────────
KEYSTORE_VERSION = 1

# Argon2id default profile. memory_cost is in KiB (argon2-cffi convention): 64 MiB.
# Strong enough for a single-user local store on modest hardware; tunable per-keystore.
DEFAULT_KDF_PARAMS: Dict[str, int] = {
    "time_cost": 3,
    "memory_cost_kib": 65536,
    "parallelism": 4,
    "hash_len": 32,
}

SALT_BYTES = 16
_DB_KEY_LEN = 32          # SQLCipher raw key = 256-bit
_KEY_CHECK_LEN = 16
_HE_WRAP_KEY_LEN = 32     # AES-256-GCM key for wrapping the HE secret
_GCM_NONCE_BYTES = 12     # 96-bit nonce, the GCM standard

# HKDF info labels — distinct per purpose so subkeys are cryptographically independent.
_INFO_DB_KEY = b"resonant-lattice/rest-db-key/v1"
_INFO_KEY_CHECK = b"resonant-lattice/key-check/v1"
# Tier-1 HE secret-wrapping subkey: sibling of the rest-db-key under the SAME master, so the
# whole key hierarchy (DB-at-rest + HE secret) hangs off the one passphrase (roadmap §4).
_INFO_HE_SECRET_WRAP = b"resonant-lattice/he-secret-wrap/v1"
# Tier-1 entity-set encryption subkey (E7 7b): another distinct sibling under the same master.
_INFO_ENTITY = b"resonant-lattice/entity-set/v1"

# AES-GCM associated data — authenticates the wrap-format version, so a blob from a future
# format cannot be silently unwrapped under this one.
HE_WRAP_VERSION = 1
_HE_WRAP_AAD = b"resonant-lattice/he-secret-wrap/v1"
ENTITY_WRAP_VERSION = 1
_ENTITY_AAD = b"resonant-lattice/entity-set/v1"

# Environment variable a non-interactive (autonomous / headless) launch can use to
# supply the passphrase. E1 replaces this with sealed sources (TPM2 / once-per-boot SSH).
ENV_PASSPHRASE = "RESONANT_LATTICE_PASSPHRASE"


class CryptoUnavailableError(RuntimeError):
    """Raised when an encrypted operation is requested but argon2-cffi is missing."""


class WrongPassphraseError(RuntimeError):
    """Raised when a supplied passphrase fails the keystore key-check."""


class WrapAuthError(RuntimeError):
    """Raised when unwrapping the HE secret fails (wrong key, tamper, or format mismatch)."""


def kdf_available() -> bool:
    return _ARGON2_AVAILABLE


def aead_available() -> bool:
    return _AEAD_AVAILABLE


def default_kdf_params() -> Dict[str, int]:
    return dict(DEFAULT_KDF_PARAMS)


# ── HKDF-SHA256 (RFC 5869), stdlib only (avoids a `cryptography` dependency) ────
def _hkdf_sha256(ikm: bytes, info: bytes, length: int = 32, salt: bytes = b"") -> bytes:
    if not salt:
        salt = b"\x00" * hashlib.sha256().digest_size
    prk = hmac.new(salt, bytes(ikm), hashlib.sha256).digest()
    okm = b""
    block = b""
    counter = 1
    while len(okm) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        okm += block
        counter += 1
    return okm[:length]


# ── Secret-buffer hygiene (best-effort, not a guarantee) ───────────────────────
def secure_zero(buf: Optional[bytearray]) -> None:
    """Overwrite a mutable secret buffer with zeros. No-op for None / immutable input."""
    if isinstance(buf, bytearray):
        for i in range(len(buf)):
            buf[i] = 0


def try_mlock(buf: bytearray) -> bool:
    """Best-effort page-lock of a secret buffer (mlock/VirtualLock). Never raises."""
    try:
        import ctypes
        addr = ctypes.addressof((ctypes.c_char * len(buf)).from_buffer(buf))
        if os.name == "nt":
            return bool(ctypes.windll.kernel32.VirtualLock(ctypes.c_void_p(addr), ctypes.c_size_t(len(buf))))
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(len(buf))) == 0
    except Exception as e:  # pragma: no cover - platform dependent
        logger.debug("try_mlock failed (non-fatal): %s", e)
        return False


def _derive_master(passphrase: bytes, salt: bytes, params: Dict[str, int]) -> bytearray:
    """Argon2id(passphrase, salt) -> master secret (bytearray, caller must secure_zero)."""
    if not _ARGON2_AVAILABLE:
        raise CryptoUnavailableError(
            "argon2-cffi is required for encrypted-at-rest memory. "
            "Install it (`pip install argon2-cffi`) or set encryption_mode=none."
        )
    raw = _argon2_hash_raw(
        secret=bytes(passphrase),
        salt=bytes(salt),
        time_cost=int(params["time_cost"]),
        memory_cost=int(params["memory_cost_kib"]),
        parallelism=int(params["parallelism"]),
        hash_len=int(params["hash_len"]),
        type=_Argon2Type.ID,
    )
    master = bytearray(raw)
    try_mlock(master)
    return master


# ── Keystore lifecycle ─────────────────────────────────────────────────────────
def create_keystore(passphrase: bytes, params: Optional[Dict[str, int]] = None) -> Dict:
    """Build a fresh keystore dict for a NEW encrypted store (does not write to disk).

    Generates a random salt, derives the master to compute the key-check, then wipes the
    master. Returns a JSON-serialisable dict containing ONLY non-secret material.
    """
    params = dict(params or DEFAULT_KDF_PARAMS)
    salt = secrets.token_bytes(SALT_BYTES)
    master = _derive_master(passphrase, salt, params)
    try:
        key_check = _hkdf_sha256(master, _INFO_KEY_CHECK, _KEY_CHECK_LEN)
    finally:
        secure_zero(master)
    return {
        "version": KEYSTORE_VERSION,
        "kdf": {"algo": "argon2id", **params},
        "salt_b64": base64.b64encode(salt).decode("ascii"),
        "key_check_b64": base64.b64encode(key_check).decode("ascii"),
    }


def save_keystore(path: str, keystore: Dict) -> None:
    """Write the keystore JSON sidecar atomically with owner-only perms where supported."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(keystore, fh, indent=2, sort_keys=True)
    try:
        os.chmod(tmp, 0o600)  # best-effort on POSIX; no-op semantics on Windows
    except Exception as e:  # pragma: no cover
        logger.debug("chmod on keystore failed (non-fatal): %s", e)
    os.replace(tmp, path)


def load_keystore(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        keystore = json.load(fh)
    if keystore.get("version") != KEYSTORE_VERSION:
        logger.warning(
            "Keystore version %s != supported %s; proceeding best-effort.",
            keystore.get("version"), KEYSTORE_VERSION,
        )
    return keystore


def keystore_is_secret_free(keystore: Dict) -> bool:
    """Audit guard: the keystore must never carry the master, the DB key, or a passphrase.

    Used by the E0 substrate test. Whitelists the known non-secret keys.
    """
    allowed_top = {"version", "kdf", "salt_b64", "key_check_b64"}
    if set(keystore) - allowed_top:
        return False
    allowed_kdf = {"algo", "time_cost", "memory_cost_kib", "parallelism", "hash_len"}
    return not (set(keystore.get("kdf", {})) - allowed_kdf)


# ── Key derivation (the runtime path) ──────────────────────────────────────────
def _params_from_keystore(keystore: Dict) -> Dict[str, int]:
    kdf = keystore.get("kdf", {})
    return {
        "time_cost": kdf.get("time_cost", DEFAULT_KDF_PARAMS["time_cost"]),
        "memory_cost_kib": kdf.get("memory_cost_kib", DEFAULT_KDF_PARAMS["memory_cost_kib"]),
        "parallelism": kdf.get("parallelism", DEFAULT_KDF_PARAMS["parallelism"]),
        "hash_len": kdf.get("hash_len", DEFAULT_KDF_PARAMS["hash_len"]),
    }


def verify_passphrase(passphrase: bytes, keystore: Dict) -> bool:
    """Constant-time check of a passphrase against the keystore key-check (no DB needed)."""
    salt = base64.b64decode(keystore["salt_b64"])
    master = _derive_master(passphrase, salt, _params_from_keystore(keystore))
    try:
        got = _hkdf_sha256(master, _INFO_KEY_CHECK, _KEY_CHECK_LEN)
    finally:
        secure_zero(master)
    expected = base64.b64decode(keystore["key_check_b64"])
    return hmac.compare_digest(got, expected)


def derive_db_key(passphrase: bytes, keystore: Dict, *, verify: bool = True) -> bytearray:
    """Re-derive the 32-byte raw SQLCipher DB key from the passphrase + keystore.

    Raises WrongPassphraseError if `verify` and the key-check fails. The returned
    bytearray is the caller's to wipe (`secure_zero`) right after `PRAGMA key`.
    """
    salt = base64.b64decode(keystore["salt_b64"])
    master = _derive_master(passphrase, salt, _params_from_keystore(keystore))
    try:
        if verify:
            got = _hkdf_sha256(master, _INFO_KEY_CHECK, _KEY_CHECK_LEN)
            expected = base64.b64decode(keystore["key_check_b64"])
            if not hmac.compare_digest(got, expected):
                raise WrongPassphraseError("passphrase does not match this keystore")
        db_key = bytearray(_hkdf_sha256(master, _INFO_DB_KEY, _DB_KEY_LEN))
    finally:
        secure_zero(master)
    try_mlock(db_key)
    return db_key


def db_key_to_pragma_value(db_key: bytes) -> str:
    """Format a raw key for SQLCipher: PRAGMA key = "x'<64 hex>'" (skips SQLCipher's own KDF)."""
    return "x'" + bytes(db_key).hex() + "'"


# ── Tier-1 (HE blind store): wrap/unwrap the HE secret key under the master ──────
# The HE keypair is randomly generated at setup (not derived from the master); the
# SECRET key is then AES-256-GCM-wrapped under a master-derived subkey and persisted
# wrapped, so it survives setup yet is recoverable only by re-deriving the master from
# the passphrase. This finishes the §4 key hierarchy: rest-db-key and the HE-secret-wrap
# key are independent siblings of the one master. The store node never holds the wrapped
# secret (it gets only the public/eval blobs); only the client/agent side unwraps it.
def derive_he_wrap_key(passphrase: bytes, keystore: Dict, *, verify: bool = True) -> bytearray:
    """Re-derive the 32-byte AES-256-GCM key that wraps the HE secret (Tier-1 hierarchy).

    Sibling of `derive_db_key` under the same master/keystore but a distinct HKDF info
    label, so it is cryptographically independent of the rest-db-key. Raises
    WrongPassphraseError if `verify` and the key-check fails. The returned bytearray is the
    caller's to wipe (`secure_zero`) right after wrap/unwrap.
    """
    salt = base64.b64decode(keystore["salt_b64"])
    master = _derive_master(passphrase, salt, _params_from_keystore(keystore))
    try:
        if verify:
            got = _hkdf_sha256(master, _INFO_KEY_CHECK, _KEY_CHECK_LEN)
            expected = base64.b64decode(keystore["key_check_b64"])
            if not hmac.compare_digest(got, expected):
                raise WrongPassphraseError("passphrase does not match this keystore")
        wrap_key = bytearray(_hkdf_sha256(master, _INFO_HE_SECRET_WRAP, _HE_WRAP_KEY_LEN))
    finally:
        secure_zero(master)
    try_mlock(wrap_key)
    return wrap_key


def wrap_he_secret(secret_blob: bytes, wrap_key: bytes) -> Dict:
    """AES-256-GCM-encrypt the serialized HE secret key under `wrap_key`.

    Returns a JSON-serialisable dict (version/alg/nonce/ciphertext) safe to persist next to
    the public/eval key blobs — it is ciphertext, not a secret. A fresh random nonce is used
    per call; the wrap-format version is bound as associated data so it is authenticated.
    """
    if not _AEAD_AVAILABLE:
        raise CryptoUnavailableError(
            "cryptography is required for the blind store's HE secret wrap. "
            "Install it (`pip install cryptography`) or use encryption_mode in (none, at_rest)."
        )
    if len(wrap_key) != _HE_WRAP_KEY_LEN:
        raise ValueError(f"wrap_key must be {_HE_WRAP_KEY_LEN} bytes, got {len(wrap_key)}")
    nonce = secrets.token_bytes(_GCM_NONCE_BYTES)
    ct = _AESGCM(bytes(wrap_key)).encrypt(nonce, bytes(secret_blob), _HE_WRAP_AAD)
    return {
        "version": HE_WRAP_VERSION,
        "alg": "AES-256-GCM",
        "nonce_b64": base64.b64encode(nonce).decode("ascii"),
        "ct_b64": base64.b64encode(ct).decode("ascii"),
    }


def unwrap_he_secret(wrapped: Dict, wrap_key: bytes) -> bytearray:
    """Inverse of `wrap_he_secret`: authenticate + decrypt back to the HE secret-key bytes.

    Raises WrapAuthError on wrong key, tampering, or a version/format mismatch (GCM's auth
    tag makes all three indistinguishable and loud, which is what we want). Returns a
    wipeable bytearray (caller `secure_zero`s it after deserializing the key); the transient
    plaintext `bytes` GCM produces cannot itself be wiped under CPython (best-effort, §2).
    """
    if not _AEAD_AVAILABLE:
        raise CryptoUnavailableError(
            "cryptography is required for the blind store's HE secret unwrap. "
            "Install it (`pip install cryptography`) or use encryption_mode in (none, at_rest)."
        )
    if wrapped.get("version") != HE_WRAP_VERSION:
        raise WrapAuthError(
            f"unsupported HE wrap version {wrapped.get('version')!r} (expected {HE_WRAP_VERSION})"
        )
    if len(wrap_key) != _HE_WRAP_KEY_LEN:
        raise ValueError(f"wrap_key must be {_HE_WRAP_KEY_LEN} bytes, got {len(wrap_key)}")
    nonce = base64.b64decode(wrapped["nonce_b64"])
    ct = base64.b64decode(wrapped["ct_b64"])
    try:
        pt = _AESGCM(bytes(wrap_key)).decrypt(nonce, ct, _HE_WRAP_AAD)
    except Exception as e:  # InvalidTag (wrong key / tamper) and any malformed input
        raise WrapAuthError("HE secret unwrap failed (wrong key or tampered blob)") from e
    out = bytearray(pt)
    try_mlock(out)
    return out


# ── Tier-1 HE keystore sidecar (public/eval blobs + WRAPPED secrets) ─────────────
# Separate from the at-rest keystore: it holds the (non-secret) CKKS public/eval/rekey
# blobs plus the AES-GCM-WRAPPED HE secrets (master + agent use-key), both keyed off the
# SAME master via derive_he_wrap_key. A real store node would receive ONLY the key_blobs;
# the wrapped secrets travel with the trusted client. Nothing plaintext-secret on disk.
HE_KEYSTORE_VERSION = 1
_HE_PUBLIC_BLOB_KEYS = {"ctx", "pub", "em", "ea", "rk"}


def create_he_keystore(key_blobs: Dict[str, bytes], wrapped_secrets: Dict[str, Dict],
                       meta: Dict) -> Dict:
    """Build a JSON-serialisable HE keystore: b64 public/eval blobs + wrapped secrets + meta."""
    return {
        "version": HE_KEYSTORE_VERSION,
        "meta": dict(meta),                       # {dim, batch, engine, he_version}
        "key_blobs_b64": {k: base64.b64encode(v).decode("ascii") for k, v in key_blobs.items()},
        "wrapped_secrets": dict(wrapped_secrets), # {master: {...}, agent: {...}} (already JSON-safe)
    }


def save_he_keystore(path: str, ks: Dict) -> None:
    """Write the HE keystore sidecar atomically, owner-only perms where supported."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ks, fh, indent=2, sort_keys=True)
    try:
        os.chmod(tmp, 0o600)  # best-effort on POSIX; no-op on Windows
    except Exception as e:  # pragma: no cover
        logger.debug("chmod on HE keystore failed (non-fatal): %s", e)
    os.replace(tmp, path)


def load_he_keystore(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        ks = json.load(fh)
    if ks.get("version") != HE_KEYSTORE_VERSION:
        logger.warning("HE keystore version %s != supported %s; proceeding best-effort.",
                       ks.get("version"), HE_KEYSTORE_VERSION)
    return ks


def he_key_blobs_from_keystore(ks: Dict) -> Dict[str, bytes]:
    """Decode the b64 public/eval blobs back to bytes for he_crypto's load_eval/load_*."""
    return {k: base64.b64decode(v) for k, v in ks.get("key_blobs_b64", {}).items()}


def he_keystore_is_secret_free(ks: Dict) -> bool:
    """Audit guard: the HE keystore carries NO unwrapped secret — only public/eval blobs and
    AES-GCM-WRAPPED secrets (which are ciphertext, safe to persist)."""
    if set(ks) - {"version", "meta", "key_blobs_b64", "wrapped_secrets"}:
        return False
    if set(ks.get("key_blobs_b64", {})) - _HE_PUBLIC_BLOB_KEYS:
        return False  # a raw secret-key blob ("sk"/"secret") in the clear would fail here
    for w in ks.get("wrapped_secrets", {}).values():
        if not (isinstance(w, dict) and w.get("alg") == "AES-256-GCM"):
            return False
    return True


def setup_or_load_blind_client(passphrase: bytes, keystore: Dict, he_keystore_path: str,
                               dim: int, *, role: str = "user"):
    """Return ``(blind_client, he_keystore_dict, created)`` for the Tier-1 blind tier.

    First run (no HE keystore): generate a fresh ``BlindRecallPRE`` keypair, AES-GCM-WRAP the
    master + agent secrets under the master-derived wrap key, persist the HE keystore, and
    return the live ``generate()`` instance (no re-deserialize -> no global eval-key
    collision). Later runs: load the public/eval blobs + unwrap the requested role's secret
    and rebuild the client. ``role`` picks which secret the client holds — 'user' (master,
    god-mode) or 'agent' (use-key; raw-DB reads are rejected). ``he_crypto`` is imported
    lazily so openfhe is pulled ONLY on the blind path."""
    import he_crypto  # lazy: openfhe only here (blind tier)
    wrap_key = derive_he_wrap_key(passphrase, keystore)
    try:
        if not os.path.exists(he_keystore_path):
            client, key_blobs, secret_blobs = he_crypto.BlindRecallPRE.generate(dim=dim)
            wrapped = {name: wrap_he_secret(blob, wrap_key) for name, blob in secret_blobs.items()}
            meta = {"dim": int(dim), "batch": int(client.batch), "engine": "BlindRecallPRE",
                    "he_version": int(getattr(he_crypto, "HE_PARAMS_VERSION", 1))}
            ks = create_he_keystore(key_blobs, wrapped, meta)
            save_he_keystore(he_keystore_path, ks)
            logger.warning("Blind-tier HE keystore CREATED at %s. The passphrase is the ONLY "
                           "way to recover the HE secret — there is NO recovery.", he_keystore_path)
            return client, ks, True
        ks = load_he_keystore(he_keystore_path)
        key_blobs = he_key_blobs_from_keystore(ks)
        meta = ks.get("meta", {})
        dim_ks, batch_ks = int(meta.get("dim", dim)), int(meta["batch"])
        secret = unwrap_he_secret(ks["wrapped_secrets"]["master" if role == "user" else "agent"], wrap_key)
        try:
            loader = he_crypto.BlindRecallPRE.load_user if role == "user" else he_crypto.BlindRecallPRE.load_client
            client = loader(key_blobs, dim_ks, batch_ks, bytes(secret))
        finally:
            secure_zero(secret)
        return client, ks, False
    finally:
        secure_zero(wrap_key)


# ── Tier-1 MULTI-keyset HE keystore (Option A, 2026-06-19) ───────────────────────
# The blind tier needs SEVERAL purpose-built HE contexts at once (decided + node-proven
# 2026-06-19): recall+PRE @ embed-dim, HRR recall+PRE @ 2·hrr-dim, and a LIGHT decay-only
# maintenance context. One sidecar holds all of them as NAMED keysets, each a self-contained
# {public/eval blobs + wrapped secrets + meta} — exactly the single-keyset shape, nested under
# `keysets`. They coexist in one runtime process: OpenFHE's global eval-key store keys by
# context tag, and each engine's generate() now serializes its eval keys BY TAG (he_crypto), so
# one setup process can keygen all keysets with isolated blobs (node-proven). The maint keyset is
# generated LIGHT (he_crypto._MAINT_BLIND_DEPTH=1, ~0.8MB vs ~63MB) because the provider path
# decays FROM ORIGIN — see he_crypto._MAINT_BLIND_DEPTH for the why.
HE_MULTI_KEYSTORE_VERSION = 2
_HE_KEYSET_NAMES = ("recall", "hrr", "maint")


def create_multi_he_keystore(keysets: Dict[str, Dict]) -> Dict:
    """Build a JSON-serialisable MULTI-keyset HE keystore from ``{name: entry}`` where each entry
    is ``{"key_blobs": {k: bytes}, "wrapped_secrets": {name: wrapped_dict}, "meta": {...}}`` — the
    public/eval blobs are b64-encoded, the wrapped secrets are already JSON-safe ciphertext."""
    out = {"version": HE_MULTI_KEYSTORE_VERSION, "keysets": {}}
    for name, ent in keysets.items():
        out["keysets"][name] = {
            "meta": dict(ent.get("meta", {})),
            "key_blobs_b64": {k: base64.b64encode(v).decode("ascii")
                              for k, v in ent["key_blobs"].items()},
            "wrapped_secrets": dict(ent["wrapped_secrets"]),
        }
    return out


def load_multi_he_keystore(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        ks = json.load(fh)
    if ks.get("version") != HE_MULTI_KEYSTORE_VERSION:
        logger.warning("multi HE keystore version %s != supported %s; proceeding best-effort.",
                       ks.get("version"), HE_MULTI_KEYSTORE_VERSION)
    return ks


def multi_he_key_blobs_from_keystore(ks: Dict, name: str) -> Dict[str, bytes]:
    """Decode one named keyset's b64 public/eval blobs back to bytes for he_crypto's load_*."""
    return {k: base64.b64decode(v)
            for k, v in ks["keysets"][name].get("key_blobs_b64", {}).items()}


def multi_he_keystore_is_secret_free(ks: Dict) -> bool:
    """Audit guard: every keyset carries ONLY public/eval blobs + AES-GCM-WRAPPED secrets (no
    unwrapped secret key in the clear). Same property as he_keystore_is_secret_free, per keyset."""
    if set(ks) - {"version", "keysets"}:
        return False
    for ent in ks.get("keysets", {}).values():
        if set(ent) - {"meta", "key_blobs_b64", "wrapped_secrets"}:
            return False
        if set(ent.get("key_blobs_b64", {})) - _HE_PUBLIC_BLOB_KEYS:
            return False  # a raw secret-key blob in the clear would fail here
        for w in ent.get("wrapped_secrets", {}).values():
            if not (isinstance(w, dict) and w.get("alg") == "AES-256-GCM"):
                return False
    return True


def setup_or_load_blind_contexts(passphrase: bytes, keystore: Dict, he_keystore_path: str, *,
                                 embed_dim: int, hrr_dim: int, maint_batch: int = 8,
                                 role: str = "user"):
    """Return ``({"recall":client, "hrr":client, "maint":client}, he_keystore, created)`` — the
    Option A multi-keyset blind tier.

      * ``recall`` = ``BlindRecallPRE`` @ ``embed_dim`` (encrypted-embedding cosine + PRE).
      * ``hrr``    = ``BlindRecallPRE`` @ ``2*hrr_dim`` (the HE dim of the HRR (cos,sin) lift).
      * ``maint``  = ``BlindMaintenance`` LIGHT (decay-only, ``_MAINT_BLIND_DEPTH``) over scalars.

    First run (no HE keystore): generate all three in ONE process — by-tag eval-key serialization
    (he_crypto) keeps the blobs isolated and the live instances usable — AES-GCM-WRAP every secret
    under the master wrap key, persist, and return the live instances (no re-deserialize). Later
    runs: load + unwrap the requested ``role``'s secret per keyset and rebuild the clients. ``role``
    picks the PRE secret for recall/HRR — 'user' (master, god-mode) | 'agent' (use-key, raw-DB reads
    rejected); maint has a single client secret (needed for settle/get_resonance). ``he_crypto`` is
    imported lazily so openfhe is pulled ONLY on the blind path."""
    import he_crypto  # lazy: openfhe only here (blind tier)
    wrap_key = derive_he_wrap_key(passphrase, keystore)
    try:
        if not os.path.exists(he_keystore_path):
            hrr_he_dim = 2 * int(hrr_dim)
            he_ver = int(getattr(he_crypto, "HE_PARAMS_VERSION", 1))
            recall, r_kb, r_sb = he_crypto.BlindRecallPRE.generate(dim=int(embed_dim))
            hrr,    h_kb, h_sb = he_crypto.BlindRecallPRE.generate(dim=hrr_he_dim)
            maint,  m_kb, m_sb = he_crypto.BlindMaintenance.generate(
                batch=int(maint_batch), depth=he_crypto._MAINT_BLIND_DEPTH)
            keysets = {
                "recall": {"key_blobs": r_kb,
                           "wrapped_secrets": {n: wrap_he_secret(b, wrap_key) for n, b in r_sb.items()},
                           "meta": {"dim": int(embed_dim), "batch": int(recall.batch),
                                    "engine": "BlindRecallPRE", "he_version": he_ver}},
                "hrr":    {"key_blobs": h_kb,
                           "wrapped_secrets": {n: wrap_he_secret(b, wrap_key) for n, b in h_sb.items()},
                           "meta": {"dim": hrr_he_dim, "batch": int(hrr.batch),
                                    "engine": "BlindRecallPRE", "he_version": he_ver}},
                "maint":  {"key_blobs": m_kb,
                           "wrapped_secrets": {"secret": wrap_he_secret(m_sb, wrap_key)},
                           "meta": {"batch": int(maint.batch), "engine": "BlindMaintenance",
                                    "depth": int(he_crypto._MAINT_BLIND_DEPTH), "he_version": he_ver}},
            }
            ks = create_multi_he_keystore(keysets)
            save_he_keystore(he_keystore_path, ks)
            logger.warning("Blind-tier MULTI HE keystore CREATED at %s (recall/hrr/maint). The "
                           "passphrase is the ONLY way to recover the HE secrets — NO recovery.",
                           he_keystore_path)
            return {"recall": recall, "hrr": hrr, "maint": maint}, ks, True

        ks = load_multi_he_keystore(he_keystore_path)
        clients: Dict[str, object] = {}
        # recall + hrr: BlindRecallPRE, role picks which secret (master|agent) to unwrap.
        for name in ("recall", "hrr"):
            meta = ks["keysets"][name]["meta"]
            kb = multi_he_key_blobs_from_keystore(ks, name)
            wrapped = ks["keysets"][name]["wrapped_secrets"]["master" if role == "user" else "agent"]
            secret = unwrap_he_secret(wrapped, wrap_key)
            try:
                loader = (he_crypto.BlindRecallPRE.load_user if role == "user"
                          else he_crypto.BlindRecallPRE.load_client)
                clients[name] = loader(kb, int(meta["dim"]), int(meta["batch"]), bytes(secret))
            finally:
                secure_zero(secret)
        # maint: BlindMaintenance has a single secret (no PRE split).
        m_meta = ks["keysets"]["maint"]["meta"]
        m_kb = multi_he_key_blobs_from_keystore(ks, "maint")
        m_secret = unwrap_he_secret(ks["keysets"]["maint"]["wrapped_secrets"]["secret"], wrap_key)
        try:
            clients["maint"] = he_crypto.BlindMaintenance.load_client(
                m_kb, int(m_meta["batch"]), bytes(m_secret))
        finally:
            secure_zero(m_secret)
        return clients, ks, False
    finally:
        secure_zero(wrap_key)


# ── Tier-1 entity-set encryption at rest (E7 7b, client-side no-leak) ────────────
# Per-fact entity NAMES are AEAD-encrypted to an OPAQUE blob with a RANDOM nonce, so the
# untrusted store can't read them AND identical entity sets are indistinguishable on disk
# (no deterministic token, so the store learns NO entity co-occurrence — user posture
# 2026-06-19). Overlap / conflict detection are CLIENT-side ops on the decrypted sets.
def derive_entity_key(passphrase: bytes, keystore: Dict, *, verify: bool = True) -> bytearray:
    """Re-derive the 32-byte AES-256-GCM key for entity-set encryption (Tier-1 hierarchy) —
    a distinct HKDF sibling of the rest-db-key / HE-wrap key under the SAME master. The
    returned bytearray is the caller's to wipe (`secure_zero`)."""
    salt = base64.b64decode(keystore["salt_b64"])
    master = _derive_master(passphrase, salt, _params_from_keystore(keystore))
    try:
        if verify:
            got = _hkdf_sha256(master, _INFO_KEY_CHECK, _KEY_CHECK_LEN)
            if not hmac.compare_digest(got, base64.b64decode(keystore["key_check_b64"])):
                raise WrongPassphraseError("passphrase does not match this keystore")
        key = bytearray(_hkdf_sha256(master, _INFO_ENTITY, _HE_WRAP_KEY_LEN))
    finally:
        secure_zero(master)
    try_mlock(key)
    return key


def encrypt_entities(entities, key: bytes) -> bytes:
    """AES-256-GCM-encrypt a fact's entity-name list to an OPAQUE blob (``version||nonce||ct``).

    Names are normalized (strip/lower) + deduped + sorted before encryption (so the plaintext
    is canonical), then encrypted under a fresh RANDOM nonce — identical entity sets yield
    DIFFERENT ciphertext, so the store sees no equality/co-occurrence. Overlap is a client op
    after `decrypt_entities`."""
    if not _AEAD_AVAILABLE:
        raise CryptoUnavailableError(
            "cryptography is required for blind entity encryption. Install it "
            "(`pip install cryptography`) or use encryption_mode in (none, at_rest).")
    if len(key) != _HE_WRAP_KEY_LEN:
        raise ValueError(f"key must be {_HE_WRAP_KEY_LEN} bytes, got {len(key)}")
    names = sorted({str(e).strip().lower() for e in (entities or []) if e and str(e).strip()})
    payload = json.dumps(names, separators=(",", ":")).encode("utf-8")
    nonce = secrets.token_bytes(_GCM_NONCE_BYTES)
    ct = _AESGCM(bytes(key)).encrypt(nonce, payload, _ENTITY_AAD)
    return bytes([ENTITY_WRAP_VERSION]) + nonce + ct


def decrypt_entities(blob: bytes, key: bytes) -> list:
    """Inverse of `encrypt_entities` → the sorted entity-name list. Raises WrapAuthError on a
    wrong key, tampering, or a version mismatch (GCM's auth tag makes all three loud)."""
    if not _AEAD_AVAILABLE:
        raise CryptoUnavailableError("cryptography is required for blind entity decryption.")
    if len(key) != _HE_WRAP_KEY_LEN:
        raise ValueError(f"key must be {_HE_WRAP_KEY_LEN} bytes, got {len(key)}")
    if not blob or blob[0] != ENTITY_WRAP_VERSION:
        raise WrapAuthError(f"unsupported entity-blob version {blob[:1]!r} (expected {ENTITY_WRAP_VERSION})")
    nonce = blob[1:1 + _GCM_NONCE_BYTES]
    ct = blob[1 + _GCM_NONCE_BYTES:]
    try:
        pt = _AESGCM(bytes(key)).decrypt(nonce, ct, _ENTITY_AAD)
    except Exception as e:
        raise WrapAuthError("entity decrypt failed (wrong key or tampered blob)") from e
    return list(json.loads(pt.decode("utf-8")))


# ── Passphrase source (E0: explicit / env / interactive) ───────────────────────
def get_passphrase(explicit: Optional[str] = None, *, prompt: bool = False,
                   prompt_label: str = "Memory passphrase: ") -> Optional[bytearray]:
    """Resolve the passphrase from, in order: explicit arg, env var, optional prompt.

    Returns a MUTABLE ``bytearray`` (or None if no source yields one) so the caller can
    ``secure_zero`` it once the keys are derived — the same wipeable-secret contract as
    ``derive_db_key``'s returned key. Every consumer feeds it through ``_derive_master``,
    which does ``bytes(passphrase)`` at the KDF boundary, so a bytearray is accepted
    everywhere a passphrase is taken. (Previously this returned immutable ``bytes``, so the
    ``isinstance(..., bytearray)`` wipe guards in the provider resolvers silently never fired.)
    NOTE: an env-sourced passphrase still lives in ``os.environ`` for the process lifetime —
    wiping our buffer is defence-in-depth, not a full scrub. E1 replaces this with sealed
    sources (TPM2 / once-per-boot SSH unlock).
    """
    if explicit:
        return bytearray(explicit.encode("utf-8"))
    env = os.environ.get(ENV_PASSPHRASE)
    if env:
        return bytearray(env.encode("utf-8"))
    if prompt:
        import getpass
        entered = getpass.getpass(prompt_label)
        if entered:
            return bytearray(entered.encode("utf-8"))
    return None
