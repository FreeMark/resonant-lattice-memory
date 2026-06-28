"""
store_common.py — shared low-level helpers for the LatticeStore mixins.

Dependency-free of LatticeStore (and of every store_* mixin) ON PURPOSE: the
test harness loads ``store`` via ``spec_from_file_location`` without registering
it in ``sys.modules``, so a mixin doing ``from store import …`` would trigger a
fresh, partial re-import of store.py and crash on a circular import. Routing the
shared primitives through this leaf module avoids that entirely.

Holds: the ``serialize_vector`` helper, the encryption-aware ``sqlite3`` binding
selection (pysqlite3 / stdlib, or ``sqlcipher3`` when encrypted-at-rest is signalled
via ``RESONANT_LATTICE_DB_ENCRYPTED``) — centralised so every module that catches
``sqlite3.IntegrityError`` shares the same exception type the connection raises —
and the optional-dependency import blocks for HRR (holographic) and the entity
extractor.

NOTE: ``sqlite_vec`` is intentionally NOT imported here — it stays a hard import
in store.py so importing the store still fails (and the store tests still skip)
when sqlite-vec is unavailable.
"""

import logging
import os
import struct
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def ensure_plugin_on_path() -> None:
    """Ensure this package dir is first on sys.path.

    The Hermes plugin loader imports modules via exec_module on bare filenames,
    so sibling imports like ``from store import …`` only resolve if the plugin
    root is on sys.path.  Call this early (before any relative-style imports)
    in every top-level module that the loader will touch.
    """
    _plugin_dir = str(Path(__file__).parent.resolve())
    if _plugin_dir not in sys.path:
        sys.path.insert(0, _plugin_dir)

# ── SQLite binding selection (encryption-aware) ────────────────────────────────
# Centralised so every module that catches ``sqlite3.IntegrityError`` shares the
# EXACT exception class the live connection raises. The binding is chosen ONCE, at
# first import, and cannot be safely swapped afterwards: the store_* mixins capture
# the class via ``from store_common import sqlite3`` at their own import time.
#
# Encrypted-at-rest (E0) selects SQLCipher (``sqlcipher3``). Because this module is
# imported at plugin load (before any config is read), the choice is signalled by
# the ``RESONANT_LATTICE_DB_ENCRYPTED`` environment variable, which the deployment /
# ``hermes memory setup`` exports when encryption is enabled. The provider validates
# that its ``encryption_mode`` matches the active binding and raises an actionable
# error on mismatch; tests select the encrypted binding in a fresh subprocess.
_ENCRYPTION_ENV = "RESONANT_LATTICE_DB_ENCRYPTED"
_ENCRYPTION_FALSEY = {"", "0", "false", "no", "none", "off"}


def env_encryption_on() -> bool:
    """True iff the env signal asks for the SQLCipher (encrypted-at-rest) binding."""
    return os.environ.get(_ENCRYPTION_ENV, "").strip().lower() not in _ENCRYPTION_FALSEY


def _select_sqlite_module(encrypted: bool):
    """Return the sqlite3-compatible module for the mode (pure helper, unit-testable).

    encrypted=True  -> ``sqlcipher3`` (raises ImportError if not installed).
    encrypted=False -> ``pysqlite3`` if present, else stdlib ``sqlite3`` (legacy).
    """
    if encrypted:
        import sqlcipher3
        return sqlcipher3
    try:
        import pysqlite3
        return pysqlite3
    except ImportError:
        import sqlite3
        return sqlite3


_ENCRYPTED_BINDING_ERROR = None
try:
    sqlite3 = _select_sqlite_module(env_encryption_on())
except ImportError as _e:
    # Encryption requested but sqlcipher3 is missing. Fall back to the plaintext
    # binding so import never hard-fails, but record the error so the provider can
    # refuse to run (rather than silently persist an UNencrypted DB).
    _ENCRYPTED_BINDING_ERROR = _e
    import sqlite3  # type: ignore[no-redef]

_SQLITE_BINDING = sqlite3.__name__


def encrypted_binding_active() -> bool:
    """True iff the live binding is SQLCipher (connections honour ``PRAGMA key``)."""
    return _SQLITE_BINDING == "sqlcipher3"


def sqlite_binding_error():
    """ImportError from a requested-but-unavailable encrypted binding, else None."""
    return _ENCRYPTED_BINDING_ERROR

ensure_plugin_on_path()

try:
    import holographic as hrr
    _HRR_AVAILABLE = hrr._HAS_NUMPY
except Exception as e:
    logger.warning("HRR (holographic) not available: %s. Compositional algebra disabled.", e)
    hrr = None            # keep the name bound so guarded references can't NameError
    _HRR_AVAILABLE = False

try:
    from entity_extractor import extract_entities as _extract_entities_fn
    logger.info(
        "EntityExtractor loaded (spaCy NER loads lazily on first extraction; "
        "regex layer always active)."
    )
    _ENTITY_EXTRACTOR_AVAILABLE = True
except Exception as e:
    logger.warning(
        "entity_extractor.py not found or failed to load: %s. "
        "Falling back to legacy regex extraction.", e
    )
    _extract_entities_fn = None
    _ENTITY_EXTRACTOR_AVAILABLE = False


def serialize_vector(vector: List[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)
