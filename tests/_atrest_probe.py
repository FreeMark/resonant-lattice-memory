r"""_atrest_probe.py - child process for the real at-rest opacity check.

Must be launched with RESONANT_LATTICE_DB_ENCRYPTED=1 ALREADY in the environment
(the sqlite binding is selected once, at store_common import time). Derives a DB
key the way the provider does (crypto_keys), writes one fact through an encrypted
LatticeStore, closes it, and reports whether the plaintext leaks into the raw DB
bytes. Prints a single JSON line; uses a fake embedding so no Ollama is needed.
"""
import json
import os
import sys
import tempfile

PLUGIN = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "resonant_lattice"))
sys.path.insert(0, PLUGIN)


def run():
    import store_common
    if not store_common.encrypted_binding_active():
        return {"ok": False, "error": "encrypted binding not active (sqlcipher3 missing or env unset)"}
    import crypto_keys
    from store import LatticeStore

    passphrase = b"test-passphrase-123"
    keystore = crypto_keys.create_keystore(passphrase)
    db_key = crypto_keys.derive_db_key(passphrase, keystore)

    db = os.path.join(tempfile.mkdtemp(), "atrest.db")
    s = LatticeStore(db_path=db, vector_dim=8, db_key=db_key)
    secret = "Acme hosting charge 4050 cents"
    s.add_or_reinforce_fact(secret, [0.1] * 8, "spend", "p", entities=["acme"])
    s.close()

    with open(db, "rb") as f:
        raw = f.read()
    return {
        "ok": True,
        "size": len(raw),
        "acme_in_bytes": b"Acme" in raw,
        "amt_in_bytes": b"4050" in raw,
        "header": raw[:16].hex(),   # SQLite plaintext starts 'SQLite format 3\0'
    }


if __name__ == "__main__":
    try:
        print(json.dumps(run()))
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
