"""GPU blind-recall backend for resonant_lattice (openfhe-gpu-backend integration).

A drop-in accelerator for ``BlindRetriever.blind_scores``: it encrypts the query with the
SAME ``BlindRecallPRE`` context the store already holds, invokes the native FIDESlib /
OpenFHE GPU scorer (``rlm_gpu_recall``) over the encrypted ``semantic_he`` corpus, and
returns ``[(fact_id, cosine), ...]`` identical in shape to the CPU path. Any failure raises,
so the caller falls back to the CPU scan.

Trust boundary (unchanged from the blind store): the GPU (untrusted evaluator) receives only
public material — the crypto context, storage public key, and public eval/rotation keys —
plus the encrypted corpus and the encrypted query. The master secret is used only to decrypt
the already-homomorphically-computed scalar scores on the trusted host. The GPU never sees a
secret and never decrypts.
"""
import json
import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)

try:
    import openfhe as _ofhe
except Exception:  # openfhe not installed (e.g. non-node dev host) -> backend simply unavailable
    _ofhe = None


class GPUBlindBackend:
    """Wraps the native GPU scorer behind a ``.available()`` / ``.scores()`` interface."""

    def __init__(self, blind, db_path, binary, *, workdir=None, topk=0, timeout=600):
        self._blind = blind          # a BlindRecallPRE (carries _cc / _pub / _sk + encrypt_unit_vector)
        self._db = db_path
        self._binary = binary
        self._topk = topk            # 0 = return all scores (full parity with the CPU list)
        self._timeout = timeout
        self._workdir = workdir or tempfile.mkdtemp(prefix="rlm_gpu_")
        self._keys_exported = False
        self._ok = None

    def available(self):
        """True iff openfhe is importable, the binary and DB exist, and a CUDA GPU is present."""
        if self._ok is not None:
            return self._ok
        ok = bool(_ofhe) and os.path.exists(self._binary) and os.path.exists(self._db) \
            and self._blind is not None and getattr(self._blind, "can_decrypt", False)
        if ok:
            try:
                subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10, check=True)
            except Exception:
                ok = False
        self._ok = ok
        return ok

    def _export_keys(self):
        if self._keys_exported:
            return
        b = self._blind
        _ofhe.SerializeToFile(os.path.join(self._workdir, "cc.bin"), b._cc, _ofhe.BINARY)
        _ofhe.SerializeToFile(os.path.join(self._workdir, "pk.bin"), b._pub, _ofhe.BINARY)
        _ofhe.SerializeToFile(os.path.join(self._workdir, "sk.bin"), b._sk, _ofhe.BINARY)
        self._keys_exported = True

    def scores(self, query_vec):
        """Return ``[(fact_id, cosine), ...]`` (score-descending) computed on the GPU."""
        self._export_keys()
        qpath = os.path.join(self._workdir, "q.bin")
        with open(qpath, "wb") as f:
            f.write(self._blind.encrypt_unit_vector(query_vec))   # serialized query ciphertext
        cmd = [self._binary, self._workdir, self._db, str(self._topk), qpath]
        p = subprocess.run(cmd, capture_output=True, timeout=self._timeout)
        if p.returncode != 0:
            raise RuntimeError(f"rlm_gpu_recall exit {p.returncode}: {p.stderr.decode()[-400:]}")
        # the native lib prints a device banner to stdout ahead of our JSON; take the JSON line
        jlines = [ln for ln in p.stdout.decode().splitlines() if ln.lstrip().startswith("{")]
        if not jlines:
            raise RuntimeError("rlm_gpu_recall produced no JSON output")
        out = json.loads(jlines[-1])
        return [(int(i), float(s)) for i, s in out["queries"][0]["top"]]
