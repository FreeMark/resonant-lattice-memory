"""GPU blind-recall backend for resonant_lattice (openfhe-gpu-backend integration).

A drop-in accelerator for ``BlindRetriever.blind_scores``: it encrypts the query with the
SAME ``BlindRecallPRE`` context the store already holds, scores the encrypted ``semantic_he``
corpus on the GPU via the native FIDESlib / OpenFHE scorer, and returns ``[(fact_id, cosine),
...]`` identical in shape to the CPU path. Any failure raises, so the caller falls back to the
CPU scan.

Two backends, tried in order:
  * a persistent daemon (``rlm_gpu_recalld``) that holds the whole encrypted corpus resident on
    the GPU and answers each query over a Unix socket in about 7 s (preferred);
  * a one-shot binary (``rlm_gpu_recall``) that re-bridges the corpus per call (about 29 s).

Trust boundary (unchanged from the blind store): the GPU (untrusted evaluator) receives only
public material - the crypto context, storage public key, and public eval/rotation keys - plus
the encrypted corpus and the encrypted query. The master secret is used only to decrypt the
already-homomorphically-computed scalar scores on the trusted host; the GPU never sees a secret
and never decrypts.
"""
import json
import logging
import os
import socket
import struct
import subprocess
import tempfile

logger = logging.getLogger(__name__)

try:
    import openfhe as _ofhe
except Exception:  # openfhe not installed (e.g. non-node dev host) -> backend simply unavailable
    _ofhe = None


def _recvall(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("daemon closed the connection early")
        buf += chunk
    return buf


class GPUBlindBackend:
    """Wraps the native GPU scorer(s) behind a ``.available()`` / ``.scores()`` interface."""

    def __init__(self, blind, db_path, binary=None, *, socket_path=None,
                 workdir=None, topk=0, timeout=600):
        self._blind = blind          # a BlindRecallPRE (carries _cc / _pub / _sk + encrypt_unit_vector)
        self._db = db_path
        self._binary = binary        # one-shot rlm_gpu_recall (optional if a daemon is up)
        self._socket_path = socket_path   # rlm_gpu_recalld Unix socket (preferred if reachable)
        self._topk = topk            # 0 = return all scores (full parity with the CPU list)
        self._timeout = timeout
        self._workdir = workdir or tempfile.mkdtemp(prefix="rlm_gpu_")
        self._keys_exported = False

    # ── availability ────────────────────────────────────────────────────────────
    def _daemon_up(self):
        if not self._socket_path:
            return False
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect(self._socket_path)
            s.close()
            return True
        except Exception:
            return False

    def available(self):
        """True iff the query can be encrypted here AND some GPU backend can score it."""
        if not (_ofhe and self._blind is not None and getattr(self._blind, "can_decrypt", False)
                and os.path.exists(self._db)):
            return False
        if self._daemon_up():
            return True                      # daemon already holds the corpus + a GPU
        if self._binary and os.path.exists(self._binary):
            try:
                subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10, check=True)
                return True
            except Exception:
                return False
        return False

    # ── scoring ───────────────────────────────────────────────────────────────────
    def scores(self, query_vec):
        """Return ``[(fact_id, cosine), ...]`` (score-descending) computed on the GPU."""
        qbytes = self._blind.encrypt_unit_vector(query_vec)   # serialized query ciphertext
        if self._daemon_up():
            return self._daemon_scores(qbytes)
        return self._oneshot_scores(qbytes)

    def _daemon_scores(self, qbytes):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self._timeout)
        s.connect(self._socket_path)
        try:
            s.sendall(struct.pack("<II", int(self._topk), len(qbytes)) + qbytes)
            (jlen,) = struct.unpack("<I", _recvall(s, 4))
            out = json.loads(_recvall(s, jlen).decode())
        finally:
            s.close()
        if "error" in out:
            raise RuntimeError(f"rlm_gpu_recalld: {out['error']}")
        return [(int(i), float(x)) for i, x in out["top"]]

    def _export_keys(self):
        if self._keys_exported:
            return
        b = self._blind
        _ofhe.SerializeToFile(os.path.join(self._workdir, "cc.bin"), b._cc, _ofhe.BINARY)
        _ofhe.SerializeToFile(os.path.join(self._workdir, "pk.bin"), b._pub, _ofhe.BINARY)
        _ofhe.SerializeToFile(os.path.join(self._workdir, "sk.bin"), b._sk, _ofhe.BINARY)
        self._keys_exported = True

    def _oneshot_scores(self, qbytes):
        self._export_keys()
        qpath = os.path.join(self._workdir, "q.bin")
        with open(qpath, "wb") as f:
            f.write(qbytes)
        cmd = [self._binary, self._workdir, self._db, str(self._topk), qpath]
        p = subprocess.run(cmd, capture_output=True, timeout=self._timeout)
        if p.returncode != 0:
            raise RuntimeError(f"rlm_gpu_recall exit {p.returncode}: {p.stderr.decode()[-400:]}")
        # the native lib prints a device banner to stdout ahead of our JSON; take the JSON line
        jlines = [ln for ln in p.stdout.decode().splitlines() if ln.lstrip().startswith("{")]
        if not jlines:
            raise RuntimeError("rlm_gpu_recall produced no JSON output")
        return [(int(i), float(s)) for i, s in json.loads(jlines[-1])["queries"][0]["top"]]
