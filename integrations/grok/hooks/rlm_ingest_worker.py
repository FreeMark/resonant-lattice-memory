#!/usr/bin/env python3
"""Detached worker (not time-critical): ship a transcript snapshot to the RLM node and launch
the ingest there under nohup, so the multi-minute cycle survives this worker and the ssh ending.
Connection settings come from rlm_grok_conf (~/.grok/rlm-grok.conf).
"""
import sys, os, time, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlm_grok_conf as C  # noqa: E402

LOG = os.path.join(os.path.expanduser("~/.grok/rlm-queue"), "dispatch.log")


def log(m):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [worker] {m}\n")
    except Exception:
        pass


if len(sys.argv) < 3:
    log("worker: missing args (snap, sid)")
    sys.exit(1)
if not C.configured():
    log("worker: rlm-grok.conf not configured (need RLM_DIR + RLM_PY; add SSH_HOST/SSH_KEY for a "
        "remote node) -- snapshot kept")
    sys.exit(1)

snap, sid = sys.argv[1], sys.argv[2]
name = os.path.basename(snap)
remote = f"{C.REMOTE_DIR}/incoming/{name}"
ingest_sid = f"{sid}_{time.strftime('%H%M%S')}"

try:
    ok, err = C.push(snap, f"incoming/{name}")        # local copy or scp, per transport
    if not ok:
        log(f"ship FAILED: {err} (snapshot kept in queue)")
        sys.exit(1)
    log(f"ship ok -> {remote} ({'local' if C.LOCAL else 'scp'})")

    cmd = (f"nohup {C.REMOTE_PY} {C.REMOTE_DIR}/rlm_ingest.py {remote} "
           f"--session-id {ingest_sid} >> {C.REMOTE_DIR}/ingest.log 2>&1 &")
    r = C.run(cmd, timeout=60, text=True)             # backgrounded locally or over ssh
    log(f"ingest launched sid={ingest_sid} ({'local' if C.LOCAL else 'ssh'}) rc={r.returncode} "
        f"{(r.stderr or '')[:150].strip()}")
except Exception as e:
    log(f"worker error: {e}")
