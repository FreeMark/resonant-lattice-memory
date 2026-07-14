#!/usr/bin/env python3
"""grok PreCompact hook -> RLM ingest (fast, fail-open).

Compaction REWRITES chat_history.jsonl (collapses it to a short summary), so we SNAPSHOT the
transcript synchronously here, BEFORE compaction, then detach a worker that ships the snapshot
to the node and launches the (multi-minute) ingest there. This hook only: read stdin -> copy
file -> spawn detached worker -> exit 0. Errors are swallowed (never block the user's compaction).
"""
import sys, os, json, time, shutil, subprocess, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE = os.path.expanduser("~/.grok/rlm-queue")
os.makedirs(QUEUE, exist_ok=True)
LOG = os.path.join(QUEUE, "dispatch.log")


def log(msg):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
    except Exception:
        pass


try:
    raw = sys.stdin.read()
    j = json.loads(raw) if raw.strip() else {}
except Exception as e:
    j = {}
    log(f"stdin parse error: {e}")

sid = j.get("sessionId") or os.environ.get("GROK_SESSION_ID", "unknown")

tpath = j.get("transcriptPath")
if not tpath:
    cwd = j.get("cwd") or os.environ.get("GROK_WORKSPACE_ROOT", "").replace("/", "\\").rstrip("\\")
    if cwd and sid:
        enc = urllib.parse.quote(cwd, safe="")
        tpath = os.path.expanduser(f"~/.grok/sessions/{enc}/{sid}/chat_history.jsonl")

if not tpath or not os.path.exists(tpath):
    log(f"PreCompact sid={sid}: transcript not found ({tpath}) -- skip")
    sys.exit(0)

ts = time.strftime("%Y%m%d-%H%M%S")
snap = os.path.join(QUEUE, f"{sid}.{ts}.jsonl")
try:
    shutil.copy2(tpath, snap)  # snapshot before compaction collapses the original
    log(f"PreCompact sid={sid}: snapshot {os.path.getsize(snap)}B -> {os.path.basename(snap)}")
except Exception as e:
    log(f"PreCompact sid={sid}: snapshot FAILED: {e}")
    sys.exit(0)

# detach the slow scp+ssh worker so this hook returns immediately (cross-platform)
kwargs = dict(close_fds=True, stdin=subprocess.DEVNULL,
              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if os.name == "nt":
    kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
else:
    kwargs["start_new_session"] = True
try:
    subprocess.Popen([sys.executable, os.path.join(HERE, "rlm_ingest_worker.py"), snap, sid], **kwargs)
    log(f"PreCompact sid={sid}: dispatched worker for {os.path.basename(snap)}")
except Exception as e:
    log(f"PreCompact sid={sid}: worker spawn FAILED: {e}")

sys.exit(0)
