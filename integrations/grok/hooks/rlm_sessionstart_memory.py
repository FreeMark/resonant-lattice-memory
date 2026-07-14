#!/usr/bin/env python3
"""grok SessionStart hook -> refresh grok's native workspace MEMORY.md from the RLM lattice.

RLM is the SOLE writer of grok's memory. This pulls the lattice projection from the node and
writes it into ~/.grok/memory/<workspace-slug>/MEMORY.md (overwriting). Grok's native memory
engine reindexes on the next memory search (first-turn injection) and surfaces it. Fail-open.

Dir resolution: grok names the workspace memory dir <reponame>-<hash>. Grok's hash is not
reproducible externally, so we glob <reponame>-* (reponame from the git origin), falling back to
the single memory dir that has an index.sqlite. Run bootstrap.sh once per repo so the dir exists.
"""
import sys, os, json, subprocess, time, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlm_grok_conf as C  # noqa: E402

MEMROOT = os.path.expanduser("~/.grok/memory")
LOG = os.path.join(os.path.expanduser("~/.grok/rlm-queue"), "memory.log")
REMOTE = f"{C.REMOTE_DIR}/rlm_export_memory.py"


def log(msg):
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
    except Exception:
        pass


def resolve_ws():
    raw = ""
    try:
        raw = sys.stdin.read()
    except Exception:
        pass
    try:
        j = json.loads(raw) if raw.strip() else {}
    except Exception:
        j = {}
    return (j.get("workspaceRoot") or os.environ.get("GROK_WORKSPACE_ROOT")
            or j.get("cwd") or os.getcwd())


def reponame_of(ws):
    try:
        url = subprocess.run(["git", "-C", ws, "remote", "get-url", "origin"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        if url:
            base = url.rstrip("/").split("/")[-1]
            return base[:-4] if base.endswith(".git") else base
    except Exception:
        pass
    return None


def resolve_memdir(ws):
    rn = reponame_of(ws)
    if rn:
        hits = [d for d in glob.glob(os.path.join(MEMROOT, rn + "-*")) if os.path.isdir(d)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            log(f"ambiguous memory dirs for reponame={rn}: {hits}")
    idx = glob.glob(os.path.join(MEMROOT, "*", "index.sqlite"))
    if len(idx) == 1:
        return os.path.dirname(idx[0])
    log(f"could not resolve memory dir (reponame={rn}, index dirs={len(idx)})")
    return None


if not C.configured():
    log("rlm-grok.conf not configured -- skip")
    sys.exit(0)

ws = resolve_ws()
memdir = resolve_memdir(ws)
if not memdir:
    log(f"SessionStart ws={ws}: no memory dir (run bootstrap.sh in the repo once) -- skip")
    sys.exit(0)

target = os.path.join(memdir, "MEMORY.md")
try:
    r = subprocess.run(["ssh", *C.SSH_OPTS, C.SSH_HOST, f"{C.REMOTE_PY} {REMOTE} 2>/dev/null"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    projection = (r.stdout or "").strip()
except Exception as e:
    log(f"SessionStart ws={ws}: export ssh error: {e} (kept existing MEMORY.md)")
    sys.exit(0)

if not projection or "# Project Memory" not in projection:
    log(f"SessionStart ws={ws}: export empty/invalid (kept existing MEMORY.md)")
    sys.exit(0)

try:
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(projection + "\n")
    os.replace(tmp, target)  # atomic
    log(f"SessionStart ws={ws}: wrote {len(projection)}B RLM projection -> {target}")
except Exception as e:
    log(f"SessionStart ws={ws}: write error: {e}")

sys.exit(0)
