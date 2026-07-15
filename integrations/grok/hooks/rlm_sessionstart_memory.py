#!/usr/bin/env python3
"""grok SessionStart hook -> refresh grok's native memory from the RLM lattice. RLM is sole writer.

SCOPE (env RLM_MEMORY_SCOPE, default 'workspace'; set it in the hook JSON's "env" to switch):
  workspace : write the projection into the per-repo grok memory dir's MEMORY.md (overwrite).
              grok keys workspace memory by git ORIGIN, so it is shared across all clones/locations
              of the same repo; each repo you want the lattice in must be bootstrapped once.
  global    : write the projection as a preserved managed block into grok's GLOBAL
              ~/.grok/memory/MEMORY.md, so the lattice surfaces in EVERY grok session (any repo),
              with no per-repo bootstrap. Other global content outside the block is preserved.
Fail-open: any error leaves the last projection in place.
"""
import sys, os, json, subprocess, time, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlm_grok_conf as C  # noqa: E402

MEMROOT = os.path.expanduser("~/.grok/memory")
LOG = os.path.join(os.path.expanduser("~/.grok/rlm-queue"), "memory.log")
REMOTE = f"{C.REMOTE_DIR}/rlm_export_memory.py"
SCOPE = os.environ.get("RLM_MEMORY_SCOPE", "workspace").lower()
BEGIN = "<!-- RLM:BEGIN (auto-generated from the lattice; do not edit inside this block) -->"
END = "<!-- RLM:END -->"


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
    idx = glob.glob(os.path.join(MEMROOT, "*", "index.sqlite"))
    return os.path.dirname(idx[0]) if len(idx) == 1 else None


def atomic_write(target, text):
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, target)


if not C.configured():
    log("rlm-grok.conf not configured -- skip")
    sys.exit(0)

if SCOPE == "global":
    try:
        sys.stdin.read()
    except Exception:
        pass
    os.makedirs(MEMROOT, exist_ok=True)
    target = os.path.join(MEMROOT, "MEMORY.md")
else:
    memdir = resolve_memdir(resolve_ws())
    if not memdir:
        log("workspace scope: no memory dir (run bootstrap.sh in the repo once) -- skip")
        sys.exit(0)
    target = os.path.join(memdir, "MEMORY.md")

try:
    r = subprocess.run(["ssh", *C.SSH_OPTS, C.SSH_HOST, f"{C.REMOTE_PY} {REMOTE} 2>/dev/null"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    projection = (r.stdout or "").strip()
except Exception as e:
    log(f"export ssh error: {e} (kept existing)")
    sys.exit(0)
if not projection or "# Project Memory" not in projection:
    log("export empty/invalid (kept existing)")
    sys.exit(0)

try:
    if SCOPE == "global":
        block = f"{BEGIN}\n{projection}\n{END}\n"
        existing = open(target, encoding="utf-8", errors="replace").read() if os.path.exists(target) else ""
        if BEGIN in existing and END in existing:
            pre = existing.split(BEGIN)[0].rstrip()
            post = existing.split(END, 1)[1].lstrip("\n")
            newtext = (pre + "\n\n" if pre.strip() else "") + block + ("\n" + post if post.strip() else "")
        else:
            newtext = (existing.rstrip() + "\n\n" if existing.strip() else "") + block
        atomic_write(target, newtext)
    else:
        atomic_write(target, projection + "\n")
    log(f"scope={SCOPE}: wrote {len(projection)}B RLM projection -> {target}")
except Exception as e:
    log(f"write error: {e}")

sys.exit(0)
