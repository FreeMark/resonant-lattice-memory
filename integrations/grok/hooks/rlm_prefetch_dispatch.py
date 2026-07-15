#!/usr/bin/env python3
"""grok UserPromptSubmit hook -> per-turn RLM prefetch (fast, fail-open).

grok injects memory automatically only on the FIRST turn and after compaction; every turn in
between is pull-only. This hook closes that gap: on every submitted prompt it detaches a worker
that runs a query-conditioned recall against the lattice (the user's actual message as the query)
and writes a <resonant_memory> block to a local file. The rlm_prefetch MCP tool then serves that
block instantly (no node round-trip), so the agent gets hermes-style per-turn recall for the cost
of one argument-free tool call. grok ignores passive hooks' stdout, so a side-effect file is the
only seam; this hook only: read stdin -> guard -> write query file -> spawn detached worker -> exit 0.
"""
import sys, os, json, time, hashlib, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE = os.path.expanduser("~/.grok/rlm-queue")
PREFETCH = os.path.join(QUEUE, "prefetch")
os.makedirs(PREFETCH, exist_ok=True)
LOG = os.path.join(QUEUE, "prefetch.log")

# Trivial acks: recall for the PREVIOUS substantive prompt stays in place (it is still the topic).
TRIVIAL = {"y", "n", "yes", "no", "ok", "okay", "k", "go", "continue", "proceed", "approved",
           "thanks", "thank you", "sounds good", "do it", "yes please", "go ahead", "sure"}
PROXY_MIN_OVERLAP = 0.6     # lexical-overlap gate: same-topic prompts reuse the existing block
PROXY_FRESH_SECS = 600      # ...but only if the existing block is this fresh


def log(msg):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
    except Exception:
        pass


def ws_hash(path):
    norm = os.path.normcase(os.path.normpath(path or "unknown")).replace("\\", "/")
    return hashlib.sha256(norm.encode("utf-8", "replace")).hexdigest()[:12]


def tokens(s):
    return {t for t in "".join(c if c.isalnum() or c == "_" else " " for c in s.lower()).split()
            if len(t) > 2}


try:
    raw = sys.stdin.read()
    j = json.loads(raw) if raw.strip() else {}
except Exception as e:
    j = {}
    log(f"stdin parse error: {e}")

# The prompt field name is grok-internal; probe the plausible names (log keys for validation).
prompt = ""
for key in ("prompt", "userPrompt", "user_prompt", "text", "input", "message"):
    v = j.get(key)
    if isinstance(v, str) and v.strip():
        prompt = v.strip()
        break

sid = j.get("sessionId") or os.environ.get("GROK_SESSION_ID", "unknown")
ws = j.get("workspaceRoot") or j.get("cwd") or os.environ.get("GROK_WORKSPACE_ROOT") or os.getcwd()
wsid = ws_hash(ws)

if not prompt:
    log(f"sid={sid}: no prompt field (payload keys={sorted(j.keys())}) -- skip")
    sys.exit(0)
if prompt.startswith("/"):
    log(f"sid={sid}: slash command -- skip")
    sys.exit(0)
if len(prompt) < 15 or len(prompt.split()) < 3 or prompt.lower().rstrip(".!") in TRIVIAL:
    log(f"sid={sid}: trivial prompt ({len(prompt)}B) -- keep existing block")
    sys.exit(0)

qfile = os.path.join(PREFETCH, f"{wsid}.query.txt")
mdfile = os.path.join(PREFETCH, f"{wsid}.md")

# Same-topic gate (mirrors hermes _prefetch_proxy_ok): if the new prompt heavily overlaps the one
# the current block was computed from, and the block is fresh, keep it -- skip the node hit.
try:
    if os.path.exists(qfile) and os.path.exists(mdfile) \
            and (time.time() - os.path.getmtime(mdfile)) < PROXY_FRESH_SECS:
        prev = open(qfile, encoding="utf-8", errors="replace").read()
        new_t = tokens(prompt)
        if new_t and len(new_t & tokens(prev)) / len(new_t) >= PROXY_MIN_OVERLAP:
            log(f"sid={sid}: proxy ok (topic overlap) -- reuse existing block")
            sys.exit(0)
except Exception:
    pass

try:
    tmp = qfile + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(prompt)
    os.replace(tmp, qfile)
except Exception as e:
    log(f"sid={sid}: query write FAILED: {e}")
    sys.exit(0)

# detach the recall worker so this hook returns immediately (cross-platform)
kwargs = dict(close_fds=True, stdin=subprocess.DEVNULL,
              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if os.name == "nt":
    kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
else:
    kwargs["start_new_session"] = True
try:
    subprocess.Popen([sys.executable, os.path.join(HERE, "rlm_prefetch_worker.py"),
                      qfile, wsid, sid], **kwargs)
    log(f"sid={sid}: dispatched prefetch worker ws={wsid} ({len(prompt)}B prompt)")
except Exception as e:
    log(f"sid={sid}: worker spawn FAILED: {e}")

sys.exit(0)
