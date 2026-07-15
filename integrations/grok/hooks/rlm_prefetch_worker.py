#!/usr/bin/env python3
"""Detached prefetch worker: run a query-conditioned recall against the RLM node using the user's
actual message as the query, and write a <resonant_memory> block to the local prefetch file that
the rlm_prefetch MCP tool serves instantly. Recall here is computed WITHOUT reinforcement; facts
strengthen only when the agent actually consumes the block (rlm_prefetch fires rlm_reinforce.py),
so resonance tracks real use, not the mere firing of the hook. Fail-open: any error leaves the
previous block in place (staleness is stamped in the header, so the reader can judge).
Connection settings come from rlm_grok_conf (~/.grok/rlm-grok.conf).
"""
import sys, os, json, time, hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlm_grok_conf as C  # noqa: E402

QUEUE = os.path.expanduser("~/.grok/rlm-queue")
LOG = os.path.join(QUEUE, "prefetch.log")
K = 8                 # resonant-block size (mirrors the hermes prefetch top-k)
QUERY_CAP = 2000      # embed the head of a very long prompt, not the whole paste


def log(m):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [worker] {m}\n")
    except Exception:
        pass


def build_block(hits):
    lines = ["<resonant_memory>",
             "(recalled from your lattice for the CURRENT user message; soft and fallible "
             "memories, not verbatim law -- verify before relying on them)"]
    for h in hits:
        pin = "[PRIORITY] " if h.get("pinned") else ""
        content = " ".join(str(h.get("content", "")).split())
        lines.append(f"- {pin}[{h.get('category')}] {content} "
                     f"(#{h.get('id')}, score {h.get('score')}, res {h.get('resonance')})")
    lines.append("</resonant_memory>")
    return "\n".join(lines)


if len(sys.argv) < 4:
    log("worker: missing args (qfile, wsid, sid)")
    sys.exit(1)
if not C.configured():
    log("worker: rlm-grok.conf not configured -- skip")
    sys.exit(1)

qfile, wsid, sid = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    query = open(qfile, encoding="utf-8", errors="replace").read().strip()
except Exception as e:
    log(f"worker: query read FAILED: {e}")
    sys.exit(1)
if not query:
    sys.exit(0)

t0 = time.time()
try:
    r = C.run(f"{C.REMOTE_PY} {C.REMOTE_DIR}/rlm_search.py --k {K} --no-reinforce",
              input=query[:QUERY_CAP].encode("utf-8"), timeout=90)
    out = (r.stdout or b"").decode("utf-8", "replace").strip()
    res = json.loads(out.splitlines()[-1]) if out else {"ok": False, "error": "no output"}
except Exception as e:
    res = {"ok": False, "error": str(e)[:200]}

if not res.get("ok"):
    log(f"worker sid={sid} ws={wsid}: recall FAILED: {res.get('error')} (previous block kept)")
    sys.exit(1)

hits = res.get("hits") or []
ids = ",".join(str(h.get("id")) for h in hits)
qsha = hashlib.sha256(query.encode("utf-8", "replace")).hexdigest()[:12]
excerpt = " ".join(query.split())[:140]
body = build_block(hits) if hits else \
    "(no relevant memories surfaced for the current message)"

text = (f"<!-- rlm-prefetch v1 sid={sid} ts={int(time.time())} qsha={qsha} ids={ids} -->\n"
        f"conditioned on: \"{excerpt}\"\n\n{body}\n")

mdfile = os.path.join(QUEUE, "prefetch", f"{wsid}.md")
try:
    tmp = mdfile + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, mdfile)
    log(f"worker sid={sid} ws={wsid}: {len(hits)} hits in {time.time() - t0:.1f}s -> "
        f"{os.path.basename(mdfile)}")
except Exception as e:
    log(f"worker sid={sid} ws={wsid}: block write FAILED: {e}")
