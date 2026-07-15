#!/usr/bin/env python3
"""RLM memory-write MCP server (stdio, dependency-free).

Exposes two tools so grok can durably write to the Resonant Lattice Memory:
  - rlm_pin      : write a fact/decision AND pin it as [PRIORITY] authority (never forgotten)
  - rlm_remember : write a durable fact (not pinned)
Each call SSHes to the node's rlm_write.py, which embeds + dedups + provenance-tags the fact
(source grok:direct-write) so the lattice stays clean. Speaks newline-delimited JSON-RPC 2.0.
Connection settings come from rlm_grok_conf (~/.grok/rlm-grok.conf).
"""
import sys, os, json, re, subprocess, time

# avoid Windows \r\n corrupting the JSON-RPC framing
try:
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlm_grok_conf as C  # noqa: E402

LOG = os.path.join(os.path.expanduser("~/.grok/rlm-queue"), "mcp.log")
CATS = ("decision, constraint, dial, config, host_topology, model_spec, path_ref, "
        "procedure, eval_result, training_run, failed_probe, concept")


def log(m):
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {m}\n")
    except Exception:
        pass


def _schema():
    return {"type": "object",
            "properties": {
                "content": {"type": "string",
                            "description": "the atomic fact or decision, self-contained and quotable"},
                "category": {"type": "string", "description": f"one of: {CATS}"}},
            "required": ["content", "category"]}


TOOLS = [
    {"name": "rlm_pin",
     "description": ("Pin a durable fact or decision into your Resonant Lattice Memory as "
                     "[PRIORITY] authority: never forgotten, surfaced first, obeyed over conflicting "
                     "notes. Use for standing decisions and hard rules to hold across ALL future sessions."),
     "inputSchema": _schema()},
    {"name": "rlm_remember",
     "description": ("Write a durable fact into your Resonant Lattice Memory (not pinned). Use for "
                     "facts worth remembering across sessions that are not hard rules. Deduplicates "
                     "against existing memory automatically."),
     "inputSchema": _schema()},
]


def do_write(content, category, pin):
    cat = re.sub(r"[^a-z_]", "", (category or "").lower()) or "concept"
    cmd = f"{C.REMOTE_PY} {C.REMOTE_DIR}/rlm_write.py --category {cat}" + (" --pin" if pin else "")
    try:
        r = subprocess.run(["ssh", *C.SSH_OPTS, C.SSH_HOST, cmd], input=content,
                           capture_output=True, text=True, timeout=90)
        out = (r.stdout or "").strip()
        return json.loads(out.splitlines()[-1]) if out else {"ok": False, "error": (r.stderr or "no output")[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
    log("mcp server started")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        method, mid = msg.get("method"), msg.get("id")

        if method == "initialize":
            pv = (msg.get("params") or {}).get("protocolVersion") or "2025-06-18"
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": pv, "capabilities": {"tools": {}},
                "serverInfo": {"name": "rlm-memory", "version": "1.0.0"}}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            p = msg.get("params") or {}
            name, a = p.get("name"), (p.get("arguments") or {})
            content = (a.get("content") or "").strip()
            pin = (name == "rlm_pin")
            if not content:
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "error: empty content"}], "isError": True}})
                continue
            if not C.configured():
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "error: rlm-grok.conf not configured"}], "isError": True}})
                continue
            res = do_write(content, a.get("category") or "concept", pin)
            if res.get("ok"):
                txt = (f"{'Pinned' if pin else 'Wrote'} fact #{res.get('id')} ({res.get('action')}) "
                       f"[{res.get('category')}]" + (" as [PRIORITY]" if res.get("pinned") else "")
                       + f": {content[:80]}")
            else:
                txt = f"write failed: {res.get('error')}"
            log(f"tools/call {name} -> {res}")
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": txt}], "isError": not res.get("ok")}})
        elif method == "ping":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif method and method.startswith("notifications/"):
            pass
        else:
            if mid is not None:
                send({"jsonrpc": "2.0", "id": mid,
                      "error": {"code": -32601, "message": f"method not found: {method}"}})


if __name__ == "__main__":
    main()
