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


def norm_cat(category):
    """Open-ended label -> safe token. Categories are samples, not a closed enum."""
    return re.sub(r"[^a-z0-9_]", "", re.sub(r"[\s-]+", "_", (category or "").strip().lower())) or "concept"


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
                "category": {"type": "string",
                             "description": f"a short lowercase label (spaces/dashes become underscores). "
                                            f"Examples: {CATS}. Any label is allowed; these are samples, not a closed set."}},
            "required": ["content", "category"]}


def _manage_schema():
    return {"type": "object",
            "properties": {
                "id": {"type": "integer",
                       "description": "the fact id to target (from a prior candidate list); preferred when known"},
                "content": {"type": "string",
                            "description": "the fact text as it appears in your memory; used to locate the fact when you have no id"}},
            "required": []}


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
    {"name": "rlm_forget",
     "description": ("Prune a fact from your Resonant Lattice Memory (the inverse of rlm_remember). "
                     "Pass the fact's `id` if you have it, else its `content` as shown in your memory. "
                     "Ambiguous text returns candidate ids to retry with; a delete is never done on a "
                     "fuzzy match. Effect is immediate in the lattice; your projected memory updates "
                     "next session."),
     "inputSchema": _manage_schema()},
    {"name": "rlm_unpin",
     "description": ("Remove the [PRIORITY] authority from a pinned fact but keep the fact (the "
                     "inverse of rlm_pin). Pass the fact's `id` or its `content`. Use when a standing "
                     "rule no longer applies but the fact itself is still true and worth keeping."),
     "inputSchema": _manage_schema()},
]


def do_write(content, category, pin):
    cat = norm_cat(category)
    cmd = f"{C.REMOTE_PY} {C.REMOTE_DIR}/rlm_write.py --category {cat}" + (" --pin" if pin else "")
    try:
        # send stdin as explicit UTF-8 bytes; text=True would use the Windows locale (cp1252)
        # and mangle non-ASCII (e.g. an em-dash -> 0x97) into invalid UTF-8 on the node.
        r = subprocess.run(["ssh", *C.SSH_OPTS, C.SSH_HOST, cmd], input=content.encode("utf-8"),
                           capture_output=True, timeout=90)
        out = (r.stdout or b"").decode("utf-8", "replace").strip()
        err = (r.stderr or b"").decode("utf-8", "replace")
        return json.loads(out.splitlines()[-1]) if out else {"ok": False, "error": (err or "no output")[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def do_manage(op, fid, content):
    cmd = f"{C.REMOTE_PY} {C.REMOTE_DIR}/rlm_manage.py --op {op}"
    inp = b""
    if fid:
        cmd += f" --id {int(fid)}"
    else:
        inp = (content or "").encode("utf-8")
    try:
        r = subprocess.run(["ssh", *C.SSH_OPTS, C.SSH_HOST, cmd], input=inp,
                           capture_output=True, timeout=90)
        out = (r.stdout or b"").decode("utf-8", "replace").strip()
        err = (r.stderr or b"").decode("utf-8", "replace")
        return json.loads(out.splitlines()[-1]) if out else {"ok": False, "error": (err or "no output")[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def send_tool(mid, text, is_error):
    send({"jsonrpc": "2.0", "id": mid, "result": {
        "content": [{"type": "text", "text": text}], "isError": is_error}})


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

            if not C.configured():
                send_tool(mid, "error: rlm-grok.conf not configured", True)
                continue

            if name in ("rlm_pin", "rlm_remember"):
                content = (a.get("content") or "").strip()
                pin = (name == "rlm_pin")
                if not content:
                    send_tool(mid, "error: empty content", True)
                    continue
                res = do_write(content, a.get("category") or "concept", pin)
                if res.get("ok"):
                    txt = (f"{'Pinned' if pin else 'Wrote'} fact #{res.get('id')} ({res.get('action')}) "
                           f"[{res.get('category')}]" + (" as [PRIORITY]" if res.get("pinned") else "")
                           + f": {content[:80]}")
                else:
                    txt = f"write failed: {res.get('error')}"
                log(f"tools/call {name} -> {res}")
                send_tool(mid, txt, not res.get("ok"))
            elif name in ("rlm_forget", "rlm_unpin"):
                op = "forget" if name == "rlm_forget" else "unpin"
                fid = a.get("id")
                content = (a.get("content") or "").strip()
                if not fid and not content:
                    send_tool(mid, "error: provide id or content to target a fact", True)
                    continue
                res = do_manage(op, fid, content)
                if res.get("ok"):
                    if op == "forget":
                        txt = (f"Forgot fact #{res.get('id')}"
                               + (" (was [PRIORITY])" if res.get("was_pinned") else "")
                               + f": {res.get('content', '')}")
                    else:
                        txt = ((f"Fact #{res.get('id')} was already unpinned" if res.get("already")
                                else f"Unpinned fact #{res.get('id')}")
                               + f": {res.get('content', '')}")
                elif res.get("candidates"):
                    lines = "; ".join(f"#{c['id']} {c['content']}" for c in res["candidates"])
                    txt = f"{res.get('error')}. Candidates: {lines}"
                else:
                    txt = f"{op} failed: {res.get('error')}"
                log(f"tools/call {name} -> {res}")
                send_tool(mid, txt, not res.get("ok"))
            else:
                send_tool(mid, f"error: unknown tool {name}", True)
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
