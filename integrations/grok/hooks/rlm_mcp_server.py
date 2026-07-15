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
    {"name": "rlm_search",
     "description": ("Semantic search over YOUR OWN Resonant Lattice Memory: live hybrid vector + "
                     "keyword recall, deeper and more relevant than the top-N projection you wake up "
                     "with. Read-only. Returns ranked facts with ids, categories, and relevance "
                     "scores. Use it to recall precisely on a topic instead of scanning the projection."),
     "inputSchema": {"type": "object",
                     "properties": {
                         "query": {"type": "string", "description": "what to recall"},
                         "k": {"type": "integer", "description": "max results (default 8)"}},
                     "required": ["query"]}},
    {"name": "external_rlm_search",
     "description": ("Search an EXTERNAL domain lattice: read-only reference knowledge from another "
                     "trained corpus (e.g. a web-dev fact base). Call with no query (or lattice='list') "
                     "to see which domain lattices are available, then search a named one. These are "
                     "reference corpora, NOT your own memory; results are labeled by source lattice."),
     "inputSchema": {"type": "object",
                     "properties": {
                         "query": {"type": "string", "description": "what to look up (omit to just list lattices)"},
                         "lattice": {"type": "string", "description": "domain lattice name, or 'list' to see available"},
                         "k": {"type": "integer", "description": "max results (default 8)"}},
                     "required": []}},
    {"name": "transfer_knowledge",
     "description": ("Import specific facts from an external domain lattice INTO your own memory, by "
                     "id. Find ids with external_rlm_search first, then transfer the ones worth "
                     "keeping. Each import is deduped against what you already know and tagged with its "
                     "source (import:<lattice>:<id>), so borrowed knowledge stays distinct from what "
                     "you learned firsthand. Facts start unpinned; pin later if one earns it."),
     "inputSchema": {"type": "object",
                     "properties": {
                         "lattice": {"type": "string", "description": "source domain lattice name"},
                         "ids": {"type": "array", "items": {"type": "integer"},
                                 "description": "fact ids to import (from an external_rlm_search result)"}},
                     "required": ["lattice", "ids"]}},
    {"name": "rlm_stats",
     "description": ("Read-only health snapshot of your OWN lattice: fact count, tier distribution "
                     "(short/mid/long), pins, memory + dream cycle clocks, episodes, entities, "
                     "relations, narratives, and pending conflicts. Use to check the state of your memory."),
     "inputSchema": {"type": "object", "properties": {}, "required": []}},
    {"name": "rlm_conflict",
     "description": ("Inspect and resolve contradictory memories in your OWN lattice. action='list' "
                     "shows pending (unresolved) conflict groups. action='resolve' with an id picks "
                     "that fact as the winner and retires the other members as superseded history. "
                     "action='dismiss' with a group_id marks the group a false positive (all members "
                     "kept). Conflicts are surfaced by the dream cycle; there may be none."),
     "inputSchema": {"type": "object",
                     "properties": {
                         "action": {"type": "string", "enum": ["list", "resolve", "dismiss"],
                                    "description": "list | resolve | dismiss"},
                         "id": {"type": "integer", "description": "winner fact id (for resolve)"},
                         "group_id": {"type": "string", "description": "conflict group id (for dismiss)"}},
                     "required": ["action"]}},
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


def _latt(name):
    """Sanitize a lattice name to a safe filename token (no path traversal / shell metachars)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "", (name or "")).strip(".")


def do_search(query, k, db_rel):
    db_arg = f" --db {C.REMOTE_DIR}/{db_rel}" if db_rel else ""
    cmd = f"{C.REMOTE_PY} {C.REMOTE_DIR}/rlm_search.py --k {int(k)}{db_arg}"
    try:
        r = subprocess.run(["ssh", *C.SSH_OPTS, C.SSH_HOST, cmd], input=query.encode("utf-8"),
                           capture_output=True, timeout=120)
        out = (r.stdout or b"").decode("utf-8", "replace").strip()
        err = (r.stderr or b"").decode("utf-8", "replace")
        return json.loads(out.splitlines()[-1]) if out else {"ok": False, "error": (err or "no output")[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def list_lattices():
    cmd = f"ls -1 {C.REMOTE_DIR}/lattices/*.db 2>/dev/null || true"
    try:
        r = subprocess.run(["ssh", *C.SSH_OPTS, C.SSH_HOST, cmd], capture_output=True, timeout=30)
        out = (r.stdout or b"").decode("utf-8", "replace")
        return [os.path.basename(x)[:-3] for x in out.splitlines() if x.strip().endswith(".db")]
    except Exception:
        return []


def do_import(lattice, ids):
    ids_str = ",".join(str(int(i)) for i in ids if str(i).lstrip("-").isdigit())
    cmd = (f"{C.REMOTE_PY} {C.REMOTE_DIR}/rlm_import.py "
           f"--lattice-db {C.REMOTE_DIR}/lattices/{lattice}.db --lattice-name {lattice} --ids '{ids_str}'")
    try:
        r = subprocess.run(["ssh", *C.SSH_OPTS, C.SSH_HOST, cmd], capture_output=True, timeout=180)
        out = (r.stdout or b"").decode("utf-8", "replace").strip()
        err = (r.stderr or b"").decode("utf-8", "replace")
        return json.loads(out.splitlines()[-1]) if out else {"ok": False, "error": (err or "no output")[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def do_stats():
    cmd = f"{C.REMOTE_PY} {C.REMOTE_DIR}/rlm_stats.py"
    try:
        r = subprocess.run(["ssh", *C.SSH_OPTS, C.SSH_HOST, cmd], capture_output=True, timeout=90)
        out = (r.stdout or b"").decode("utf-8", "replace").strip()
        err = (r.stderr or b"").decode("utf-8", "replace")
        return json.loads(out.splitlines()[-1]) if out else {"ok": False, "error": (err or "no output")[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def do_conflict(op, winner_id, group_id):
    cmd = f"{C.REMOTE_PY} {C.REMOTE_DIR}/rlm_conflicts.py --op {op}"
    if winner_id:
        cmd += f" --winner-id {int(winner_id)}"
    if group_id:
        gid = re.sub(r"[^A-Za-z0-9_.-]", "", str(group_id))
        if gid:
            cmd += f" --group-id {gid}"
    try:
        r = subprocess.run(["ssh", *C.SSH_OPTS, C.SSH_HOST, cmd], capture_output=True, timeout=90)
        out = (r.stdout or b"").decode("utf-8", "replace").strip()
        err = (r.stderr or b"").decode("utf-8", "replace")
        return json.loads(out.splitlines()[-1]) if out else {"ok": False, "error": (err or "no output")[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def fmt_stats(res):
    if not res.get("ok"):
        return f"stats failed: {res.get('error')}"
    tiers = ", ".join(f"{k}={v}" for k, v in (res.get("by_tier") or {}).items())
    return (f"lattice: {res.get('total_facts')} facts ({tiers}); {res.get('pinned')} pinned; "
            f"memory_cycle {res.get('memory_cycle')}, dream_cycle {res.get('dream_cycle')}; "
            f"{res.get('entities')} entities, {res.get('relations')} relations, "
            f"{res.get('narratives')} narratives, {res.get('pending_conflicts')} pending conflicts")


def fmt_hits(res, source):
    if not res.get("ok"):
        return f"search failed: {res.get('error')}"
    hits = res.get("hits") or []
    if not hits:
        return f"no results in {source}"
    lines = [f"{len(hits)} results from {source}:"]
    for h in hits:
        pin = " [PRIORITY]" if h.get("pinned") else ""
        lines.append(f"  #{h.get('id')} [{h.get('category')}] (score {h.get('score')}){pin}: {h.get('content', '')}")
    return "\n".join(lines)


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
            elif name == "rlm_search":
                query = (a.get("query") or "").strip()
                if not query:
                    send_tool(mid, "error: empty query", True)
                    continue
                res = do_search(query, int(a.get("k") or 8), None)
                log(f"tools/call rlm_search -> ok={res.get('ok')} n={res.get('count')}")
                send_tool(mid, fmt_hits(res, "your memory"), not res.get("ok"))
            elif name == "external_rlm_search":
                lattice = _latt(a.get("lattice"))
                query = (a.get("query") or "").strip()
                if not query or lattice in ("", "list"):
                    names = list_lattices()
                    txt = (("available external lattices: " + ", ".join(names)
                            + ". Call external_rlm_search with lattice=<name> and a query.")
                           if names else
                           "no external lattices available yet (drop domain .db files in the lattices dir).")
                    send_tool(mid, txt, False)
                    continue
                res = do_search(query, int(a.get("k") or 8), f"lattices/{lattice}.db")
                log(f"tools/call external_rlm_search[{lattice}] -> ok={res.get('ok')} n={res.get('count')}")
                send_tool(mid, fmt_hits(res, f"lattice '{lattice}'"), not res.get("ok"))
            elif name == "transfer_knowledge":
                lattice = _latt(a.get("lattice"))
                ids = a.get("ids") or []
                if not lattice or not ids:
                    send_tool(mid, "error: provide lattice + ids (find ids with external_rlm_search)", True)
                    continue
                res = do_import(lattice, ids)
                if res.get("ok"):
                    imp = res.get("imported") or []
                    added = sum(1 for x in imp if x.get("status") == "added")
                    reinf = sum(1 for x in imp if "reinforced" in str(x.get("status")))
                    miss = sum(1 for x in imp if x.get("status") == "not_found")
                    txt = (f"imported {added} new, reinforced {reinf} existing"
                           + (f", {miss} not found" if miss else "") + f" from '{lattice}'")
                else:
                    txt = f"transfer failed: {res.get('error')}"
                log(f"tools/call transfer_knowledge[{lattice}] -> {res.get('ok')}")
                send_tool(mid, txt, not res.get("ok"))
            elif name == "rlm_stats":
                res = do_stats()
                log(f"tools/call rlm_stats -> {res.get('ok')}")
                send_tool(mid, fmt_stats(res), not res.get("ok"))
            elif name == "rlm_conflict":
                action = (a.get("action") or "").strip()
                if action not in ("list", "resolve", "dismiss"):
                    send_tool(mid, "error: action must be list, resolve, or dismiss", True)
                    continue
                res = do_conflict(action, a.get("id"), a.get("group_id"))
                if res.get("ok"):
                    if action == "list":
                        cs = res.get("conflicts") or []
                        txt = (f"{len(cs)} pending conflict group(s): {json.dumps(cs)[:600]}"
                               if cs else "no pending conflicts")
                    else:
                        txt = f"{action} done: {json.dumps(res.get('result'))[:400]}"
                else:
                    txt = f"conflict {action} failed: {res.get('error')}"
                log(f"tools/call rlm_conflict[{action}] -> {res.get('ok')}")
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
