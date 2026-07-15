#!/usr/bin/env python3
"""RLM memory MCP server (stdio, dependency-free).

Exposes grok's Resonant Lattice Memory as tools. Each call SSHes to a node script that runs
against the agent's own lattice (embed + dedup + provenance on writes, read-only for the rest).
Speaks newline-delimited JSON-RPC 2.0. Connection settings come from rlm_grok_conf
(~/.grok/rlm-grok.conf).

  write / curate : rlm_remember, rlm_pin, rlm_forget, rlm_unpin
  recall         : rlm_search (own), external_rlm_search + transfer_knowledge (domain lattices)
  feedback       : rlm_feedback (soft resonance nudge, helpful/unhelpful)
  inspect        : rlm_inspect (one fact + its belief history), rlm_entity (entity-graph walk)
  identity       : rlm_self_model (read / allowlisted-key write)
  health         : rlm_stats, rlm_conflict (list / resolve / dismiss)
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
    {"name": "rlm_feedback",
     "description": ("Give soft usefulness feedback on a fact in your OWN lattice - the gentle "
                     "counterpart to pin/forget. feedback='helpful' nudges its resonance up (so it "
                     "rises toward long-term); feedback='unhelpful' lowers it so a stale or wrong "
                     "fact FADES toward dormancy (recoverable, not deleted). Resonance-only: it never "
                     "changes the pin bit, so a [PRIORITY] fact keeps its protection. Use it to steer "
                     "which memories strengthen without the hard levers of pinning or forgetting."),
     "inputSchema": {"type": "object",
                     "properties": {
                         "id": {"type": "integer", "description": "the fact id to give feedback on"},
                         "feedback": {"type": "string", "enum": ["helpful", "unhelpful"],
                                      "description": "helpful raises resonance; unhelpful lowers it (fades toward dormancy)"}},
                     "required": ["id", "feedback"]}},
    {"name": "rlm_inspect",
     "description": ("Inspect ONE fact in full by id (read-only): its content, category, tier, "
                     "resonance, pin state, source, and any conflict group, PLUS its belief history - "
                     "the chain of facts that superseded it and the predecessors it replaced. Use "
                     "after a search or conflict listing hands you an id and you want the whole story "
                     "of that memory, not just the fuzzy content match."),
     "inputSchema": {"type": "object",
                     "properties": {
                         "id": {"type": "integer", "description": "the fact id to inspect"}},
                     "required": ["id"]}},
    {"name": "rlm_entity",
     "description": ("Walk your OWN entity graph (read-only), complementary to rlm_search. Give an "
                     "'entity' name to get the facts linked to it (ranked by resonance) plus the "
                     "entities that co-occur with it - 'everything I know about X, and what's "
                     "connected to X'. Or give a 'fact_id' to list the entities on that one fact. "
                     "Graph traversal, not semantic similarity."),
     "inputSchema": {"type": "object",
                     "properties": {
                         "entity": {"type": "string", "description": "entity name to walk (facts about it + neighbours)"},
                         "fact_id": {"type": "integer", "description": "instead, list the entities on this fact"},
                         "k": {"type": "integer", "description": "max facts/neighbours (default 15)"}},
                     "required": []}},
    {"name": "rlm_self_model",
     "description": ("Read or carefully update your self-model (identity) - the curated, "
                     "authoritative record of who you are that is surfaced deterministically at "
                     "session start. op='get' reads one key (or the whole model if no key). op='set' "
                     "upserts one key; only a small allowlist of keys is writable (e.g. role, "
                     "relationship_with_user, mandate, current_focus) so identity core does not drift "
                     "session to session - an off-list key is rejected with the allowed set. Use "
                     "deliberately, as you would pin a standing decision, not for passing moods."),
     "inputSchema": {"type": "object",
                     "properties": {
                         "op": {"type": "string", "enum": ["get", "set"], "description": "get | set"},
                         "key": {"type": "string", "description": "identity key (e.g. role, mandate, current_focus)"},
                         "value": {"type": "string", "description": "the value to record (for set)"}},
                     "required": ["op"]}},
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


def _run(cmd, inp=b"", timeout=90):
    """SSH the node command; parse its last stdout line as JSON. Shared shape for the read verbs."""
    try:
        r = subprocess.run(["ssh", *C.SSH_OPTS, C.SSH_HOST, cmd], input=inp,
                           capture_output=True, timeout=timeout)
        out = (r.stdout or b"").decode("utf-8", "replace").strip()
        err = (r.stderr or b"").decode("utf-8", "replace")
        return json.loads(out.splitlines()[-1]) if out else {"ok": False, "error": (err or "no output")[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def do_feedback(fid, fb):
    cmd = f"{C.REMOTE_PY} {C.REMOTE_DIR}/rlm_feedback.py --id {int(fid)} --fb {fb}"
    return _run(cmd)


def do_inspect(fid):
    return _run(f"{C.REMOTE_PY} {C.REMOTE_DIR}/rlm_inspect.py --id {int(fid)}")


def do_entity(entity, fact_id, k):
    if fact_id:
        return _run(f"{C.REMOTE_PY} {C.REMOTE_DIR}/rlm_entity.py --fact-id {int(fact_id)}")
    return _run(f"{C.REMOTE_PY} {C.REMOTE_DIR}/rlm_entity.py --k {int(k)}",
                inp=(entity or "").encode("utf-8"))


def _key(k):
    """Sanitize a self-model key to a shell-safe identifier token."""
    return re.sub(r"[^a-z0-9_]", "", (k or "").strip().lower())


def do_self_model(op, key, value):
    cmd = f"{C.REMOTE_PY} {C.REMOTE_DIR}/rlm_self_model.py --op {op}"
    sk = _key(key)
    if sk:
        cmd += f" --key {sk}"
    inp = value.encode("utf-8") if (op == "set" and value) else b""
    return _run(cmd, inp=inp)


def fmt_inspect(res):
    if not res.get("ok"):
        return f"inspect failed: {res.get('error')}"
    f = res.get("fact") or {}
    pin = " [PRIORITY]" if f.get("pinned") else ""
    cg = f" [CONFLICT {f.get('conflict_group_id')}]" if f.get("conflict_group_id") else ""
    lines = [f"#{f.get('id')} [{f.get('category')}] tier={f.get('tier')} "
             f"resonance={f.get('resonance_count')}{pin}{cg}",
             f"  content: {f.get('content', '')}"]
    if f.get("source_ref"):
        lines.append(f"  source_ref: {f.get('source_ref')}")
    lines.append(f"  learned_cycle={f.get('learned_at_cycle')} "
                 f"last_confirmed={f.get('last_confirmed_cycle')}")
    chain = res.get("superseded_by_chain") or []
    if chain:
        lines.append("  superseded by: " + " -> ".join(f"#{r.get('id')} {r.get('content','')[:50]}"
                                                        for r in chain))
    replaced = res.get("replaced") or []
    if replaced:
        lines.append("  replaced (predecessors): "
                     + "; ".join(f"#{r.get('id')} {r.get('content','')[:50]}" for r in replaced))
    return "\n".join(lines)


def fmt_entity(res):
    if not res.get("ok"):
        return f"entity walk failed: {res.get('error')}"
    if res.get("mode") == "fact":
        ents = res.get("entities") or []
        return (f"fact #{res.get('fact_id')} entities: " + ", ".join(ents)) if ents \
            else f"fact #{res.get('fact_id')} has no linked entities"
    facts = res.get("facts") or []
    related = res.get("related") or []
    lines = [f"entity '{res.get('entity')}': {len(facts)} fact(s)"]
    for h in facts:
        pin = " [PRIORITY]" if h.get("pinned") else ""
        lines.append(f"  #{h.get('id')} [{h.get('category')}] (res {h.get('resonance_count')}){pin}: "
                     f"{h.get('content', '')}")
    if related:
        lines.append("neighbours: "
                     + ", ".join(f"{r.get('entity')} ({r.get('shared_facts')})" for r in related))
    return "\n".join(lines)


def fmt_self_model(res):
    if not res.get("ok"):
        err = f"self-model {res.get('op', '')} failed: {res.get('error')}"
        if res.get("allowed"):
            err += ". Writable keys: " + ", ".join(res["allowed"])
        return err
    if res.get("op") == "set":
        return f"self-model updated: {res.get('key')} = {res.get('value')}"
    model = res.get("model")
    if model is None:
        return f"no self-model entry for '{res.get('key')}'"
    if isinstance(model, dict):
        return f"{model.get('key')}: {model.get('value')}"
    if not model:
        return "self-model is empty"
    return "\n".join(f"- {m.get('key')}: {m.get('value')}" for m in model)


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
            elif name == "rlm_feedback":
                fid = a.get("id")
                fb = (a.get("feedback") or "").strip()
                if not fid or fb not in ("helpful", "unhelpful"):
                    send_tool(mid, "error: provide id and feedback=helpful|unhelpful", True)
                    continue
                res = do_feedback(fid, fb)
                if res.get("ok"):
                    txt = (f"feedback '{fb}' on #{res.get('id')}: resonance "
                           f"{res.get('resonance_before')} -> {res.get('resonance_after')} "
                           f"(delta {res.get('delta')})")
                    if res.get("note"):
                        txt += f". {res['note']}"
                else:
                    txt = f"feedback failed: {res.get('error')}"
                log(f"tools/call rlm_feedback -> {res.get('ok')}")
                send_tool(mid, txt, not res.get("ok"))
            elif name == "rlm_inspect":
                fid = a.get("id")
                if not fid:
                    send_tool(mid, "error: provide a fact id", True)
                    continue
                res = do_inspect(fid)
                log(f"tools/call rlm_inspect[{fid}] -> {res.get('ok')}")
                send_tool(mid, fmt_inspect(res), not res.get("ok"))
            elif name == "rlm_entity":
                entity = (a.get("entity") or "").strip()
                fact_id = a.get("fact_id")
                if not entity and not fact_id:
                    send_tool(mid, "error: provide entity or fact_id", True)
                    continue
                res = do_entity(entity, fact_id, int(a.get("k") or 15))
                log(f"tools/call rlm_entity -> {res.get('ok')}")
                send_tool(mid, fmt_entity(res), not res.get("ok"))
            elif name == "rlm_self_model":
                op = (a.get("op") or "").strip()
                if op not in ("get", "set"):
                    send_tool(mid, "error: op must be get or set", True)
                    continue
                if op == "set" and not (a.get("key") and (a.get("value") or "").strip()):
                    send_tool(mid, "error: set requires key and value", True)
                    continue
                res = do_self_model(op, a.get("key"), (a.get("value") or "").strip())
                log(f"tools/call rlm_self_model[{op}] -> {res.get('ok')}")
                send_tool(mid, fmt_self_model(res), not res.get("ok"))
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
