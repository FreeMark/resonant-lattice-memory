#!/usr/bin/env python3
"""Watch a grok /compact -> RLM ingest ("memory cycle") to completion.

A grok /compact fires the PreCompact hook, which snapshots the transcript, ships it to the RLM
node and launches rlm_ingest.py there. That ingest is the multi-minute "memory cycle" (per-window
fact mining + dream + narrative). This tool shows the cycle's live progress and tells you clearly
when it is DONE, so you know when it is safe to continue the grok conversation.

Progress is read three ways so it stays honest even though rlm_ingest.py's stdout is block-buffered
into ingest.log (its log lines lag; the first two signals below do not):
  1. is an rlm_ingest.py process alive on the node?   -> ground-truth running / done
  2. live semantic_facts row count                    -> grows per window, buffer-immune
  3. the tail of ingest.log for this cycle            -> window X/N, dream, narrative, DONE

Transport (local vs ssh) comes from rlm_grok_conf, same as every other hook, so this works
whether grok runs on the RLM node or over ssh.

Usage:
  python rlm_watch_ingest.py             # live watch; refresh until DONE, then banner + bell
  python rlm_watch_ingest.py --once      # print one status snapshot and exit
  python rlm_watch_ingest.py --interval 3
"""
import os
import re
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlm_grok_conf as C  # noqa: E402

DB_NAME = "resonant_lattice_memory.db"

try:  # emoji/box drawing: nice on modern Windows Terminal, harmless fallback otherwise
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def remote_status_cmd():
    """A single POSIX-sh snippet that emits a parseable status block (one ssh/sh round-trip).

    pgrep -f rlm_ingest.py also matches THIS status command (it names the script) and the ssh/sh
    shell running it, so each candidate's full args (ps -ww, untruncated) are filtered: keep only a
    real ingest (contains rlm_ingest.py), drop the matcher itself (pgrep / shell wrapper / watcher).
    """
    db = f"{C.REMOTE_DIR}/{DB_NAME}"
    log = f"{C.REMOTE_DIR}/ingest.log"
    return (
        f"DB={db}\n"
        f"LOG={log}\n"
        "echo RLMSTATUS_V1\n"
        "for p in $(pgrep -f rlm_ingest.py); do\n"
        '  a=$(ps -ww -o args= -p "$p" 2>/dev/null)\n'
        '  case "$a" in\n'
        '    *pgrep*|*"bash -c"*|*"sh -c"*|*rlm_watch*) continue;;\n'
        "  esac\n"
        '  case "$a" in *rlm_ingest.py*) ;; *) continue;; esac\n'
        '  et=$(ps -o etimes= -p "$p" 2>/dev/null | tr -d " ")\n'
        '  snap=$(printf "%s" "$a" | grep -oE "/[^ ]*/incoming/[^ ]+\\.jsonl" | head -n1)\n'
        '  sz=$(stat -c %s "$snap" 2>/dev/null)\n'
        '  echo "PROC pid=$p etimes=$et snapsize=$sz args=$a"\n'
        "done\n"
        "echo \"FACTS $(sqlite3 \"$DB\" 'select count(*) from semantic_facts;' 2>/dev/null)\"\n"
        'echo "NODEUTC $(date -u +%s)"\n'
        "echo BLOCKSTART\n"
        'tail -n 60 "$LOG" 2>/dev/null\n'
        "echo BLOCKEND\n"
    )


def fetch():
    """Run the remote status command; return parsed dict or {'error': ...}."""
    try:
        r = C.run(remote_status_cmd(), timeout=25, text=True,
                  encoding="utf-8", errors="replace")
    except Exception as e:
        return {"error": f"transport error: {e}"}
    out = r.stdout or ""
    if "RLMSTATUS_V1" not in out:
        return {"error": (r.stderr or out or "no response").strip()[:300]}

    procs, facts, block, in_block = [], None, [], False
    for line in out.splitlines():
        if line == "BLOCKSTART":
            in_block = True
            continue
        if line == "BLOCKEND":
            in_block = False
            continue
        if in_block:
            block.append(line)
            continue
        if line.startswith("PROC "):
            et = re.search(r"etimes=(\d+)", line)
            sz = re.search(r"snapsize=(\d+)", line)
            sid = re.search(r"--session-id\s+(\S+)", line)
            snap = re.search(r"incoming/(\S+\.jsonl)", line)
            procs.append({
                "etimes": int(et.group(1)) if et else 0,
                "snapsize": int(sz.group(1)) if sz and sz.group(1).isdigit() else 0,
                "sid": sid.group(1) if sid else "?",
                "snap": snap.group(1) if snap else "?",
            })
        elif line.startswith("FACTS "):
            v = line[6:].strip()
            facts = int(v) if v.isdigit() else None

    return {"procs": procs, "facts": facts, "block": block}


def isolate_block(block, sid):
    """Return the log lines for one ingest cycle.

    The log is a shared append-only file. Split it at '[ingest] file=' boundaries. If a proc sid is
    given (a cycle is running), return the block that names that sid -- or [] if it has not flushed
    any lines yet (block-buffered), so we never mislabel a running cycle with a prior one's progress.
    With no sid (nothing running), return the last (most recently finished) block.
    """
    idxs = [i for i, l in enumerate(block) if l.startswith("[ingest] file=")]
    if not idxs:
        return []
    cycles = []
    for j, st in enumerate(idxs):
        en = idxs[j + 1] if j + 1 < len(idxs) else len(block)
        cycles.append(block[st:en])
    if sid:
        for cyc in cycles:
            if any(sid in l for l in cyc):
                return cyc
        return []  # running, but this cycle's lines are still buffered on the node
    return cycles[-1]


def parse_cycle(cycle):
    """Pull progress markers out of the current cycle's log lines."""
    info = {"windows": None, "cur": None, "turns": None,
            "dream": False, "narrative": False, "done": None}
    for ln in cycle:
        m = re.search(r"windows=(\d+)", ln)
        if m:
            info["windows"] = int(m.group(1))
        m = re.search(r"usable turns=(\d+)", ln)
        if m:
            info["turns"] = int(m.group(1))
        m = re.search(r"window (\d+)/(\d+)", ln)
        if m:
            info["cur"] = (int(m.group(1)), int(m.group(2)))
        if "dream cycle" in ln and "ran" in ln:
            info["dream"] = True
        if "narrative" in ln and " via " in ln:
            info["narrative"] = True
        m = re.search(r"=== DONE facts (\d+) -> (\d+)\s+born=(\d+)\s+\((\d+)s\)", ln)
        if m:
            info["done"] = {"before": int(m.group(1)), "after": int(m.group(2)),
                            "born": int(m.group(3)), "secs": int(m.group(4))}
    return info


def fmt_secs(s):
    return f"{s // 60:02d}:{s % 60:02d}"


def fmt_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0


def render(state, facts_start, ascii_only):
    """Return (panel_text, done_bool)."""
    if "error" in state:
        return (f"  [!] cannot reach node: {state['error']}", False)

    procs = state["procs"]
    facts = state["facts"]
    running = bool(procs)
    # the active cycle is the most recently launched proc = smallest etimes
    proc = min(procs, key=lambda p: p["etimes"]) if procs else None
    cycle_lines = isolate_block(state["block"], proc["sid"] if proc else None)
    cyc = parse_cycle(cycle_lines)
    done = (not running) and (cyc["done"] is not None)

    ok = "[DONE]" if ascii_only else "✅ DONE"
    run = "[RUNNING]" if ascii_only else "\U0001f7e1 RUNNING"
    idle = "[IDLE]" if ascii_only else "⬜ IDLE"
    bar = "=" * 60
    thin = "-" * 60

    L = []
    L.append(bar)
    L.append("  RLM memory cycle  -  grok /compact ingest")
    L.append(bar)

    if running:
        el = fmt_secs(proc["etimes"])
        L.append(f"  status   : {run}   elapsed {el}")
        L.append(f"  cycle    : {proc['sid']}")
        if proc["snapsize"]:
            L.append(f"  snapshot : {fmt_size(proc['snapsize'])}"
                     + (f"   ({cyc['turns']} usable turns)" if cyc["turns"] else ""))
    elif cyc["done"]:
        d = cyc["done"]
        L.append(f"  status   : {ok}   ({d['secs']}s)")
        L.append(f"  result   : facts {d['before']} -> {d['after']}   born {d['born']}")
    else:
        L.append(f"  status   : {idle}   (no ingest running; no recent cycle in the log)")

    # windows progress
    if cyc["cur"]:
        cur, tot = cyc["cur"]
        L.append(f"  windows  : {cur} / {tot}")
    elif cyc["windows"]:
        L.append(f"  windows  : 0 / {cyc['windows']}  (mining window 1)")
    elif running:
        L.append("  windows  : -   (log not flushed yet; ingest is buffered)")

    # live fact count (buffer-immune)
    if facts is not None:
        delta = f"  (+{facts - facts_start} since watch start)" if facts_start is not None else ""
        L.append(f"  facts    : {facts}{delta}")

    # phase hint
    if running:
        if cyc["narrative"]:
            phase = "writing narrative"
        elif cyc["dream"]:
            phase = "dream cycle"
        elif cyc["cur"]:
            phase = f"mining window {cyc['cur'][0]}"
        elif facts is not None and facts_start is not None and facts > facts_start:
            phase = "mining (facts committing; log buffered on node)"
        else:
            phase = "mining window 1 (log buffered on node)"
        L.append(f"  phase    : {phase}")

    # recent log lines
    tail = [ln for ln in cycle_lines if ln.strip()][-8:]
    if tail:
        L.append(thin)
        L.append("  recent log:")
        for ln in tail:
            L.append(f"    {ln[:88]}")
    L.append(thin)
    return ("\n".join(L), done)


def main():
    ap = argparse.ArgumentParser(description="Watch a grok /compact RLM ingest to completion.")
    ap.add_argument("--once", action="store_true", help="print one status snapshot and exit")
    ap.add_argument("--interval", type=float, default=5.0, help="refresh seconds (default 5)")
    ap.add_argument("--ascii", action="store_true", help="ASCII status markers (no emoji)")
    args = ap.parse_args()

    if not C.configured():
        print("rlm-grok.conf not configured (need RLM_DIR + RLM_PY; add SSH_HOST/SSH_KEY for a "
              "remote node).")
        sys.exit(1)

    facts_start = None
    try:
        while True:
            state = fetch()
            if facts_start is None and state.get("facts") is not None:
                facts_start = state["facts"]
            panel, done = render(state, facts_start, args.ascii)

            if not args.once:
                os.system("cls" if os.name == "nt" else "clear")
            print(panel)

            if args.once:
                return
            if done or (not state.get("error") and not state.get("procs")):
                # finished (or nothing running): show the closing banner and stop
                bar = "=" * 60
                mark = "[DONE]" if args.ascii else "✅"
                print(bar)
                print(f"  {mark} MEMORY CYCLE COMPLETE - safe to continue your grok chat")
                print(bar)
                sys.stdout.write("\a")  # terminal bell
                sys.stdout.flush()
                return
            print(f"  refreshing every {args.interval:g}s - Ctrl+C to stop watching")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n(stopped watching; the ingest keeps running on the node)")


if __name__ == "__main__":
    main()
