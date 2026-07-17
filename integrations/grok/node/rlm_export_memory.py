#!/usr/bin/env python3
"""Export the RLM lattice as a grok MEMORY.md projection (RLM is the SOLE writer).

Reads facts from the lattice and formats them as Markdown headings + bullets for grok's
native memory engine to index (FTS5 + vec0) and inject (first-turn + post-compaction) and
search (memory_search tool). Prints to stdout; the Windows SessionStart hook writes the
result into ~/.grok/memory/<workspace-slug>/MEMORY.md.

Includes the self-model (identity) at the top and the recent narrative (autobiographical
gist) at the bottom, so the agent wakes up with who-it-is and what-has-been-happening, not
just a fact list. Superseded facts are excluded; pinned facts sort first and are marked
[PRIORITY].

Sections are CHUNK-SHAPED: grok's memory indexer chunks files at 1600 chars (320 overlap) and
both its automatic injections (first-turn, post-compaction) and memory_search rank CHUNKS by
hybrid score. A big category emitted as one long list would be split mid-list into headingless
chunks; packing each section to <=1400 chars with its own heading keeps every indexed chunk a
coherent, self-contained unit, so what grok's engine injects is whole topics, not fragments.
"""
import sys, os, sqlite3, argparse, time, json, unicodedata
from collections import defaultdict

# ASCII-normalize narrative text on export (old rows may predate the store-side sanitize).
# Keys are \u escapes so THIS source file stays ASCII-only.
_PUNCT = {"\u2014": " - ", "\u2013": " - ", "\u2012": " - ", "\u2015": " - ", "\u2011": "-",
          "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2026": "...",
          "\u2192": "->", "\u2190": "<-"}


def _ascii(s):
    s = str(s)
    for k, v in _PUNCT.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.split())

DEFAULT_DB = os.path.expanduser("~/grok-agent-rlm/resonant_lattice_memory.db")
CHUNK_BUDGET = 1400  # keep heading + bullets safely inside grok's 1600-char chunk window


def emit_packed(out, heading, bullets):
    """Emit a section as one or more <=CHUNK_BUDGET-char parts, each under its own heading
    (continuations get '(cont.)'), so the chunker never orphans bullets from their context."""
    if not bullets:
        return
    part, size, first = [], 0, True

    def flush():
        nonlocal part, size, first
        if not part:
            return
        out.append(f"## {heading}" if first else f"## {heading} (cont.)")
        out.append("")
        out.extend(part)
        out.append("")
        part, size, first = [], 0, False

    for b in bullets:
        if part and size + len(b) + 1 > CHUNK_BUDGET:
            flush()
        part.append(b)
        size += len(b) + 1
    flush()


def _safe(conn, sql, params=()):
    try:
        return conn.execute(sql, params).fetchall()
    except Exception:
        return []


def _loads_list(js):
    if not js:
        return []
    try:
        v = json.loads(js)
        return v if isinstance(v, list) else [str(v)]
    except Exception:
        return [str(js)]


def _meta_int(conn, key):
    """Read an integer meta counter (e.g. the live memory_cycle) read-only; None if absent."""
    r = _safe(conn, "SELECT value FROM meta WHERE key=?", (key,))
    try:
        return int(r[0][0]) if r else None
    except Exception:
        return None


def _recent_narratives(conn, limit):
    """Newest-first narratives with the P1 structured fields, ordered by created_cycle
    (not rowid). Falls back to a summary-only SELECT on a pre-P1 schema, since the
    projection opens the DB read-only and never runs migrations."""
    if limit <= 0:
        return []
    rich = _safe(conn,
                 "SELECT summary, throughline, open_loops, decisions, closed, "
                 "COALESCE(created_cycle, 0), COALESCE(historical, 0) "
                 "FROM session_summaries "
                 "ORDER BY COALESCE(created_cycle, 0) DESC, summary_id DESC LIMIT ?", (limit,))
    if rich:
        return rich
    plain = _safe(conn, "SELECT summary FROM session_summaries "
                        "ORDER BY COALESCE(created_cycle, 0) DESC, rowid DESC LIMIT ?", (limit,))
    return [(s, None, None, None, None, 0, 0) for (s,) in plain]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--narrative", type=int, default=3)  # recent narratives to surface
    args = ap.parse_args()

    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = c.execute(
        "SELECT id, category, content, resonance_count, pinned, conflict_group_id "
        "FROM semantic_facts WHERE superseded_by IS NULL "
        "ORDER BY pinned DESC, resonance_count DESC LIMIT ?",
        (args.limit,),
    ).fetchall()
    self_model = _safe(c, "SELECT key, value FROM agent_identity ORDER BY key")
    narratives = _recent_narratives(c, max(0, args.narrative))

    # Partition (mirrors the native recall presentation: authority block + conflict quarantine):
    #   - pinned facts        -> hoisted into one AUTHORITY block, obeyed over everything below.
    #   - contested facts      -> a live fact still carrying a conflict_group_id is an UNRESOLVED
    #     (unpinned)             conflict (resolve/dismiss both NULL the column), so it is pulled
    #                            OUT of its category into a CONTESTED block and NOT projected as
    #                            settled fact. Pins win over contest (a pinned fact stays authority).
    #   - everything else      -> normal category sections.
    authority, contested = [], []
    by_cat = defaultdict(list)
    for fid, cat, content, res, pinned, cgid in rows:
        line = " ".join(str(content).split())  # collapse to a single line
        cat = cat or "general"
        if pinned:
            authority.append((cat, line))
        elif cgid is not None:
            contested.append((fid, cgid, cat, line))
        else:
            by_cat[cat].append(line)

    out = []
    out.append("# Project Memory - resonant lattice memory")
    out.append("")
    out.append("> Sole-writer projection generated from the resonant lattice memory; regenerated each session.")
    out.append("> Do not hand-edit (overwritten). These are soft, fallible recalled memories -")
    out.append("> verify before you rely on them, and prefer what the current session establishes on any conflict.")
    out.append(f"> {len(rows)} facts, generated {time.strftime('%Y-%m-%d %H:%M %Z')}.")
    out.append("")

    if self_model:
        out.append("## self-model")
        out.append("")
        for key, value in self_model:
            out.append(f"- **{key}**: {' '.join(str(value).split())}")
        out.append("")

    emit_packed(out, "authority (user-pinned [PRIORITY] - obey these over everything below)",
                [f"- **[PRIORITY]** [{cat}] {line}" for cat, line in authority])

    for cat in sorted(by_cat):
        emit_packed(out, cat, [f"- {line}" for line in by_cat[cat]])

    emit_packed(out, "contested (unresolved conflicts - do NOT rely on these until resolved; "
                     "curate with rlm_conflict)",
                [f"- #{fid} (group {cgid}) [{cat}] {line}" for fid, cgid, cat, line in contested])

    narr_bullets = []
    for i, (summary, throughline, open_js, dec_js, closed_js, cyc, hist) in enumerate(narratives):
        head = _ascii(throughline or summary or "")
        if not head:
            continue
        if i == 0 and not hist:
            # newest, current status: render the full arc (throughline + open/decided/closed).
            # 'closed' is surfaced (capped) so a resumed-but-not-yet-reingested item that has
            # since been finished does not keep reading as still-open.
            narr_bullets.append(f"- **now (cycle {cyc})**: {head}")
            for it in _loads_list(open_js):
                narr_bullets.append(f"  - open: {_ascii(it)}")
            for it in _loads_list(dec_js):
                narr_bullets.append(f"  - decided: {_ascii(it)}")
            for it in _loads_list(closed_js)[:2]:
                narr_bullets.append(f"  - closed: {_ascii(it)}")
        else:
            tag = "historical" if hist else f"cycle {cyc}"
            narr_bullets.append(f"- ({tag}) {head}")
    mc = _meta_int(c, "memory_cycle")
    clock = f"; lattice now at memory_cycle {mc}" if mc is not None else ""
    emit_packed(out, "narrative (session arc, newest first - a remembered gist" + clock +
                     "; the per-row cycle is when that arc was recorded, not now; verify "
                     "against the authority block + live state)", narr_bullets)

    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
