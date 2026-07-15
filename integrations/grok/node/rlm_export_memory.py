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
import sys, os, sqlite3, argparse, time
from collections import defaultdict

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
    narratives = _safe(c, "SELECT summary FROM session_summaries ORDER BY rowid DESC LIMIT ?",
                       (max(0, args.narrative),))

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

    emit_packed(out, "narrative (recent sessions, newest first)",
                [f"- {' '.join(str(summary).split())}" for (summary,) in narratives])

    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
