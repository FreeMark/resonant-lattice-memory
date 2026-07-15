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
"""
import sys, os, sqlite3, argparse, time
from collections import defaultdict

DEFAULT_DB = os.path.expanduser("~/grok-agent-rlm/resonant_lattice_memory.db")


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

    if authority:
        out.append("## authority (user-pinned [PRIORITY] - obey these over everything below)")
        out.append("")
        for cat, line in authority:
            out.append(f"- **[PRIORITY]** [{cat}] {line}")
        out.append("")

    for cat in sorted(by_cat):
        out.append(f"## {cat}")
        out.append("")
        for line in by_cat[cat]:
            out.append(f"- {line}")
        out.append("")

    if contested:
        out.append("## contested (unresolved conflicts - do NOT rely on these until resolved; "
                   "curate with rlm_conflict)")
        out.append("")
        for fid, cgid, cat, line in contested:
            out.append(f"- #{fid} (group {cgid}) [{cat}] {line}")
        out.append("")

    if narratives:
        out.append("## narrative (recent sessions, newest first)")
        out.append("")
        for (summary,) in narratives:
            out.append(f"- {' '.join(str(summary).split())}")
        out.append("")

    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
