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
        "SELECT category, content, resonance_count, pinned FROM semantic_facts "
        "WHERE superseded_by IS NULL "
        "ORDER BY pinned DESC, resonance_count DESC LIMIT ?",
        (args.limit,),
    ).fetchall()
    self_model = _safe(c, "SELECT key, value FROM agent_identity ORDER BY key")
    narratives = _safe(c, "SELECT summary FROM session_summaries ORDER BY rowid DESC LIMIT ?",
                       (max(0, args.narrative),))

    by_cat = defaultdict(list)
    for cat, content, res, pinned in rows:
        by_cat[cat or "general"].append((content, pinned))

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

    for cat in sorted(by_cat):
        out.append(f"## {cat}")
        out.append("")
        for content, pinned in by_cat[cat]:
            line = " ".join(str(content).split())  # collapse to a single line
            prefix = "**[PRIORITY]** " if pinned else ""
            out.append(f"- {prefix}{line}")
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
