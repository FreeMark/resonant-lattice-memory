#!/usr/bin/env python3
"""Export the RLM lattice as a grok MEMORY.md projection (RLM is the SOLE writer).

Reads facts from the lattice and formats them as Markdown headings + bullets for grok's native
memory engine to index (FTS5 + vec0) and inject (first-turn + post-compaction) and search
(memory_search tool). Prints to stdout; the SessionStart hook writes it into grok's MEMORY.md.

Superseded facts are excluded; pinned facts sort first and are marked [PRIORITY].
"""
import sys, os, sqlite3, argparse, time
from collections import defaultdict

DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resonant_lattice_memory.db")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--limit", type=int, default=800)
    args = ap.parse_args()

    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = c.execute(
        "SELECT category, content, resonance_count, pinned FROM semantic_facts "
        "WHERE superseded_by IS NULL "
        "ORDER BY pinned DESC, resonance_count DESC LIMIT ?",
        (args.limit,),
    ).fetchall()

    by_cat = defaultdict(list)
    for cat, content, res, pinned in rows:
        by_cat[cat or "general"].append((content, pinned))

    out = []
    out.append("# Project Memory - Resonant Lattice Memory (RLM)")
    out.append("")
    out.append("> Sole-writer projection generated from the RLM lattice; regenerated each session.")
    out.append("> Do not hand-edit (overwritten). These are soft, fallible recalled memories -")
    out.append("> your coherence kit (AGENTS.md / DECISIONS) wins on any conflict.")
    out.append(f"> {len(rows)} facts, generated {time.strftime('%Y-%m-%d %H:%M %Z')}.")
    out.append("")
    for cat in sorted(by_cat):
        out.append(f"## {cat}")
        out.append("")
        for content, pinned in by_cat[cat]:
            line = " ".join(str(content).split())
            prefix = "**[PRIORITY]** " if pinned else ""
            out.append(f"- {prefix}{line}")
        out.append("")

    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
