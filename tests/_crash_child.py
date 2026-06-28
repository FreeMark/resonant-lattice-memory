r"""_crash_child.py - commit N facts, then HARD-exit mid-write (crash simulation).

Usage: python _crash_child.py <db_path> <n>
Commits n facts (each add_or_reinforce_fact autocommits), then opens an
uncommitted transaction with a partial INSERT and calls os._exit() — no commit,
no close. On reopen, SQLite must roll the partial write back and keep the n
committed facts. Uses fake fixed embeddings, so it needs no Ollama.
"""
import os
import sys

PLUGIN = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "resonant_lattice"))
sys.path.insert(0, PLUGIN)

from store import LatticeStore   # noqa: E402

DIM = 768


def run(db, n):
    s = LatticeStore(db_path=db, vector_dim=DIM)
    emb = [0.1] * DIM
    for k in range(n):
        s.add_or_reinforce_fact(f"crash fact {k}: durable record number {k}.",
                                emb, "biz", "crash", entities=[f"rec{k}"])
    # begin a write and DIE mid-transaction (no commit, no close)
    try:
        s._conn.execute("BEGIN")
        s._conn.execute(
            "INSERT INTO semantic_facts (content, category, tier, resonance_count) "
            "VALUES ('HALF-WRITTEN crash fact that must roll back', 'biz', 'short', 1)")
    except Exception:
        pass
    sys.stdout.write(f"committed {n}, dying mid-write\n")
    sys.stdout.flush()
    os._exit(137)   # hard kill — uncommitted INSERT must not survive


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
