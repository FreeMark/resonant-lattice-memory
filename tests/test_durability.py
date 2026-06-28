r"""test_durability.py - operational robustness: concurrency + crash/restart.

"Long-term" memory must also survive the messy realities of deployment:
  CONCURRENCY  - the Hermes host may call add/recall from multiple threads. We
                 run 8 threads hammering add_or_reinforce + search on ONE store
                 and assert: every committed write lands (no lost/corrupt rows),
                 PRAGMA integrity_check is clean, and nothing deadlocks.
  CRASH/RESTART - a process can die mid-write. A subprocess commits N facts, then
                 HARD-exits (os._exit, no graceful close) while starting another
                 write; we reopen the DB and assert it is intact (integrity_check)
                 and the committed facts survived (a half-written one rolled back).

Pure substrate (no LLM); embeddings only for the recall side.
"""
import os
import sys
import threading
import time
import _common as C

CRASH_CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_crash_child.py")


def main():
    if not C.ollama_up():
        print("Ollama not reachable (embeddings)."); return 2
    suite = C.Suite("Durability — concurrency + crash/restart", model="(no LLM)")

    # ---- CONCURRENCY ----
    s, R, _, db = C.make_store("concurrency.db")
    N_THREADS, PER = 8, 40
    errors = []
    done = []

    def worker(tid):
        try:
            for k in range(PER):
                emb = R._get_embedding(f"thread {tid} fact {k}: account ACT-{tid:02d}{k:03d} note.")
                s.add_or_reinforce_fact(f"thread {tid} fact {k}: account ACT-{tid:02d}{k:03d} note.",
                                        emb, "biz", f"t{tid}", entities=[f"acct{tid}{k}"])
                if k % 7 == 0:
                    R.search(f"account ACT-{tid:02d}{k:03d}", limit=5)   # interleave reads
            done.append(tid)
        except Exception as e:
            errors.append((tid, repr(e)))

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    elapsed = time.time() - t0
    alive = [t for t in threads if t.is_alive()]
    suite.report("concurrency", f"{len(done)}/{N_THREADS} threads done, {len(errors)} errors, {elapsed:.1f}s")
    suite.hard("no deadlock — all threads completed", not alive and len(done) == N_THREADS, f"alive={len(alive)}")
    suite.hard("no thread raised under concurrent add+recall", not errors, str(errors[:3]))
    rows = s._conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
    suite.hard("every concurrent write landed (no lost rows)", rows == N_THREADS * PER,
               f"{rows}/{N_THREADS * PER}")
    integ = s._conn.execute("PRAGMA integrity_check").fetchone()[0]
    suite.hard("DB integrity_check clean after concurrent load", integ == "ok", integ)
    s.close()

    # ---- CRASH / RESTART ----
    import tempfile
    import subprocess
    crash_db = os.path.join(tempfile.mkdtemp(), "crash.db")
    # child commits 25 facts then hard-exits (os._exit) while beginning a 26th
    out = subprocess.run([sys.executable, CRASH_CHILD, crash_db, "25"],
                         capture_output=True, text=True, timeout=120)
    suite.report("crash child", (out.stdout or out.stderr or "").strip()[:80])
    # reopen on the same db (simulates restart after crash)
    store_mod = C.load("store")
    s2 = store_mod.LatticeStore(db_path=crash_db, vector_dim=768)
    integ2 = s2._conn.execute("PRAGMA integrity_check").fetchone()[0]
    survived = s2._conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
    suite.report("after crash+restart", f"integrity={integ2}, committed facts survived={survived}")
    suite.hard("DB is intact after a mid-write crash (integrity_check ok)", integ2 == "ok", integ2)
    suite.hard("committed facts survived the crash (>=25)", survived >= 25, f"{survived}")
    # one fact recallable post-restart (the store is usable, not just intact)
    R2 = C.load("retrieval").LatticeRetriever(s2, C.OLLAMA, C.EMBED_MODEL, min_similarity=0.30)
    hits = R2.search("crash fact 1", limit=5)
    suite.hard("store is usable post-restart (recall returns rows)", len(hits) > 0, f"{len(hits)} hits")
    s2.close()

    return suite.finish("durability_results.md")


if __name__ == "__main__":
    sys.exit(main())
