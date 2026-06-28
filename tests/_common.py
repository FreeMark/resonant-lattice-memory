r"""_common.py - shared substrate-test harness for the corrected business tests.

These tests follow the same discipline as live_e2e.py:

  * HARD checks  -> deterministic / substrate invariants. A failure is a real
                    defect; it is collected and forces a non-zero exit code.
  * SOFT checks  -> LLM-dependent yields (extraction, distillation, narrative,
                    LLM relation triples). Reported honestly as PASS/WARN but
                    they do NOT fail the run, because they vary by model. A WARN
                    means "the model under-produced this run", not "the system is
                    broken".
  * report()     -> pure measurements (counts, tiers, timings) written verbatim.

Result files written by these tests contain ONLY measured values - never
hardcoded "expected" text presented as output, and never an unconditional
"works" conclusion. If a thing was not verified, the file says so.

Run a single test:   python tests/<name>.py
Run all:             python tests/run_all.py
"""
import importlib.util
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# resonant_lattice lives one level up from this dir (repo_root/tests -> repo_root).
PLUGIN = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "resonant_lattice"))
if PLUGIN not in sys.path:
    sys.path.insert(0, PLUGIN)

# Windows consoles default to cp1252; reports use unicode bullets/arrows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OLLAMA = os.environ.get("RL_OLLAMA", "http://localhost:11434")
EMBED_MODEL = os.environ.get("RL_EMBED_MODEL", "nomic-embed-text")
REASON_MODEL = os.environ.get("RL_REASON_MODEL", "nemotron-3-super:cloud")

OUT_DIR = Path(__file__).parent.parent / "results"


def load(name):
    """Load a plugin module by bare filename (the Hermes loader pattern)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(PLUGIN, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ollama_up(timeout=5):
    """True if the Ollama daemon answers /api/tags."""
    try:
        req = urllib.request.Request(f"{OLLAMA}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            json.loads(r.read().decode())
        return True
    except Exception:
        return False


def warm_reason_model(model=REASON_MODEL, timeout=120):
    """Warm a (possibly cold) reasoning model; return (ok, seconds)."""
    t0 = time.time()
    try:
        payload = {"model": model, "prompt": 'Return only this JSON: {"ok": true}',
                   "stream": False, "options": {"temperature": 0.1}}
        req = urllib.request.Request(f"{OLLAMA}/api/generate",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            json.loads(r.read().decode())
        return True, time.time() - t0
    except Exception as e:
        return False, time.time() - t0


def encrypted_binding_available():
    """True iff the SQLCipher binding (sqlcipher3) is genuinely importable.

    NOTE: store_common.sqlite_binding_error() only reports whether a *requested*
    encrypted binding failed - it returns None in plaintext mode too, so it is a
    false-positive for "is encryption possible". We check the module directly.
    Used to SKIP (never fake) the at-rest opacity test when the binding is absent.
    """
    try:
        import sqlcipher3  # noqa: F401
        return True
    except Exception:
        return False


class Suite:
    """Tiny check/report collector with hard vs soft semantics + a results dump."""

    def __init__(self, title, model=REASON_MODEL):
        self.title = title
        self.model = model
        self.hard_fail = []
        self.soft_warn = []
        self.lines = []          # ordered (kind, name, status, detail) for the report
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")

    def hard(self, name, cond, detail=""):
        ok = bool(cond)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] (hard) {name}" + (f"  -- {detail}" if detail else ""))
        self.lines.append(("hard", name, status, str(detail)))
        if not ok:
            self.hard_fail.append(name)
        return ok

    def soft(self, name, cond, detail=""):
        ok = bool(cond)
        status = "PASS" if ok else "WARN"
        print(f"  [{status}] (soft) {name}" + (f"  -- {detail}" if detail else ""))
        self.lines.append(("soft", name, status, str(detail)))
        if not ok:
            self.soft_warn.append(name)
        return ok

    def report(self, name, value):
        print(f"  [INFO] {name}: {value}")
        self.lines.append(("info", name, "INFO", str(value)))

    def skip(self, name, reason):
        print(f"  [SKIP] {name}  -- {reason}")
        self.lines.append(("skip", name, "SKIP", str(reason)))

    def write_md(self, filename, extra_sections=None):
        OUT_DIR.mkdir(exist_ok=True)
        path = OUT_DIR / filename
        n_hard = sum(1 for k, *_ in self.lines if k == "hard")
        n_hard_pass = n_hard - len(self.hard_fail)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {self.title}\n\n")
            f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Memory model**: {self.model}  \n")
            f.write(f"**Embed model**: {EMBED_MODEL}  \n\n")
            verdict = "PASS" if not self.hard_fail else "FAIL"
            f.write(f"**Verdict (hard invariants)**: {verdict} "
                    f"({n_hard_pass}/{n_hard})  \n")
            if self.soft_warn:
                f.write(f"**Soft/LLM warnings**: {len(self.soft_warn)} "
                        f"({', '.join(self.soft_warn)})  \n")
            f.write("\n## Checks (measured)\n\n")
            f.write("| kind | check | status | detail |\n|---|---|---|---|\n")
            for kind, name, status, detail in self.lines:
                d = detail.replace("|", "\\|").replace("\n", " ")[:300]
                f.write(f"| {kind} | {name} | {status} | {d} |\n")
            if extra_sections:
                for heading, body in extra_sections.items():
                    f.write(f"\n## {heading}\n\n{body}\n")
        print(f"\nResults written to {path}")
        return path

    def finish(self, md_filename=None, extra_sections=None):
        if md_filename:
            self.write_md(md_filename, extra_sections)
        print(f"\n{'-' * 72}")
        if self.hard_fail:
            print(f"RESULT [{self.title}]: FAIL -- hard invariants failed: {self.hard_fail}")
            code = 1
        else:
            print(f"RESULT [{self.title}]: PASS -- all hard invariants held"
                  + (f" ({len(self.soft_warn)} soft warning(s))" if self.soft_warn else ""))
            code = 0
        return code


def make_store(db_name="t.db", *, min_similarity=0.30, freshness_halflife=0.0, **overrides):
    """Build a real LatticeStore + LatticeRetriever on a temp DB.
    Retriever-only knobs (min_similarity, freshness_halflife) are kw-only; all
    other overrides go to the LatticeStore constructor.
    Returns (store, retriever, store_module, db_path).
    """
    import tempfile
    store_mod = load("store")
    retr_mod = load("retrieval")
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, db_name)
    kw = dict(db_path=db, vector_dim=768, promotion_threshold=4,
              short_tier_cycles=1, mid_tier_cycles=1, initial_resonance=5)
    kw.update(overrides)
    s = store_mod.LatticeStore(**kw)
    R = retr_mod.LatticeRetriever(s, OLLAMA, EMBED_MODEL,
                                  min_similarity=min_similarity,
                                  freshness_halflife=freshness_halflife)
    return s, R, store_mod, db


def add_fact(s, R, content, category="general", entities=None, session="t",
             source_quote=None, with_relations=False, use_llm=False, model=REASON_MODEL):
    """Embed + (optionally) HRR-encode + add_or_reinforce_fact. Returns (action, fid).
    Relations are extracted only when explicitly requested.
    """
    entities = entities or []
    emb = R._get_embedding(content)
    hrr_vec = None
    try:
        from store_common import hrr, _HRR_AVAILABLE
        if _HRR_AVAILABLE:
            hrr_vec = hrr.encode_fact(content, entities, dim=s.hrr_dim)
    except Exception:
        pass
    kw = dict(hrr_vector=hrr_vec, entities=entities)
    if source_quote is not None:
        kw["source_quote"] = source_quote
    action, fid = s.add_or_reinforce_fact(content, emb, category, session, **kw)
    if with_relations and fid > 0 and action == "added":
        s.extract_and_store_relations(fid, content, entities=entities, min_confidence=0.5,
                                      reason_model=model, ollama_endpoint=OLLAMA, use_llm=use_llm)
    return action, fid


import random as _random

# Shared distractor generator (diverse, unique-token business facts) for the
# "under load" tests. Limited pools => some natural near-duplicates (realistic).
_D_FIRST = "Ava Noah Mia Liam Emma Ezra Iris Omar Lena Theo Nadia Cyrus Priya Soren Yusuf Greta Dario Hana Felix Ingrid".split()
_D_LAST = "Reyes Okafor Lindqvist Tanaka Mwangi Costa Devi Halloran Voss Bauer Nakamura Abara Petrov Singh Ferro Yoon".split()
_D_COMP = "Tanager Meridian Halcyon Vantyx Borealis Cindra Quokka Vellum Kestrel Lumen Pylon Draxis Orrery Nimbus Saffron".split()
_D_CITY = "Tromso Reykjavik Lisbon Osaka Nairobi Tallinn Boise Cusco Perth Ghent Almaty Hobart".split()
_D_PLAN = "Starter Growth Enterprise Scale Atlas Sovereign".split()
_D_PROD = "the billing API the data pipeline the auth gateway the search index the export job".split()
_D_DEPT = "finance security platform growth support data".split()


def distractor_fact(i, rng=None):
    """A distinct, irrelevant business fact (content, entities). Unique id token
    keeps rows distinct; limited pools yield some natural near-duplicates."""
    rng = rng or _random.Random(i)
    p = f"{rng.choice(_D_FIRST)} {rng.choice(_D_LAST)}"
    co = rng.choice(_D_COMP)
    pick = rng.randint(0, 4)
    if pick == 0:
        return (f"Customer {co} (account ACT-{i:06d}) moved to the {rng.choice(_D_PLAN)} plan; MRR {rng.randint(1000,900000)} cents.", [co.lower()])
    if pick == 1:
        return (f"{p} from {co} filed ticket TK-{i:06d} about {rng.choice(_D_PROD)} in {rng.choice(_D_CITY)}.", [co.lower(), p.split()[0].lower()])
    if pick == 2:
        return (f"Note N-{i:06d}: reviewed {rng.choice(_D_DEPT)} metrics for {co}; follow up with {p}.", [co.lower()])
    if pick == 3:
        return (f"Invoice INV-{i:06d}: {co} billed {rng.randint(1000,500000)} cents for {rng.choice(_D_PROD)}.", [co.lower()])
    return (f"{p} prefers {rng.choice(['async standups','dark mode','weekly digests'])} and works in {rng.choice(_D_DEPT)} at {co}.", [p.split()[0].lower(), co.lower()])


def load_distractors(s, R, n, start=0, session="bg"):
    """Bulk-ingest n distractor facts. Returns count actually added (vs merged)."""
    rng = _random.Random(20260626 + start)
    added = 0
    for k in range(start, start + n):
        content, ents = distractor_fact(k, rng)
        a, _ = add_fact(s, R, content, category="bg", entities=ents, session=session)
        added += (a == "added")
    return added


def make_provider(config=None, session="t", home=None):
    """Construct + initialize a real LatticeMemoryProvider, stubbing the Hermes
    host packages. If `home` is given, reopen the DB there (simulate a RESTART);
    else a fresh temp home. Returns (provider, hermes_home). Needs Ollama."""
    import json as _json
    import tempfile
    import types as _types
    from pathlib import Path as _Path
    stubs = [
        ("agent", {}),
        ("agent.memory_provider", {"MemoryProvider": type("MemoryProvider", (object,), {})}),
        ("tools", {}),
        ("tools.registry", {"tool_error": lambda msg: _json.dumps({"error": msg})}),
    ]
    for name, attrs in stubs:
        if name not in sys.modules:
            m = _types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m
    sys.modules["agent"].memory_provider = sys.modules["agent.memory_provider"]
    sys.modules["tools"].registry = sys.modules["tools.registry"]
    home = home or tempfile.mkdtemp()
    if "hermes_constants" not in sys.modules:
        hc = _types.ModuleType("hermes_constants")
        hc.get_hermes_home = lambda: _Path(home)
        sys.modules["hermes_constants"] = hc
    prov_mod = load("__init__")
    cfg = dict(config or {})
    cfg.setdefault("embed_model", EMBED_MODEL)
    cfg.setdefault("reason_model", REASON_MODEL)
    prov = prov_mod.LatticeMemoryProvider(cfg)
    prov.initialize(session, hermes_home=home, agent_context="primary")
    return prov, home


def make_conflict_group(s, winner_id, loser_id, group="cg-test",
                        since_cycle=1, now_cycle=3, win_res=4, lose_res=3):
    """Deterministically place two facts into a conflict group exactly as
    resolve_hrr_conflicts would, so the conflict *machinery* (pending_conflicts +
    resolve_conflict) can be asserted independently of the HRR detection heuristic.
    Mirrors test_resonant_lattice.test_store_pending_conflicts_and_resolve.
    """
    for fid, res in ((winner_id, win_res), (loser_id, lose_res)):
        s._conn.execute(
            "UPDATE semantic_facts SET conflict_group_id=?, tier='long', "
            "resonance_count=?, conflict_since_cycle=? WHERE id=?",
            (group, res, since_cycle, fid))
    s.set_cycle_counts(memory_cycle=now_cycle)
    s._conn.commit()
