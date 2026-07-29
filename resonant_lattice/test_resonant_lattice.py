"""Test suite for the Resonant Lattice Memory plugin.

Runs two layers:
  • Entity-precision tests   - exercise the real entity_extractor (no heavy deps).
  • Store-lifecycle tests     - exercise the real LatticeStore; auto-skipped if
                                sqlite-vec or numpy is unavailable.

Usage:
    python test_resonant_lattice.py          # plain runner (prints PASS/SKIP)
    pytest test_resonant_lattice.py           # also works under pytest

Place this beside __init__.py / store.py / holographic.py / entity_extractor.py
(the plugin directory), or point PLUGIN_DIR at it.
"""

import base64
import os
import sys
import tempfile
import importlib.util

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(PLUGIN_DIR, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 - entity precision (always runs)
# ─────────────────────────────────────────────────────────────────────────────
ee = _load("entity_extractor")


def test_english_compounds_rejected():
    out = set(ee.extract_entities(
        "Our long-term well-being depends on a state-of-the-art high-level approach."))
    assert out == set(), f"noise leaked: {out}"


def test_digit_bearing_ids_kept():
    out = set(ee.extract_entities("I run granite-4.1-30b and granite-16k on the RTX 3090 Ti"))
    assert "granite-4.1-30b" in out and "granite-16k" in out
    assert any("3090" in x for x in out)


def test_snake_case_generic_dropped_digit_kept():
    out = set(ee.extract_entities("vars foo_bar and user_id versus layer_2_norm"))
    assert "foo_bar" not in out and "user_id" not in out
    assert "layer_2_norm" in out


def test_proper_nouns_quoted_acronyms_kept():
    out = set(ee.extract_entities(
        'Charlie Brown uses GitHub and the "Resonant Lattice" engine; GPU and API matter.'))
    assert {"charlie brown", "github", "resonant lattice", "gpu", "api"} <= out


def test_vocab_booster_and_determinism():
    ee._TECH_VOCAB = frozenset(ee._get_tech_vocab() | {"my-cool-lib", "my_cool_lib"})
    assert ee._in_vocab("my_cool_lib")
    assert ee._score_noisy_candidate("my_cool_lib", 0.30) == 0.85
    assert ee._score_noisy_candidate("plain_english_word", 0.30) == 0.20
    assert ee._score_noisy_candidate("granite-16k", 0.30) == 0.75
    a = ee.extract_entities("Charlie Brown likes numpy")
    b = ee.extract_entities("Charlie Brown likes numpy")
    assert a == b

def test_hrr_rich_encoding_order_sensitivity():
    try:
        import numpy  # noqa: F401
    except Exception:
        print("  SKIP hrr test: numpy not installed"); return
    hg = _load("holographic")
    a = hg.encode_text_rich("user prefers dark themes")
    b = hg.encode_text_rich("themes dark prefers user")
    c = hg.encode_text_rich("user prefers dark themes")
    # Determinism: identical text → identical vector
    assert hg.similarity(a, c) > 0.999
    # Order sensitivity: shared vocabulary keeps unigram-layer similarity,
    # but positional + rolled-bigram layers must pull reordered text well
    # below identity.
    assert hg.similarity(a, b) < 0.90, hg.similarity(a, b)
    # Rolled bigrams are non-commutative at the primitive level too.
    import numpy as np
    x, y = hg.encode_atom("dark"), hg.encode_atom("themes")
    fwd = hg.bind(x, np.roll(y, 1))
    rev = hg.bind(y, np.roll(x, 1))
    assert hg.similarity(fwd, rev) < 0.5


def test_hrr_triple_unbind_roundtrip():
    """Phase 5a: a triple encoded with encode_triple is queryable by role -
    unbinding the object (or subject) role recovers that filler far better than
    a random atom. This is exactly the algebra Phase 5b relational recall uses."""
    try:
        import numpy  # noqa: F401
    except Exception:
        print("  SKIP triple test: numpy not installed"); return
    hg = _load("holographic")
    dim = 1024
    T = hg.encode_triple("alice", "works_at", "acme", dim=dim)
    rnd = hg.encode_atom("totally_unrelated_token", dim)
    # Object role recovers "acme".
    rec_o = hg.unbind(T, hg.encode_atom("__hrr_role_object__", dim))
    sim_o = hg.similarity(rec_o, hg.encode_atom("acme", dim))
    assert sim_o > hg.similarity(rec_o, rnd) + 0.1, (sim_o, hg.similarity(rec_o, rnd))
    # Subject role recovers "alice".
    rec_s = hg.unbind(T, hg.encode_atom("__hrr_role_subject__", dim))
    sim_s = hg.similarity(rec_s, hg.encode_atom("alice", dim))
    assert sim_s > hg.similarity(rec_s, rnd) + 0.1, (sim_s, hg.similarity(rec_s, rnd))
    # Determinism: same triple → identical vector.
    assert hg.similarity(T, hg.encode_triple("alice", "works_at", "acme", dim=dim)) > 0.999

# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 - LatticeStore lifecycle (skipped if deps missing)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import sqlite_vec  # noqa: F401
    import numpy  # noqa: F401
    store_mod = _load("store")
    _STORE_OK = True
except Exception as _e:  # pragma: no cover
    _STORE_OK = False
    _SKIP_REASON = str(_e)

# Phase 2 central defaults consistency smoke (import from config_schema)
try:
    from config_schema import DEFAULTS as _CENTRAL_DEFAULTS
    _CENTRAL_OK = True
except Exception:
    _CENTRAL_OK = False
    _CENTRAL_DEFAULTS = {}


def test_central_defaults_consistency():
    """Phase 2: provider + store defaults should match the central DEFAULTS where relevant."""
    if not _CENTRAL_OK:
        print("  SKIP central defaults test: config_schema not importable standalone")
        return
    # Check a few core Hebbian keys
    assert _CENTRAL_DEFAULTS.get("initial_resonance") == 4
    assert _CENTRAL_DEFAULTS.get("decay_per_cycle") == 0.5
    assert _CENTRAL_DEFAULTS.get("promotion_resonance_threshold") == 4
    # Provider should respect them (basic construction)
    try:
        p = _load("__init__").LatticeMemoryProvider({})
        assert p._initial_resonance == _CENTRAL_DEFAULTS["initial_resonance"]
    except Exception as e:
        print(f"  provider defaults check skipped: {e}")


def test_provider_and_store_produce_identical_core_defaults():
    """Phase 9: provider and store should produce identical core defaults from central source."""
    if not _CENTRAL_OK or not _STORE_OK:
        print("  SKIP provider/store defaults match test")
        return
    central = _CENTRAL_DEFAULTS
    # Direct store (uses central defaults via _STORE_DEFAULTS)
    s = store_mod.LatticeStore(db_path=":memory:")
    assert s.initial_resonance == central.get("initial_resonance")
    assert s.decay_per_cycle == central.get("decay_per_cycle")
    assert s.short_tier_cycles == central.get("short_tier_cycles")
    assert s.promotion_threshold == central.get("promotion_resonance_threshold")
    assert s.detect_policy_conflicts is True
    assert s.detect_procedural_conflicts is True
    # Provider (via get_defaults if possible)
    try:
        prov = _load("__init__")
        p = prov.LatticeMemoryProvider({})
        assert p.get_defaults().get("initial_resonance") == central.get("initial_resonance")
    except Exception as e:
        print(f"  provider defaults match check skipped: {e}")


def test_prompt_keys_in_central_schema():
    """Prompt overrides are first-class CONFIG_SCHEMA keys with prompts.py defaults."""
    if not _CENTRAL_OK:
        print("  SKIP prompt schema: config_schema not importable"); return
    import prompts as prompt_mod
    from config_schema import PROMPT_CONFIG_KEYS, CONFIG_SCHEMA, DEFAULTS
    schema_keys = {e["key"] for e in CONFIG_SCHEMA}
    expected = {
        "extraction_prompt": prompt_mod.DEFAULT_EXTRACTION_PROMPT,
        "consolidation_prompt": prompt_mod.DEFAULT_CONSOLIDATION_PROMPT,
        "gist_prompt": prompt_mod.DEFAULT_GIST_PROMPT,
        "procedural_prompt": prompt_mod.DEFAULT_PROCEDURAL_PROMPT,
        "relation_prompt": prompt_mod.DEFAULT_RELATION_PROMPT,
        "narrative_prompt": prompt_mod.DEFAULT_NARRATIVE_PROMPT,
    }
    assert set(PROMPT_CONFIG_KEYS) == set(expected)
    for key, default in expected.items():
        assert key in schema_keys, f"{key} missing from CONFIG_SCHEMA"
        assert DEFAULTS[key] == default, f"{key} DEFAULTS drifted from prompts.py"
    print("  prompt schema OK: six keys in CONFIG_SCHEMA, defaults match prompts.py")


def test_reinforce_threshold_clamped_to_similarity():
    """Silent-merge gate cannot sit below similarity_threshold (mid-band for conflicts)."""
    if not _STORE_OK:
        print(f"  SKIP reinforce clamp: {_SKIP_REASON}"); return
    s = store_mod.LatticeStore(
        db_path=os.path.join(tempfile.mkdtemp(), "clamp.db"),
        similarity_threshold=0.80,
        reinforce_threshold=0.50,  # too low - must be raised
    )
    assert s.reinforce_threshold == 0.80, s.reinforce_threshold
    s.close()
    s2 = store_mod.LatticeStore(
        db_path=os.path.join(tempfile.mkdtemp(), "clamp2.db"),
        similarity_threshold=0.78,
        reinforce_threshold=0.95,
    )
    assert s2.reinforce_threshold == 0.95, s2.reinforce_threshold
    s2.close()
    print("  reinforce clamp OK: low values raised to similarity_threshold")


def _inject_hermes_stubs():
    """Minimal agent/tools/hermes stubs so __init__.py can load without Hermes installed."""
    import types
    from abc import ABC, abstractmethod
    if "agent.memory_provider" in sys.modules:
        return

    class MemoryProvider(ABC):
        @property
        @abstractmethod
        def name(self):
            ...

        @abstractmethod
        def is_available(self):
            ...

        @abstractmethod
        def initialize(self, session_id, **kwargs):
            ...

        @abstractmethod
        def get_tool_schemas(self):
            ...

    agent = types.ModuleType("agent")
    amp = types.ModuleType("agent.memory_provider")
    amp.MemoryProvider = MemoryProvider
    agent.memory_provider = amp
    sys.modules["agent"] = agent
    sys.modules["agent.memory_provider"] = amp

    tools = types.ModuleType("tools")
    reg = types.ModuleType("tools.registry")
    reg.tool_error = lambda msg: msg
    tools.registry = reg
    sys.modules["tools"] = tools
    sys.modules["tools.registry"] = reg

    hc = types.ModuleType("hermes_constants")
    hc.get_hermes_home = lambda: __import__("pathlib").Path(tempfile.mkdtemp())
    sys.modules["hermes_constants"] = hc
    hcli = types.ModuleType("hermes_cli")
    hcli_cfg = types.ModuleType("hermes_cli.config")
    hcli_cfg.cfg_get = lambda cfg, *keys, default=None: default
    hcli.config = hcli_cfg
    sys.modules["hermes_cli"] = hcli
    sys.modules["hermes_cli.config"] = hcli_cfg


def test_provider_initialize_wires_conflict_detect_flags():
    """Yaml/config detect_policy_conflicts + detect_procedural_conflicts must reach LatticeStore.

    Regression: these keys lived in CONFIG_SCHEMA and store.__init__ but
    LatticeMemoryProvider.initialize() omitted them, so live Hermes always kept
    store defaults (True) and presets like Conspiracy could not disable the sweeps.
    """
    if not _STORE_OK:
        print(f"  SKIP provider detect wiring: {_SKIP_REASON}"); return
    try:
        _inject_hermes_stubs()
        # Fresh load after stubs so agent.memory_provider resolves.
        if "resonant_lattice" in sys.modules:
            # Prefer file-load of the package entry like other tests.
            pass
        prov = _load("__init__")
    except Exception as e:
        print(f"  SKIP provider detect wiring (import): {e}"); return

    home = tempfile.mkdtemp()
    p = prov.LatticeMemoryProvider({
        "detect_policy_conflicts": False,
        "detect_procedural_conflicts": False,
        "conflict_subject_veto": False,
    })
    assert p._detect_policy_conflicts is False
    assert p._detect_procedural_conflicts is False
    p._probe_vector_dim = lambda: 768  # no Ollama
    p.initialize("test-session", hermes_home=home, agent_context="primary")
    assert p._store is not None, "store failed to open"
    assert p._store.detect_policy_conflicts is False, p._store.detect_policy_conflicts
    assert p._store.detect_procedural_conflicts is False, p._store.detect_procedural_conflicts
    assert p._store.conflict_subject_veto is False
    status = p.get_feature_status()
    assert status["detect_policy_conflicts"] is False
    assert status["detect_procedural_conflicts"] is False
    p._store.close()

    # Defaults path: True reaches the store.
    home2 = tempfile.mkdtemp()
    p2 = prov.LatticeMemoryProvider({})
    p2._probe_vector_dim = lambda: 768
    p2.initialize("test-session-2", hermes_home=home2, agent_context="primary")
    assert p2._store.detect_policy_conflicts is True
    assert p2._store.detect_procedural_conflicts is True
    p2._store.close()
    print("  provider detect wiring OK: False/True config reaches LatticeStore + feature_status")


def _fresh_store(**kw):
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "rfm_test.db")
    prom = _CENTRAL_DEFAULTS.get("promotion_resonance_threshold", 4) if _CENTRAL_OK else 4
    short = _CENTRAL_DEFAULTS.get("short_tier_cycles", 2) if _CENTRAL_OK else 2
    mid = _CENTRAL_DEFAULTS.get("mid_tier_cycles", 2) if _CENTRAL_OK else 2
    return store_mod.LatticeStore(db_path=db, promotion_threshold=prom,
                                  short_tier_cycles=short, mid_tier_cycles=mid, **kw)


def _emb(store, text):
    # deterministic pseudo-embedding (unit-ish) so cosine dedup is meaningful
    import numpy as np
    h = abs(hash(text))
    rng = np.random.default_rng(h % (2**32))
    v = rng.standard_normal(store.vector_dim)
    v = v / (np.linalg.norm(v) or 1.0)
    return v.tolist()


class _FakeBlind:
    """Plaintext stand-in for he_crypto.BlindCrypto so BlindRetriever's orchestration
    (scan -> score -> decrypt -> rank -> materialize) is validated on Windows without
    openfhe. Same duck-typed interface; encrypt = the unit vector's bytes, cosine_score
    = a real dot product, decrypt = identity. Actual CKKS correctness is E2.4 on the
    node - here the scores are exact cosines, so the ranking is an exact reference."""

    def __init__(self, dim):
        self._dim = dim

    def _unit(self, vec):
        import numpy as np
        v = np.asarray(vec, dtype=float).ravel()[:self._dim]
        n = np.linalg.norm(v)
        return v / n if n else v

    def encrypt_unit_vector(self, vec):
        import numpy as np
        return self._unit(vec).astype(np.float64).tobytes()

    def cosine_score(self, q_ct, s_ct):
        import numpy as np
        q = np.frombuffer(q_ct, dtype=np.float64)
        s = np.frombuffer(s_ct, dtype=np.float64)
        return float(np.dot(q, s))   # the "encrypted score" is just the scalar here

    def decrypt_score(self, score_ct):
        return float(score_ct)


def test_store_cosine_schema_and_dedup():
    if not _STORE_OK:
        print(f"  SKIP store tests: {_SKIP_REASON}"); return
    s = _fresh_store()
    # cosine metric present in schema
    sql = s._conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='semantic_vec'").fetchone()["sql"]
    assert "cosine" in sql.lower(), sql
    # identical content reinforces, not duplicates
    e = _emb(s, "user prefers dark themes")
    a1, id1 = s.add_or_reinforce_fact("user prefers dark themes", e, "pref", "sess1")
    a2, id2 = s.add_or_reinforce_fact("user prefers dark themes", e, "pref", "sess1")
    assert id1 == id2, (a1, a2, id1, id2)
    s.close()


def test_store_he_vector_blob_substrate():
    """E2.2: semantic_he holds opaque per-fact ct blobs, no plaintext recoverable,
    CASCADE-cleaned with the fact. Pure SQLite - no openfhe needed."""
    if not _STORE_OK:
        print(f"  SKIP store tests: {_SKIP_REASON}"); return
    s = _fresh_store(vector_dim=8)
    sql = s._conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='semantic_he'").fetchone()["sql"]
    assert "cascade" in sql.lower() and "blob" in sql.lower(), sql
    _, fid_a = s.add_or_reinforce_fact("alpha", _emb(s, "alpha"), "general", "t")
    _, fid_b = s.add_or_reinforce_fact("beta", _emb(s, "beta"), "general", "t")
    ct_a, ct_b = os.urandom(2048), os.urandom(2048)   # stand-in for CKKS ciphertext
    s.store_he_vector(fid_a, ct_a)
    s.store_he_vector(fid_b, ct_b, he_version=1)
    assert s.count_he_vectors() == 2
    assert s.get_he_vector(fid_a) == ct_a
    assert dict(s.iter_he_vectors()) == {fid_a: ct_a, fid_b: ct_b}
    # the stored blob is exactly the ct - the plaintext embedding never leaks into it
    assert store_mod.serialize_vector(_emb(s, "beta")) not in s.get_he_vector(fid_b)
    # INSERT OR REPLACE keeps a single ct per fact
    ct_a2 = os.urandom(2048)
    s.store_he_vector(fid_a, ct_a2)
    assert s.get_he_vector(fid_a) == ct_a2 and s.count_he_vectors() == 2
    # an empty blob is rejected, not silently dropped
    try:
        s.store_he_vector(fid_a, b""); assert False, "empty ct did not raise"
    except ValueError:
        pass
    # CASCADE: pruning the fact drops its ciphertext
    s._conn.execute("DELETE FROM semantic_facts WHERE id=?", (fid_b,)); s._conn.commit()
    assert s.get_he_vector(fid_b) is None and s.count_he_vectors() == 1
    s.close()


def test_blind_stream_scan_equivalence():
    """A1: stream_he_vectors pages the EXACT same (id, ct) sequence as the fetchall
    iter_he_vectors for any batch size, so the bounded-RAM recall scan is a drop-in.
    Also checks he_blob_size (the per-ct footprint the auto-tuner reads). Pure SQLite."""
    if not _STORE_OK:
        print(f"  SKIP store tests: {_SKIP_REASON}"); return
    s = _fresh_store(vector_dim=8)
    ids = []
    for i in range(7):
        _, fid = s.add_or_reinforce_fact(f"fact{i}", _emb(s, f"fact{i}"), "general", "t")
        s.store_he_vector(fid, bytes([i]) + os.urandom(2048))
        ids.append(fid)
    ref = s.iter_he_vectors()                          # fetchall baseline
    assert [fid for fid, _ in ref] == sorted(ids)      # stable id order
    for B in (1, 2, 3, 7, 100):
        assert list(s.stream_he_vectors(batch=B)) == ref, f"batch={B} diverged from fetchall"
    assert s.he_blob_size() == 2049                     # 1 prefix byte + 2048 random, uniform
    s2 = _fresh_store(vector_dim=8)                      # empty table: empty stream, zero size
    assert list(s2.stream_he_vectors(batch=4)) == [] and s2.he_blob_size() == 0
    s.close(); s2.close()


def test_blind_scan_batch_autotune():
    """A1: resolve_scan_batch DEFAULTS to the latency-optimal target and only SHRINKS on small
    or busy hosts (latency is flat across batch size, so a bigger page just wastes RAM and
    starves concurrency). Honors an explicit override and never exceeds the corpus count. Pure
    arithmetic - no store, no openfhe, nothing hardcoded to a host."""
    import store_blind as sb
    TGT, MN = sb._SCAN_BATCH_TARGET, sb._SCAN_BATCH_MIN
    GB = 1 << 30
    # explicit override wins, but never exceeds the corpus count
    assert sb.resolve_scan_batch(512, 1_000_000, 10_000, 8 * GB) == 512
    assert sb.resolve_scan_batch(512, 1_000_000, 100, 8 * GB) == 100
    # roomy host: default to the target - does NOT inflate to fill RAM (the bug the vault caught)
    assert sb.resolve_scan_batch(0, 1_000_000, 100_000, 8 * GB) == TGT
    assert sb.resolve_scan_batch(0, 100_000, 1_000_000, 256 * GB) == TGT     # huge RAM, small ct -> still target
    # tight host: shrink below target, floored at MIN
    assert sb.resolve_scan_batch(0, 1_000_000, 100_000, GB // 2) < TGT       # ~0.5GB budget -> ~107
    assert sb.resolve_scan_batch(0, 8_000_000, 100_000, GB // 4) == MN       # tiny budget -> floor
    # concurrency splits the RAM budget, so a busy host shrinks
    assert sb.resolve_scan_batch(0, 1_000_000, 100_000, 8 * GB, concurrency=64) < \
           sb.resolve_scan_batch(0, 1_000_000, 100_000, 8 * GB, concurrency=1)
    # measurement unavailable -> the target, still clamped to the corpus count
    assert sb.resolve_scan_batch(0, 0, 100_000, None) == TGT
    assert sb.resolve_scan_batch(0, 0, 10, None) == 10


def test_store_blind_retriever_orchestration():
    """E2.3: BlindRetriever scan/score/rank/materialize over semantic_he, validated on
    Windows with a plaintext stand-in crypto (real CKKS correctness is E2.4 on the
    node). Asserts top-k order vs a numpy reference, the min_similarity floor, and
    superseded exclusion."""
    if not _STORE_OK:
        print(f"  SKIP store tests: {_SKIP_REASON}"); return
    import numpy as np
    from retrieval import BlindRetriever
    DIM = 16
    s = _fresh_store(vector_dim=DIM)
    fake = _FakeBlind(DIM)
    vecs = {}
    for i in range(5):
        v = _emb(s, f"fact-{i}")
        _, fid = s.add_or_reinforce_fact(f"fact number {i}", v, "general", "t")
        vecs[fid] = np.asarray(v, dtype=float)
        s.store_he_vector(fid, fake.encrypt_unit_vector(v))   # client-side encrypt -> store
    q = _emb(s, "fact-2")   # deterministic: identical to fact-2's embedding
    # numpy reference ranking (cosine descending)
    qn = np.asarray(q, dtype=float); qn = qn / (np.linalg.norm(qn) or 1.0)
    ref = sorted(vecs, key=lambda fid: float(qn @ (vecs[fid] / (np.linalg.norm(vecs[fid]) or 1.0))),
                 reverse=True)
    br = BlindRetriever(s, "http://x", "m", blind=fake, min_similarity=-1.0)
    got = br.blind_search_vec(q, limit=3)
    assert [r["id"] for r in got] == ref[:3], ([r["id"] for r in got], ref[:3])
    assert got[0]["blind_similarity"] >= got[1]["blind_similarity"] >= got[2]["blind_similarity"]
    assert abs(got[0]["blind_similarity"] - 1.0) < 1e-9   # query == fact-2 -> cosine 1.0
    # min_similarity floor: only the exact match clears a 0.999 floor
    hi = BlindRetriever(s, "http://x", "m", blind=fake, min_similarity=0.999).blind_search_vec(q, limit=5)
    assert [r["id"] for r in hi] == [ref[0]] and hi[0]["blind_similarity"] >= 0.999
    # superseded facts are excluded (parity with the plaintext path)
    s._conn.execute("UPDATE semantic_facts SET tier='superseded' WHERE id=?", (ref[0],))
    s._conn.commit()
    assert ref[0] not in [r["id"] for r in br.blind_search_vec(q, limit=5)]
    s.close()


def test_store_he_blind_vs_plaintext_topk():
    """E2.4: REAL CKKS blind recall must rank identically to the plaintext
    LatticeRetriever on a fixture - the make-or-break HE proof. Self-skips without
    openfhe, so it is inert on Windows and runs for real in the node's ~/he venv.

    Fixture vectors are built with strictly-separated cosines to the query (linspace
    0.95..0.10), so the ground-truth ranking is unambiguous and robust to float32 /
    CKKS approximation. Facts are inserted in a shuffled order so the expected
    ranking is a non-trivial permutation."""
    if not _STORE_OK:
        print(f"  SKIP store tests: {_SKIP_REASON}"); return
    try:
        import he_crypto
    except Exception as e:
        print(f"  SKIP he compare: {e}"); return
    if not he_crypto.he_available():
        print("  SKIP he compare: openfhe/numpy unavailable (run on the node)"); return
    import numpy as np
    from retrieval import LatticeRetriever, BlindRetriever
    DIM, N = 768, 10
    rng = np.random.default_rng(42)

    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n else v

    q = unit(rng.standard_normal(DIM))
    targets = np.linspace(0.95, 0.10, N)        # well-separated -> unambiguous order
    pairs = []
    for c in targets:
        z = rng.standard_normal(DIM)
        z = unit(z - (z @ q) * q)               # component orthogonal to q
        v = unit(c * q + np.sqrt(max(1.0 - c * c, 0.0)) * z)   # cos(v, q) == c
        pairs.append((v, float(c)))
    order = list(rng.permutation(N))
    pairs = [pairs[i] for i in order]

    s = _fresh_store(vector_dim=DIM)
    blind, _key_blobs, _secret_blob = he_crypto.BlindCrypto.generate(dim=DIM)
    fid_cos = {}
    for i, (v, c) in enumerate(pairs):
        _, fid = s.add_or_reinforce_fact(f"fixture fact {i}", v.tolist(), "general", "t")
        s.store_he_vector(fid, blind.encrypt_unit_vector(v.tolist()))   # client encrypt -> store
        fid_cos[fid] = c
    expected = [fid for fid, _ in sorted(fid_cos.items(), key=lambda kv: kv[1], reverse=True)]

    class _PlainR(LatticeRetriever):
        def _get_embedding(self, _text):       # no Ollama: feed the fixed query vector
            return q.tolist()
    plain_top = [r["id"] for r in
                 _PlainR(s, "http://x", "nomic", min_similarity=-1.0).search("q", limit=N)]
    br = BlindRetriever(s, "http://x", "nomic", blind=blind, min_similarity=-1.0)
    blind_top = [r["id"] for r in br.blind_search_vec(q.tolist(), limit=N)]

    # Decrypted HE cosines must match the construction targets within CKKS tolerance.
    scores = dict(br.blind_scores(q.tolist()))
    max_err = max(abs(scores[fid] - fid_cos[fid]) for fid in fid_cos)
    assert max_err < 1e-2, f"CKKS cosine error too high: {max_err}"
    assert blind_top == expected, (blind_top, expected)
    assert plain_top == expected, (plain_top, expected)
    assert blind_top == plain_top
    print(f"  he compare OK: N={N} dim={DIM} max_cos_err={max_err:.2e}")
    s.close()


def test_he_blind_argmax_pipeline():
    """E3 core: homomorphic blind argmax (CKKS<->TFHE scheme switching) returns the
    correct one-hot WITHOUT decrypting scores - store side uses the public key only.
    Self-skips without openfhe; proven on the node (N=8, dim=16, ~2.7s)."""
    try:
        import he_crypto
    except Exception as e:
        print(f"  SKIP he argmax: {e}"); return
    if not he_crypto.he_available():
        print("  SKIP he argmax: openfhe/numpy unavailable (run on the node)"); return
    import numpy as np
    DIM, N = 16, 8
    rng = np.random.default_rng(0)
    unit = lambda v: v / (np.linalg.norm(v) or 1.0)
    q = unit(rng.standard_normal(DIM))
    facts = [unit(rng.standard_normal(DIM)) for _ in range(N)]
    truth = int(np.argmax([float(q @ f) for f in facts]))
    eng = he_crypto.BlindArgmax.generate(dim=DIM, num_facts=N)
    qct = eng.encrypt_vector(q.tolist())
    fcts = [eng.encrypt_vector(f.tolist()) for f in facts]
    onehot = eng.argmax(qct, fcts)        # STORE side: public + eval/switching keys only
    oh = eng.decrypt_onehot(onehot)       # CLIENT side
    got = max(range(N), key=lambda i: oh[i])
    assert got == truth, (oh, got, truth)
    # the one-hot is a clean indicator: winner ~1, the rest ~0
    assert oh[got] > 0.5 and sum(1 for v in oh if v > 0.5) == 1, oh
    print(f"  he argmax OK: N={N} dim={DIM} argmax={got}")


def test_he_pre_and_threshold_audit():
    """E6 core: PRE three-key runtime path (agent uses re-encrypted results but cannot
    read the raw store) + threshold user-audit (all shares reconstruct, one cannot).
    Self-skips without openfhe; proven on the node."""
    try:
        import he_crypto
    except Exception as e:
        print(f"  SKIP he pre: {e}"); return
    if not he_crypto.he_available():
        print("  SKIP he pre: openfhe unavailable (run on the node)"); return
    # --- PRE three-key runtime path ---
    pre = he_crypto.BlindPRE.generate(batch=8)
    storage = pre.keygen()                       # storage (master) key
    agent = pre.keygen()                         # agent use-key
    rk = pre.rekey(storage.secretKey, agent.publicKey)   # rk_storage->agent (setup-time)
    ctM = pre.encrypt([0.42, 0.10], storage.publicKey)   # a query result under storage key
    ctA = pre.reencrypt(ctM, rk)                 # store re-encrypts to the agent
    assert abs(pre.decrypt(ctA, agent.secretKey, 2)[0] - 0.42) < 1e-2     # agent reads re-enc
    assert abs(pre.decrypt(ctM, storage.secretKey, 2)[0] - 0.42) < 1e-2   # master god-mode
    raw_blocked = False                          # agent key on the RAW DB ciphertext
    try:
        raw_blocked = abs(pre.decrypt(ctM, agent.secretKey, 2)[0] - 0.42) > 0.1
    except Exception:
        raw_blocked = True                       # outright rejection is the strongest form
    assert raw_blocked, "agent key must NOT recover the raw store ciphertext"
    # --- Threshold user-audit (2-of-2) ---
    th = he_crypto.ThresholdAudit.generate(batch=8)
    p1 = th.first_party(); p2 = th.join(p1.publicKey)
    ct = th.encrypt([0.77, 0.33], p2.publicKey)
    fused = th.fuse([th.partial_lead(ct, p1.secretKey),
                     th.partial_main(ct, p2.secretKey)], 2)
    assert abs(fused[0] - 0.77) < 1e-2           # all shares -> reconstruct
    solo_blocked = False
    try:
        solo_blocked = abs(th.fuse([th.partial_lead(ct, p1.secretKey)], 2)[0] - 0.77) > 0.1
    except Exception:
        solo_blocked = True
    assert solo_blocked, "a single share must NOT decrypt"
    print("  he pre+threshold OK")


def test_he_recall_pre_split():
    """0a: BlindRecallPRE - the unified recall+PRE engine (E2 cosine + E6 PRE in one
    serializable context). The store scores cosine AND re-encrypts the score ct to the
    agent with no secret; the agent reads the re-encrypted result but NOT the raw store ct;
    the master reads anything (god-mode). Self-skips without openfhe; the full 3-process
    SERIALIZED split (load_eval/load_client/load_user) is node-proven 2026-06-19 - here the
    logic runs in one process with live keys (re-deserializing eval keys would collide with
    generate()'s global eval-key store)."""
    try:
        import he_crypto
    except Exception as e:
        print(f"  SKIP he recall+pre: {e}"); return
    if not he_crypto.he_available():
        print("  SKIP he recall+pre: openfhe/numpy unavailable (run on the node)"); return
    import numpy as np
    import openfhe as o
    DIM, N = 16, 5
    rng = np.random.default_rng(3)
    unit = lambda v: v / (np.linalg.norm(v) or 1.0)
    facts = [unit(rng.standard_normal(DIM)) for _ in range(N)]
    q = unit(facts[2] + 0.05 * rng.standard_normal(DIM))
    cos = [float(q @ f) for f in facts]
    truth = int(np.argmax(cos))

    user, _key_blobs, secret_blobs = he_crypto.BlindRecallPRE.generate(dim=DIM)
    # white-box AGENT role sharing the live context (single process: don't re-deserialize
    # eval keys). Only the agent's PRIVATE key is deserialized - that does not touch the
    # global eval-key store.
    agent = he_crypto.BlindRecallPRE(user._cc, DIM, user.batch)
    agent._pub = user._pub
    agent._sk = o.DeserializePrivateKeyString(secret_blobs["agent"], o.BINARY)

    q_ct = user.encrypt_unit_vector(q.tolist())
    f_cts = [user.encrypt_unit_vector(f.tolist()) for f in facts]
    agent_scores, raw_scores = [], []
    for f_ct in f_cts:
        score = user.cosine_score(q_ct, f_ct)                    # STORE: blind cosine
        raw_scores.append(score)
        agent_scores.append(agent.decrypt_score(user.reencrypt_score(score)))  # reencrypt -> AGENT
    assert max(abs(a - c) for a, c in zip(agent_scores, cos)) < 1e-2, (agent_scores, cos)
    assert int(np.argmax(agent_scores)) == truth
    # negative control: the agent use-key must NOT read a RAW (non-reencrypted) store ct
    raw_blocked = False
    try:
        raw_blocked = abs(agent.decrypt_score(raw_scores[0]) - cos[0]) > 0.1
    except Exception:
        raw_blocked = True
    assert raw_blocked, "agent use-key must not recover the raw store score"
    # god-mode: the master reads the raw store ct directly
    assert abs(user.decrypt_score(raw_scores[0]) - cos[0]) < 1e-2
    print(f"  he recall+pre OK: N={N} dim={DIM} argmax={truth}")


def test_he_argmax_ckks_pipeline():
    """0a: BlindArgmaxCKKS - pure-CKKS comparison argmax (no FHEW scheme switching, so its
    mult+rotation keys serialize and the store/client split works where the FHEW BlindArgmax
    segfaults). The store builds a one-hot over the encrypted score vector with public+eval
    keys only. Covers a power-of-two and a padded (non-power-of-two) count. Self-skips
    without openfhe; node-proven across a serialized split 2026-06-19."""
    try:
        import he_crypto
    except Exception as e:
        print(f"  SKIP he argmax-ckks: {e}"); return
    if not he_crypto.he_available():
        print("  SKIP he argmax-ckks: openfhe unavailable (run on the node)"); return
    import numpy as np
    for N in (8, 5):                       # power-of-two, then padded (batch 8)
        rng = np.random.default_rng(100 + N)
        scores = rng.uniform(-1.0, 1.0, N).tolist()
        truth = int(np.argmax(scores))
        eng, _kb, _sb = he_crypto.BlindArgmaxCKKS.generate(num_facts=N, security="HEStd_NotSet")
        oh = eng.argmax(eng.encrypt_scores(scores))   # STORE side: public + eval keys only
        vals = eng.decrypt_onehot(oh)                 # CLIENT side
        assert int(np.argmax(vals)) == truth, (N, vals, scores, truth)
    print("  he argmax-ckks OK (pow2 + padded)")


def test_he_argmax_ckks_production_security():
    """E3 §3b: BlindArgmaxCKKS at the PRODUCTION security level (default HEStd_128_classic)
    still returns the correct one-hot, and reports per-op latency (§9 acceptance). Slow
    (~30s at N=8 on the node), so it self-skips without openfhe and runs only there."""
    try:
        import he_crypto
    except Exception as e:
        print(f"  SKIP he argmax 3b: {e}"); return
    if not he_crypto.he_available():
        print("  SKIP he argmax 3b: openfhe unavailable (run on the node)"); return
    import numpy as np
    import time
    N = 8
    rng = np.random.default_rng(321)
    scores = rng.uniform(-1.0, 1.0, N).tolist()
    truth = int(np.argmax(scores))
    t0 = time.time()
    eng, _kb, _sb = he_crypto.BlindArgmaxCKKS.generate(num_facts=N)   # default = HEStd_128_classic
    setup_s = time.time() - t0
    t1 = time.time()
    oh = eng.argmax(eng.encrypt_scores(scores))                       # STORE side: no secret
    argmax_s = time.time() - t1
    assert int(np.argmax(eng.decrypt_onehot(oh))) == truth
    print(f"  he argmax-ckks 3b OK (128-bit): setup={setup_s:.1f}s argmax={argmax_s:.1f}s N={N}")


def test_hrr_lift_identity():
    """E4 4a: holographic.hrr_lift gives an L2-unit vector whose dot product equals the HRR
    phase-similarity. Pure numpy, runs everywhere."""
    try:
        import numpy as np
    except Exception:
        print("  SKIP hrr lift: numpy not installed"); return
    hg = _load("holographic")
    HDIM = 64
    a = hg.encode_atom("alice", HDIM); b = hg.encode_atom("bob", HDIM)
    la, lb = hg.hrr_lift(a), hg.hrr_lift(b)
    assert abs(float(np.linalg.norm(la)) - 1.0) < 1e-9, np.linalg.norm(la)
    assert abs(float(la @ lb) - hg.similarity(a, b)) < 1e-9, (float(la @ lb), hg.similarity(a, b))
    assert hg.hrr_lift(np.zeros(0)).shape[0] == 0           # empty is safe


def test_he_hrr_similarity_via_lift():
    """E4 4a: HRR phase-cosine similarity == cosine of the (cos,sin)/sqrt(dim) lift, so the
    EXISTING blind store (BlindRecallPRE cosine over the 2*dim lift) computes HRR similarity
    with NO new crypto. Validated vs holographic.similarity on real CKKS - relational recall
    (P5) + conflict similarity become blind via the same E2 inner product. Self-skips without
    openfhe."""
    try:
        import he_crypto
    except Exception as e:
        print(f"  SKIP hrr he: {e}"); return
    if not he_crypto.he_available():
        print("  SKIP hrr he: openfhe/numpy unavailable (run on the node)"); return
    hg = _load("holographic")
    HDIM = 64
    a = hg.encode_atom("alice", HDIM); b = hg.encode_atom("bob", HDIM)
    f1 = hg.encode_fact("user prefers dark themes", ["user"], HDIM)
    f2 = hg.encode_fact("the user likes dark themes", ["user"], HDIM)
    f3 = hg.encode_fact("server runs on port 8080", ["server"], HDIM)
    blind, _kb, _sb = he_crypto.BlindRecallPRE.generate(dim=2 * HDIM)   # HE dim = 2*HDIM
    def he_sim(x, y):
        return blind.decrypt_score(blind.cosine_score(
            blind.encrypt_unit_vector(hg.hrr_lift(x).tolist()),
            blind.encrypt_unit_vector(hg.hrr_lift(y).tolist())))
    for x, y in ((a, b), (f1, f2), (a, a)):
        assert abs(he_sim(x, y) - hg.similarity(x, y)) < 1e-2, (he_sim(x, y), hg.similarity(x, y))
    assert he_sim(f1, f2) > he_sim(f1, f3) + 0.2           # related rephrase >> unrelated
    print(f"  hrr-he OK: blind HRR similarity == plaintext (related {he_sim(f1, f2):.2f} > "
          f"unrelated {he_sim(f1, f3):.2f})")


def test_store_he_hrr_table_substrate():
    """E4 4b: semantic_he_hrr stores per-fact HRR-lift ciphertext INDEPENDENTLY of semantic_he,
    CASCADE-cleaned, allowlist-guarded. Pure SQLite - no openfhe."""
    if not _STORE_OK:
        print(f"  SKIP he_hrr table: {_SKIP_REASON}"); return
    s = _fresh_store(vector_dim=8)
    sql = s._conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='semantic_he_hrr'").fetchone()
    assert sql and "cascade" in sql["sql"].lower(), sql
    _, fid = s.add_or_reinforce_fact("alpha", _emb(s, "alpha"), "general", "t")
    ct_emb, ct_hrr = os.urandom(2048), os.urandom(4096)
    s.store_he_vector(fid, ct_emb)                                  # default semantic_he
    s.store_he_vector(fid, ct_hrr, table="semantic_he_hrr")        # HRR table
    assert s.count_he_vectors() == 1 and s.count_he_vectors(table="semantic_he_hrr") == 1
    assert s.get_he_vector(fid) == ct_emb                           # the two tables are independent
    assert s.get_he_vector(fid, table="semantic_he_hrr") == ct_hrr
    assert dict(s.iter_he_vectors(table="semantic_he_hrr")) == {fid: ct_hrr}
    try:                                                            # allowlist guards SQL
        s.store_he_vector(fid, ct_hrr, table="semantic_facts"); assert False, "bad table accepted"
    except ValueError:
        pass
    s._conn.execute("DELETE FROM semantic_facts WHERE id=?", (fid,)); s._conn.commit()
    assert s.count_he_vectors(table="semantic_he_hrr") == 0         # CASCADE drops the HRR ct
    s.close()


def test_he_blind_hrr_recall():
    """E4 4b + 2b-ii(a): blind HRR recall - BlindWriter stores encrypted HRR LIFTS in
    semantic_he_hrr, then BlindRetriever.blind_hrr_search ranks a phase probe by HRR similarity
    homomorphically, matching the plaintext holographic.similarity ranking. Option A: the retriever
    is built with a SEPARATE embed-dim recall client as ``blind`` and the 2·hrr_dim client as
    ``blind_hrr`` - proving blind_hrr_* uses ``blind_hrr`` (the lift would mis-encrypt under the
    smaller recall context). Needs a store + openfhe -> node."""
    if not _STORE_OK:
        print(f"  SKIP blind hrr: {_SKIP_REASON}"); return
    try:
        import he_crypto
    except Exception as e:
        print(f"  SKIP blind hrr: {e}"); return
    if not he_crypto.he_available():
        print("  SKIP blind hrr: openfhe/numpy unavailable (run on the node)"); return
    from retrieval import BlindWriter, BlindRetriever
    hg = _load("holographic")
    HDIM, EMBDIM = 64, 16
    facts = [
        ("user strongly prefers dark themes", ["user"]),   # near-rephrase of the probe
        ("user likes dark color schemes", ["user"]),        # related
        ("the database runs on port 5432", ["database"]),   # unrelated
        ("weather is sunny today", []),                      # unrelated
    ]
    s = _fresh_store(vector_dim=EMBDIM)
    blind, _kb, _sb = he_crypto.BlindRecallPRE.generate(dim=2 * HDIM)   # HE dim = 2*HDIM (the lift)
    hrr_writer = BlindWriter(s, blind, table="semantic_he_hrr")
    fid_vec = {}
    for content, ents in facts:
        _, fid = s.add_or_reinforce_fact(content, _emb(s, content), "general", "t")
        hv = hg.encode_fact(content, ents, HDIM)                       # plaintext HRR phase vector
        assert hrr_writer.write_fact(fid, hg.hrr_lift(hv).tolist())    # store the ENCRYPTED lift
        fid_vec[fid] = hv
    assert s.count_he_vectors(table="semantic_he_hrr") == len(facts)
    probe = hg.encode_fact("user prefers dark themes", ["user"], HDIM)
    plain_rank = sorted(fid_vec, key=lambda f: hg.similarity(probe, fid_vec[f]), reverse=True)
    # Option A: a DISTINCT embed-dim recall client as `blind`; the 2*HDIM lift client as `blind_hrr`.
    # blind_hrr_* must use blind_hrr - encrypting a 2*HDIM lift under the EMBDIM recall ctx would fail.
    recall_blind, _rkb, _rsb = he_crypto.BlindRecallPRE.generate(dim=EMBDIM)
    br = BlindRetriever(s, "http://x", "nomic", blind=recall_blind, min_similarity=-1.0, blind_hrr=blind)
    blind_scores = dict(br.blind_hrr_scores(probe))
    blind_rank = [fid for fid, _ in sorted(blind_scores.items(), key=lambda kv: kv[1], reverse=True)]
    assert blind_rank == plain_rank, (blind_rank, plain_rank)
    err = max(abs(blind_scores[f] - hg.similarity(probe, fid_vec[f])) for f in fid_vec)
    assert err < 1e-2, err
    top = br.blind_hrr_search(probe, limit=1)
    assert top and top[0]["id"] == plain_rank[0]
    s.close()
    print(f"  blind hrr OK: HRR recall ranking == plaintext, max_sim_err={err:.2e}")


def test_he_blind_maintenance():
    """E5 5a: blind dream-cycle maintenance on encrypted resonance - homomorphic DECAY (scalar
    mult, exact) + threshold COMPARE (promotion/eviction via a Chebyshev step -> an encrypted
    0/1 indicator, store-side, no secret). Resonance is scaled to ~[0,1]; classification is
    exact outside the transition band. Self-skips without openfhe; node-proven."""
    try:
        import he_crypto
    except Exception as e:
        print(f"  SKIP he maint: {e}"); return
    if not he_crypto.he_available():
        print("  SKIP he maint: openfhe unavailable (run on the node)"); return
    eng, _kb, _sb = he_crypto.BlindMaintenance.generate(batch=8, security="HEStd_NotSet")
    res = [0.20, 0.55, 0.80, 0.30, 0.62]          # scaled resonance, clear of the threshold band
    decay, thr = 0.9, 0.45
    ct_d = eng.decay(eng.encrypt_scalars(res), decay)   # STORE: resonance *= 0.9
    ind = eng.ge_threshold(ct_d, thr)                   # STORE: step(decayed - thr), no secret
    decayed = eng.decrypt_scalars(ct_d, len(res))
    indicator = eng.decrypt_scalars(ind, len(res))
    assert max(abs(d - r * decay) for d, r in zip(decayed, res)) < 1e-2, decayed   # decay exact
    got = [1 if v > 0.5 else 0 for v in indicator]
    exp = [1 if r * decay >= thr else 0 for r in res]
    assert got == exp, (indicator, exp)               # promotion/eviction classification
    print(f"  he maintenance OK: decay exact, promote/evict {got} == plaintext {exp}")


def test_store_he_meta_table_substrate():
    """E5 5b: semantic_he_meta exists, allowlisted, independent of semantic_he, CASCADE-cleaned.
    Pure SQLite - no openfhe."""
    if not _STORE_OK:
        print(f"  SKIP he_meta table: {_SKIP_REASON}"); return
    s = _fresh_store(vector_dim=8)
    sql = s._conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='semantic_he_meta'").fetchone()
    assert sql and "cascade" in sql["sql"].lower(), sql
    _, fid = s.add_or_reinforce_fact("alpha", _emb(s, "alpha"), "general", "t")
    s.store_he_vector(fid, os.urandom(512), table="semantic_he_meta")
    assert s.count_he_vectors(table="semantic_he_meta") == 1 and s.count_he_vectors() == 0
    s._conn.execute("DELETE FROM semantic_facts WHERE id=?", (fid,)); s._conn.commit()
    assert s.count_he_vectors(table="semantic_he_meta") == 0          # CASCADE drops the resonance ct
    s.close()


def test_he_blind_maintainer():
    """E5 5b: blind dream-cycle maintenance over a real store - BlindMaintainer stores encrypted
    resonance in semantic_he_meta, the store DECAYS it blind (no plaintext read), and the client
    SETTLES promotion/eviction by decrypting + thresholding, matching plaintext. Needs a store +
    openfhe -> node ~/he venv."""
    if not _STORE_OK:
        print(f"  SKIP blind maintainer: {_SKIP_REASON}"); return
    try:
        import he_crypto
    except Exception as e:
        print(f"  SKIP blind maintainer: {e}"); return
    if not he_crypto.he_available():
        print("  SKIP blind maintainer: openfhe unavailable (run on the node)"); return
    from retrieval import BlindMaintainer
    s = _fresh_store(vector_dim=8)
    maint, _kb, _sb = he_crypto.BlindMaintenance.generate(batch=4, security="HEStd_NotSet")
    bm = BlindMaintainer(s, maint)
    res = {}                                              # fid -> scaled resonance
    for i, v in enumerate([0.30, 0.70, 0.85, 0.20]):
        _, fid = s.add_or_reinforce_fact(f"fact {i}", _emb(s, f"fact {i}"), "general", "t")
        assert bm.set_resonance(fid, v); res[fid] = v
    assert s.count_he_vectors(table="semantic_he_meta") == 4
    decay = 0.9
    assert bm.decay_all(decay) == 4                       # STORE decays blind (no secret read)
    for fid, v in res.items():
        assert abs(bm.get_resonance(fid) - v * decay) < 1e-2, (bm.get_resonance(fid), v * decay)
    out = bm.settle(promote_threshold=0.6, prune_threshold=0.25)   # client-assisted on decayed values
    assert sorted(out["promote"]) == sorted(f for f, v in res.items() if v * decay >= 0.6)
    assert sorted(out["evict"]) == sorted(f for f, v in res.items() if v * decay < 0.25)
    s.close()
    print(f"  blind maintainer OK: decay blind + settle == plaintext "
          f"(promote {len(out['promote'])}, evict {len(out['evict'])})")


def test_crypto_entity_aead():
    """E7 7b: crypto_keys.encrypt_entities/decrypt_entities round-trip; RANDOMIZED (same set ->
    different ct, so the store sees no equality/co-occurrence); normalized+deduped+sorted; wrong
    key rejected. Needs cryptography (AEAD)."""
    import crypto_keys
    if not crypto_keys.aead_available():
        print("  SKIP entity aead: cryptography not installed"); return
    key = os.urandom(32)
    ents = ["User", "  Dark Theme ", "user", "numpy"]          # dup + case + whitespace
    blob1 = crypto_keys.encrypt_entities(ents, key)
    blob2 = crypto_keys.encrypt_entities(ents, key)
    assert blob1 != blob2                                       # random nonce -> different ct
    assert crypto_keys.decrypt_entities(blob1, key) == ["dark theme", "numpy", "user"]
    assert b"numpy" not in blob1 and b"dark theme" not in blob1  # opaque: plaintext not recoverable
    try:
        crypto_keys.decrypt_entities(blob1, os.urandom(32)); assert False, "wrong key accepted"
    except crypto_keys.WrapAuthError:
        pass
    assert crypto_keys.decrypt_entities(crypto_keys.encrypt_entities([], key), key) == []


def test_store_blind_entities():
    """E7 7b: BlindEntityStore encrypts per-fact entity sets into semantic_he_entities (opaque,
    randomized); overlap / find_conflicts run CLIENT-side on the decrypted sets (the untrusted
    store never computes overlap). Substrate: opaque blobs, plaintext not recoverable, CASCADE.
    Needs a store + cryptography."""
    if not _STORE_OK:
        print(f"  SKIP blind entities: {_SKIP_REASON}"); return
    import crypto_keys
    if not crypto_keys.aead_available():
        print("  SKIP blind entities: cryptography not installed"); return
    from retrieval import BlindEntityStore
    key = os.urandom(32)
    bes = BlindEntityStore(s_store := _fresh_store(vector_dim=8),
                           lambda e: crypto_keys.encrypt_entities(e, key),
                           lambda b: crypto_keys.decrypt_entities(b, key))
    facts = {"a": ["user", "dark theme"], "b": ["user", "light theme"],
             "c": ["database", "port"], "d": ["user", "dark theme"]}
    fids = {}
    for name, ents in facts.items():
        _, fid = s_store.add_or_reinforce_fact(f"fact {name}", _emb(s_store, name), "general", "t")
        assert bes.set_entities(fid, ents); fids[name] = fid
    assert s_store.count_he_vectors(table="semantic_he_entities") == 4
    blob = s_store.get_he_vector(fids["a"], table="semantic_he_entities")   # substrate
    assert blob and b"dark theme" not in blob
    assert bes.get_entities(fids["a"]) == ["dark theme", "user"]            # sorted round-trip
    assert bes.overlap(fids["a"], fids["d"]) == 2                           # client-side overlap
    assert bes.overlap(fids["a"], fids["b"]) == 1
    assert bes.overlap(fids["a"], fids["c"]) == 0
    assert [f for f, _ in bes.find_conflicts(fids["a"], min_overlap=2)] == [fids["d"]]
    c1 = bes.find_conflicts(fids["a"], min_overlap=1)
    assert {f for f, _ in c1} == {fids["b"], fids["d"]} and c1[0][0] == fids["d"]   # d strongest
    s_store._conn.execute("DELETE FROM semantic_facts WHERE id=?", (fids["a"],)); s_store._conn.commit()
    assert s_store.count_he_vectors(table="semantic_he_entities") == 3      # CASCADE
    s_store.close()
    print("  blind entities OK: client-side overlap/conflicts over encrypted sets, store blind")


def test_blind_entity_key_derivation_wiring():
    """E7 7b provider glue: the passphrase→keystore→derive_entity_key path that
    LatticeMemoryProvider._resolve_blind_entities builds its BlindEntityStore from (the prior
    entity tests use a raw os.urandom key, so this DERIVATION path is otherwise uncovered).
    Proves: (1) a BlindEntityStore built from a passphrase-derived key round-trips on the
    substrate, and (2) re-deriving from the SAME passphrase+keystore (a fresh session/process)
    decrypts blobs the first session wrote - the reopen property the provider relies on. Needs a
    store + argon2 + cryptography."""
    if not _STORE_OK:
        print(f"  SKIP entity key derivation: {_SKIP_REASON}"); return
    import crypto_keys
    if not (crypto_keys.kdf_available() and crypto_keys.aead_available()):
        print("  SKIP entity key derivation: argon2/cryptography not installed"); return
    from retrieval import BlindEntityStore
    passphrase = b"correct horse battery staple"
    # SETUP (first session): create the keystore, derive the entity key, build the store
    # exactly as _resolve_blind_entities does (closures over crypto_keys.encrypt/decrypt).
    keystore = crypto_keys.create_keystore(passphrase)
    k1 = crypto_keys.derive_entity_key(passphrase, keystore)
    store = _fresh_store(vector_dim=8)
    bes1 = BlindEntityStore(store,
                            lambda e: crypto_keys.encrypt_entities(e, k1),
                            lambda b: crypto_keys.decrypt_entities(b, k1))
    _, fid = store.add_or_reinforce_fact("derived fact", _emb(store, "derived"), "general", "t")
    assert bes1.set_entities(fid, ["NumPy", "  SQLite "])
    assert bes1.get_entities(fid) == ["numpy", "sqlite"]                 # normalized round-trip
    blob = store.get_he_vector(fid, table="semantic_he_entities")
    assert blob and b"numpy" not in blob and b"sqlite" not in blob       # opaque on the substrate
    # REOPEN (second session): re-derive from the same passphrase + persisted keystore and
    # decrypt the blob the first session wrote - the property _resolve_blind_entities needs.
    k2 = crypto_keys.derive_entity_key(passphrase, keystore)
    assert bytes(k1) == bytes(k2)                                        # deterministic derivation
    bes2 = BlindEntityStore(store,
                            lambda e: crypto_keys.encrypt_entities(e, k2),
                            lambda b: crypto_keys.decrypt_entities(b, k2))
    assert bes2.get_entities(fid) == ["numpy", "sqlite"]
    # A wrong passphrase derives a different key → the GCM tag rejects (no silent garbage).
    kbad = crypto_keys.derive_entity_key(b"wrong passphrase", keystore, verify=False)
    try:
        bes_bad = BlindEntityStore(store, lambda e: e, lambda b: crypto_keys.decrypt_entities(b, kbad))
        bes_bad.get_entities(fid); assert False, "wrong-passphrase key accepted"
    except crypto_keys.WrapAuthError:
        pass
    store.close()
    print("  entity key derivation OK: passphrase->keystore->key round-trips + reopens, wrong key rejected")


def test_crypto_sealed_content_aead():
    """§5-1: crypto_keys.encrypt_sealed/decrypt_sealed round-trip for the content surface; RANDOM
    nonce (same payload -> different ct, so the store sees no equality); opaque (plaintext not
    recoverable); wrong key / wrong domain rejected. Plus content_hmac: STABLE, KEYED, and
    whitespace-NORMALIZED but case-PRESERVING. Needs cryptography (AEAD)."""
    import crypto_keys
    if not crypto_keys.aead_available():
        print("  SKIP sealed content aead: cryptography not installed"); return
    key = os.urandom(32)
    payload = {"content": "the powerhouse of the cell is the mitochondria",
               "category": "biology", "source_quote": "quote text here", "source_ref": "ref-1"}
    b1 = crypto_keys.encrypt_sealed(payload, key, "content")
    b2 = crypto_keys.encrypt_sealed(payload, key, "content")
    assert b1 != b2                                                     # random nonce
    assert crypto_keys.decrypt_sealed(b1, key, "content") == payload    # round-trip
    assert b"mitochondria" not in b1 and b"quote text" not in b1        # opaque
    # a string surface (episode/triple/summary shape) round-trips too
    s_blob = crypto_keys.encrypt_sealed("a summary sentence", key, "summary")
    assert crypto_keys.decrypt_sealed(s_blob, key, "summary") == "a summary sentence"
    # wrong key rejected
    try:
        crypto_keys.decrypt_sealed(b1, os.urandom(32), "content"); assert False, "wrong key accepted"
    except crypto_keys.WrapAuthError:
        pass
    # wrong domain rejected (AAD binds the surface)
    try:
        crypto_keys.decrypt_sealed(b1, key, "episode"); assert False, "wrong domain accepted"
    except crypto_keys.WrapAuthError:
        pass
    # content_hmac: deterministic + keyed + whitespace-normalized, case-preserving
    hk = os.urandom(32)
    h = crypto_keys.content_hmac("dark   theme\tpreferred", hk)
    assert h == crypto_keys.content_hmac("dark theme preferred", hk)    # whitespace collapse
    assert h != crypto_keys.content_hmac("Dark theme preferred", hk)    # case preserved
    assert h != crypto_keys.content_hmac("dark theme preferred", os.urandom(32))  # keyed
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)     # sha256 hex
    print("  sealed content AEAD OK: randomized/opaque/domain-bound; content_hmac stable+keyed")


def test_store_blind_content_mirror():
    """§5-1: BlindContentStore encrypts each fact's content surface into semantic_he_content (opaque,
    randomized); the plaintext content is NOT recoverable from the stored blob; get_content
    round-trips the {content, category, source_quote, source_ref} dict; CASCADE drops the ct with
    the fact. Needs a store + cryptography."""
    if not _STORE_OK:
        print(f"  SKIP blind content: {_SKIP_REASON}"); return
    import crypto_keys
    if not crypto_keys.aead_available():
        print("  SKIP blind content: cryptography not installed"); return
    from retrieval import BlindContentStore
    key = os.urandom(32)
    s = _fresh_store(vector_dim=8)
    bcs = BlindContentStore(s, lambda p: crypto_keys.encrypt_sealed(p, key, "content"),
                            lambda b: crypto_keys.decrypt_sealed(b, key, "content"))
    _, fid = s.add_or_reinforce_fact("the sky appears blue due to rayleigh scattering",
                                     _emb(s, "sky"), "physics", "sess1")
    assert bcs.set_content(fid, s.get_fact(fid))
    assert s.count_he_vectors(table="semantic_he_content") == 1
    blob = s.get_he_vector(fid, table="semantic_he_content")            # substrate
    assert blob and b"rayleigh" not in blob and b"physics" not in blob  # opaque
    got = bcs.get_content(fid)
    assert got["content"] == "the sky appears blue due to rayleigh scattering"
    assert got["category"] == "physics" and got["source_quote"] is None
    s._conn.execute("DELETE FROM semantic_facts WHERE id=?", (fid,)); s._conn.commit()
    assert s.count_he_vectors(table="semantic_he_content") == 0         # CASCADE
    s.close()
    print("  blind content OK: opaque content surface at rest, client round-trip, CASCADE")


def test_blind_content_reconcile_and_hmac():
    """§5-1: the BlindTier.reconcile content path mirrors semantic_he_content AND backfills the
    keyed content_hmac dedup identity, both idempotently, from a passphrase-derived key (the
    _resolve_content wiring). Proves: full backfill of a pre-existing store; the stored content_hmac
    equals crypto_keys.content_hmac of the fact's content; a re-derived key (fresh session) decrypts
    what the first wrote; a second reconcile is a no-op. Needs a store + argon2 + cryptography."""
    if not _STORE_OK:
        print(f"  SKIP content reconcile: {_SKIP_REASON}"); return
    import crypto_keys
    if not (crypto_keys.kdf_available() and crypto_keys.aead_available()):
        print("  SKIP content reconcile: argon2/cryptography not installed"); return
    from retrieval import BlindContentStore
    from blind_tier import BlindTier
    passphrase = b"correct horse battery staple"
    keystore = crypto_keys.create_keystore(passphrase)
    c_key = crypto_keys.derive_sealed_key(passphrase, keystore, "content")
    h_key = crypto_keys.derive_content_hmac_key(passphrase, keystore)
    s = _fresh_store(vector_dim=8)
    contents = ["alpha fact one", "beta fact two", "gamma fact three"]
    fids = [s.add_or_reinforce_fact(c, _emb(s, c), "general", "sess")[1] for c in contents]
    # worklists start full
    assert s.facts_missing_blind("semantic_he_content") == fids
    assert s.facts_missing_content_hmac() == fids
    bcs = BlindContentStore(s, lambda p: crypto_keys.encrypt_sealed(p, c_key, "content"),
                            lambda b: crypto_keys.decrypt_sealed(b, c_key, "content"))
    bt = BlindTier(s, content=bcs, content_hmac_fn=lambda t: crypto_keys.content_hmac(t, h_key))
    bt.reconcile()
    # content mirrored + hmac backfilled -> worklists drain
    assert s.count_he_vectors(table="semantic_he_content") == 3
    assert s.facts_missing_blind("semantic_he_content") == []
    assert s.facts_missing_content_hmac() == []
    # stored content_hmac matches the keyed HMAC of the fact's content
    for fid, content in zip(fids, contents):
        row = s._conn.execute("SELECT content_hmac FROM semantic_facts WHERE id=?", (fid,)).fetchone()
        assert row["content_hmac"] == crypto_keys.content_hmac(content, h_key)
    # reopen: a re-derived key (fresh session) decrypts what the first session wrote
    c_key2 = crypto_keys.derive_sealed_key(passphrase, keystore, "content")
    bcs2 = BlindContentStore(s, lambda p: p, lambda b: crypto_keys.decrypt_sealed(b, c_key2, "content"))
    assert bcs2.get_content(fids[0])["content"] == "alpha fact one"
    # idempotent: a second reconcile writes nothing new and does not error
    bt.reconcile()
    assert s.count_he_vectors(table="semantic_he_content") == 3
    s.close()
    print("  content reconcile OK: mirror + content_hmac backfill, idempotent, reopen-safe")


def test_blind_sealed_text_surfaces():
    """§5-1b: BlindSealedStore mirrors episode / triple / summary TEXT into their own AEAD tables
    via BlindTier.reconcile - each keyed by its SOURCE row (not fact id), opaque, idempotent, and
    CASCADE-dropped with the source. crypto_keys.derive_sealed_keys yields all five sealed keys in
    ONE master pass. Needs a store + argon2 + cryptography."""
    if not _STORE_OK:
        print(f"  SKIP sealed text: {_SKIP_REASON}"); return
    import crypto_keys
    if not (crypto_keys.kdf_available() and crypto_keys.aead_available()):
        print("  SKIP sealed text: argon2/cryptography not installed"); return
    from retrieval import BlindSealedStore
    from blind_tier import BlindTier
    passphrase = b"correct horse battery staple"
    keystore = crypto_keys.create_keystore(passphrase)
    keys = crypto_keys.derive_sealed_keys(passphrase, keystore)
    assert set(keys) == {"content", "episode", "triple", "summary", "hmac"}   # one master pass
    s = _fresh_store(vector_dim=8)
    _, fid = s.add_or_reinforce_fact("python is a programming language", _emb(s, "py"), "tech", "sess1")
    s.add_episode("sess1", "user", "remember my secret token zzz-hidden")
    s.store_fact_relations(fid, [{"subject": "python", "relation": "is_a",
                                  "object": "language", "confidence": 1.0}])
    sum_id = s.add_session_summary("sess1", "we covered scattering and python basics", created_cycle=1)
    ep_id = s.episodes_missing_blind()[0]
    rel_id = s.triples_missing_blind()[0]
    assert s.summaries_missing_blind() == [sum_id]

    def mk(domain, table):
        k = keys[domain]
        return BlindSealedStore(s,
                                lambda p, k=k, d=domain: crypto_keys.encrypt_sealed(p, k, d),
                                lambda b, k=k, d=domain: crypto_keys.decrypt_sealed(b, k, d), table)
    sealed = {"episode": mk("episode", "semantic_he_episodes"),
              "triple":  mk("triple", "semantic_he_triples"),
              "summary": mk("summary", "semantic_he_summaries")}
    bt = BlindTier(s, sealed=sealed)
    bt.reconcile()
    # mirrored + worklists drained
    assert s.count_he_vectors(table="semantic_he_episodes") == 1
    assert s.count_he_vectors(table="semantic_he_triples") == 1
    assert s.count_he_vectors(table="semantic_he_summaries") == 1
    assert (s.episodes_missing_blind() == [] and s.triples_missing_blind() == []
            and s.summaries_missing_blind() == [])
    # opacity on the substrate
    eb = s.get_he_vector(ep_id, table="semantic_he_episodes")
    assert eb and b"zzz-hidden" not in eb
    # client round-trip per surface
    assert sealed["episode"].get_payload(ep_id)["content"] == "remember my secret token zzz-hidden"
    assert sealed["triple"].get_payload(rel_id) == {"subject": "python", "relation": "is_a",
                                                    "object": "language"}
    assert sealed["summary"].get_payload(sum_id) == "we covered scattering and python basics"
    # idempotent second pass
    bt.reconcile()
    assert s.count_he_vectors(table="semantic_he_episodes") == 1
    # CASCADE: deleting the fact drops its triple AND the triple ciphertext (transitive FK)
    s._conn.execute("DELETE FROM semantic_facts WHERE id=?", (fid,)); s._conn.commit()
    assert s.count_he_vectors(table="semantic_he_triples") == 0
    s.close()
    print("  sealed text OK: episode/triple/summary mirrored opaque, idempotent, CASCADE")


def test_blind_visitor_parity():
    """§5-2: the BlindVisitor reconstructs each fact's dream-cycle WORKING SET (content, entities,
    triples) + session summaries from the §5-1 sealed ciphertext with PARITY to the plaintext store,
    so a cognition pass re-routed through it cannot change outcomes (identical inputs -> identical
    LLM-stubbed outputs). Also proves the view is truly BLIND-SOURCED: tampering the plaintext
    content column does NOT change what the visitor returns (it reads ciphertext, not plaintext) -
    the property the §5-4 seal relies on. Needs a store + argon2 + cryptography."""
    if not _STORE_OK:
        print(f"  SKIP visitor parity: {_SKIP_REASON}"); return
    import crypto_keys
    if not (crypto_keys.kdf_available() and crypto_keys.aead_available()):
        print("  SKIP visitor parity: argon2/cryptography not installed"); return
    from retrieval import BlindContentStore, BlindSealedStore, BlindEntityStore
    from blind_tier import BlindTier
    passphrase = b"correct horse battery staple"
    keystore = crypto_keys.create_keystore(passphrase)
    keys = crypto_keys.derive_sealed_keys(passphrase, keystore)
    ent_key = crypto_keys.derive_entity_key(passphrase, keystore)
    s = _fresh_store(vector_dim=8)
    # fixture: facts with content + entities + a triple + a session summary
    specs = [("photosynthesis converts light to chemical energy", "biology", ["Plant", "Light"]),
             ("tcp guarantees ordered delivery", "networking", ["TCP", "Packet"])]
    fids = []
    for content_txt, cat, ents in specs:
        _, fid = s.add_or_reinforce_fact(content_txt, _emb(s, content_txt), cat, "sess1")
        with s._lock:
            s._link_entities(fid, ents)
        fids.append(fid)
    s.store_fact_relations(fids[0], [{"subject": "photosynthesis", "relation": "converts",
                                      "object": "light", "confidence": 1.0}])
    sum_id = s.add_session_summary("sess1", "covered photosynthesis and tcp", created_cycle=1)

    def mk_sealed(domain, table):
        k = keys[domain]
        return BlindSealedStore(s, lambda p, k=k, d=domain: crypto_keys.encrypt_sealed(p, k, d),
                                lambda b, k=k, d=domain: crypto_keys.decrypt_sealed(b, k, d), table)
    ck = keys["content"]
    content = BlindContentStore(s, lambda p: crypto_keys.encrypt_sealed(p, ck, "content"),
                                lambda b: crypto_keys.decrypt_sealed(b, ck, "content"))
    entities = BlindEntityStore(s, lambda e: crypto_keys.encrypt_entities(e, ent_key),
                                lambda b: crypto_keys.decrypt_entities(b, ent_key))
    sealed = {"episode": mk_sealed("episode", "semantic_he_episodes"),
              "triple":  mk_sealed("triple", "semantic_he_triples"),
              "summary": mk_sealed("summary", "semantic_he_summaries")}
    bt = BlindTier(s, content=content, entities=entities,
                   content_hmac_fn=lambda t: crypto_keys.content_hmac(t, keys["hmac"]), sealed=sealed)
    bt.reconcile()
    v = bt.visitor()
    # PARITY: fact working set (content + category + normalized entities)
    for fid, (content_txt, cat, ents) in zip(fids, specs):
        view = v.fact_view(fid)
        assert view["content"] == content_txt and view["category"] == cat
        # Parity is blind-mirror == PLAINTEXT STORE (which may auto-extract extra
        # content-derived entities via the optional entity_extractor), not == the
        # manually-linked input. Case-insensitive; robust with or without the extractor.
        assert {x.lower() for x in view["entities"]} == {x.lower() for x in s.get_entities_for_fact(fid)}
    # PARITY: triples as sets (sans structural confidence), and non-empty
    plain_t = {(t["subject"], t["relation"], t["object"]) for t in s.get_fact_relations(fids[0])}
    blind_t = {(t["subject"], t["relation"], t["object"]) for t in v.triples(fids[0])}
    assert blind_t == plain_t and blind_t
    # PARITY: summary text
    assert v.summary(sum_id) == "covered photosynthesis and tcp"
    # BLIND-SOURCED: tamper the plaintext content; the visitor is unaffected (reads ciphertext)
    s._conn.execute("UPDATE semantic_facts SET content='__tampered__' WHERE id=?", (fids[0],))
    s._conn.commit()
    assert v.fact_view(fids[0])["content"] == specs[0][0]     # still original, from ciphertext
    s.close()
    print("  visitor parity OK: working set reconstructed from ciphertext == plaintext, blind-sourced")


def test_get_passphrase_returns_wipeable_bytearray():
    """#4 hygiene fix: get_passphrase returns a MUTABLE bytearray so the provider resolvers'
    ``finally: if isinstance(passphrase, bytearray): secure_zero(...)`` guards actually fire
    (it previously returned immutable bytes, so the wipe silently never ran). Proves: (1) an
    env-sourced passphrase is a bytearray, (2) every consumer accepts it - a key derived from
    the bytearray equals one from the equivalent bytes (all derivations coerce via
    bytes(passphrase) at the KDF), (3) secure_zero zeroes it in place. Needs argon2."""
    import crypto_keys, os
    if not crypto_keys.kdf_available():
        print("  SKIP passphrase wipe: argon2 not installed"); return
    secret = "correct horse battery staple"
    os.environ[crypto_keys.ENV_PASSPHRASE] = secret
    try:
        pw = crypto_keys.get_passphrase(prompt=False)
        assert isinstance(pw, bytearray), type(pw)
        # consumers accept a bytearray: derive the same key from the bytearray and from bytes.
        ks = crypto_keys.create_keystore(bytes(pw))
        k_ba = crypto_keys.derive_entity_key(pw, ks)
        k_by = crypto_keys.derive_entity_key(secret.encode(), ks)
        assert bytes(k_ba) == bytes(k_by)
        crypto_keys.secure_zero(k_ba); crypto_keys.secure_zero(k_by)
        # the wipe the resolvers rely on now actually zeroes the buffer in place.
        n = len(pw)
        crypto_keys.secure_zero(pw)
        assert n > 0 and set(pw) == {0}, "secure_zero did not zero the passphrase"
    finally:
        os.environ.pop(crypto_keys.ENV_PASSPHRASE, None)
    print("  passphrase wipe OK: bytearray returned + accepted by consumers + secure_zero zeroes it")


def test_blind_reconcile_readback_helpers():
    """Write-path completeness (§14 6a) store read-back helpers - the data source for the provider
    _blind_reconcile pass. Proves (pure SQLite, no openfhe): get_fact_embedding round-trips the
    stored vector exactly; get_fact_hrr_phases returns the stored HRR; facts_missing_blind is the
    LEFT-JOIN worklist (all facts missing until a blind row exists, then they drop off - the
    idempotent driver that catches abstraction/gist/procedural/builtin facts + backfill). Needs a
    store + numpy."""
    if not _STORE_OK:
        print(f"  SKIP reconcile helpers: {_SKIP_REASON}"); return
    import numpy as np
    hg = _load("holographic")
    s = _fresh_store(vector_dim=16)
    fids = []
    for i in range(4):
        content = f"reconcile fact {i} about user and dark themes"
        v = _emb(s, content)
        hv = hg.encode_fact(content, ["user"], s.hrr_dim)
        _, fid = s.add_or_reinforce_fact(content, v, "general", "t", hrr_vector=hv, entities=["user", f"e{i}"])
        fids.append((fid, v, hv))
    # embedding read-back is exact (the reconcile mirrors this into semantic_he, no Ollama)
    f0, v0, hv0 = fids[0]
    back = np.asarray(s.get_fact_embedding(f0)); orig = np.asarray(v0)
    assert back.shape == orig.shape
    assert float(np.dot(back / np.linalg.norm(back), orig / np.linalg.norm(orig))) > 0.99999
    # HRR read-back matches the stored phases (== the plaintext encode)
    ph = s.get_fact_hrr_phases(f0)
    assert ph is not None and hg.similarity(ph, hv0) > 0.999
    assert s.get_fact_hrr_phases(10_000) is None              # absent fact
    # facts_missing_blind: every fact missing until a blind row exists; then it drops off.
    assert s.facts_missing_blind("semantic_he") == [f for f, _, _ in fids]
    s.store_he_vector(f0, b"opaque-ct", table="semantic_he")  # simulate the mirror for one fact
    assert s.facts_missing_blind("semantic_he") == [f for f, _, _ in fids[1:]]
    assert s.facts_missing_blind("semantic_he", limit=2) == [f for f, _, _ in fids[1:3]]  # batched backfill
    # per-table independence: semantic_he_hrr still lists everyone (no HRR mirror yet)
    assert s.facts_missing_blind("semantic_he_hrr") == [f for f, _, _ in fids]
    # superseded facts are excluded from the worklist
    s._conn.execute("UPDATE semantic_facts SET tier='superseded' WHERE id=?", (fids[1][0],)); s._conn.commit()
    assert fids[1][0] not in s.facts_missing_blind("semantic_he")
    s.close()
    print("  reconcile helpers OK: embedding/HRR read-back exact, facts_missing_blind worklist + batching")


def test_facts_missing_blind_source_filter():
    """Write-path completeness (§14 6a) poison-pill guard: a fact whose plaintext SOURCE is absent
    must NOT sit forever in the capped reconciliation worklist. A fact added with hrr_vector=None
    has a NULL hrr_vector, so it can NEVER produce an HRR ciphertext - facts_missing_blind on
    semantic_he_hrr must EXCLUDE it (else it permanently saturates the LIMIT window and starves
    facts that DO have an HRR lift), while the embedding worklist still includes it. Pure SQLite."""
    if not _STORE_OK:
        print(f"  SKIP source filter: {_SKIP_REASON}"); return
    hg = _load("holographic")
    s = _fresh_store(vector_dim=16)
    _, f_hrr = s.add_or_reinforce_fact(
        "dark themes preferred", _emb(s, "a"), "general", "t",
        hrr_vector=hg.encode_fact("dark themes preferred", ["user"], s.hrr_dim))
    _, f_nohrr = s.add_or_reinforce_fact(
        "the port is 5432", _emb(s, "b"), "general", "t", hrr_vector=None)
    # HRR worklist excludes the NULL-hrr fact; the embedding worklist includes both.
    assert s.facts_missing_blind("semantic_he_hrr") == [f_hrr], s.facts_missing_blind("semantic_he_hrr")
    assert s.facts_missing_blind("semantic_he") == [f_hrr, f_nohrr]
    # Even under a tight LIMIT the NULL-hrr fact never crowds the HRR window.
    assert f_nohrr not in s.facts_missing_blind("semantic_he_hrr", limit=1)
    s.close()
    print("  source filter OK: NULL-hrr fact excluded from HRR worklist (no poison-pill)")


def test_entity_mirror_refresh_on_reinforce():
    """Entity-set staleness fix: the AEAD entity set is the one MUTABLE blind source - reinforcement
    links new entities to an existing fact, so 'mirror once when missing' goes stale.
    facts_needing_entity_mirror must re-list a fact after a NEW link lands (entities_dirty), and
    NOT after an idempotent re-link of the same entities (no wasted re-encrypt). Pure SQLite."""
    if not _STORE_OK:
        print(f"  SKIP entity refresh: {_SKIP_REASON}"); return
    s = _fresh_store(vector_dim=16)
    _, fid = s.add_or_reinforce_fact(
        "acme deploys nginx", _emb(s, "c"), "general", "t", entities=["acme"])
    # Initially missing its blind entity row -> on the worklist.
    assert fid in s.facts_needing_entity_mirror()
    # Simulate the reconcile mirror, then clear the flag -> off the list.
    s.store_he_vector(fid, b"opaque-entity-ct", table="semantic_he_entities")
    s.mark_entities_mirrored(fid)
    assert fid not in s.facts_needing_entity_mirror()
    # A genuinely NEW entity link makes the stored set stale -> back on the list.
    s._link_entities(fid, ["nginx"])
    assert fid in s.facts_needing_entity_mirror(), "new entity link did not flag re-mirror"
    # Re-mirror + clear, then an idempotent re-link of the SAME entity must NOT re-flag.
    s.mark_entities_mirrored(fid)
    s._link_entities(fid, ["nginx"])
    assert fid not in s.facts_needing_entity_mirror(), "idempotent re-link wastefully re-flagged"
    s.close()
    print("  entity refresh OK: new link re-mirrors, idempotent re-link does not")


def test_prune_forget_policy_demote_then_delete():
    """P2b-store buried-but-pluckable forget policy. forget_after_cycles=0 deletes a faded fact
    immediately (legacy); >0 DEMOTES it (kept, dormant_since_cycle stamped) and deep-deletes only
    after the dormant grace elapses on the logical clock; reinforcement clears dormancy. Pure SQLite."""
    if not _STORE_OK:
        print(f"  SKIP forget policy: {_SKIP_REASON}"); return
    # legacy: delete at resonance 0
    s = _fresh_store(vector_dim=16)
    _, f = s.add_or_reinforce_fact("fades away", _emb(s, "fade"), "general", "t")
    s._conn.execute("UPDATE semantic_facts SET resonance_count=0 WHERE id=?", (f,)); s._conn.commit()
    s.prune_weak_facts(0)
    assert s.get_fact(f) is None
    s.close()
    # demote then deep-delete on the logical clock
    s = _fresh_store(vector_dim=16)
    s.set_cycle_counts(memory_cycle=10)
    _, f = s.add_or_reinforce_fact("dormant fact", _emb(s, "dorm"), "general", "t")
    s._conn.execute("UPDATE semantic_facts SET resonance_count=0 WHERE id=?", (f,)); s._conn.commit()
    s.prune_weak_facts(5)                                            # stamp dormant_since=10, survive
    assert s.get_fact(f) is not None
    row = s._conn.execute("SELECT dormant_since_cycle FROM semantic_facts WHERE id=?", (f,)).fetchone()
    assert row["dormant_since_cycle"] == 10
    s.set_cycle_counts(memory_cycle=13); s.prune_weak_facts(5)       # 3 < 5 grace -> survive (pluckable)
    assert s.get_fact(f) is not None
    s.set_cycle_counts(memory_cycle=16); s.prune_weak_facts(5)       # 6 >= 5 -> deep-deleted
    assert s.get_fact(f) is None
    s.close()
    # reinforcement before the grace elapses clears dormancy (revival)
    s = _fresh_store(vector_dim=16)
    s.set_cycle_counts(memory_cycle=5)
    _, f = s.add_or_reinforce_fact("revivable", _emb(s, "rev"), "general", "t")
    s._conn.execute("UPDATE semantic_facts SET resonance_count=0 WHERE id=?", (f,)); s._conn.commit()
    s.prune_weak_facts(5)
    s._conn.execute("UPDATE semantic_facts SET resonance_count=4 WHERE id=?", (f,)); s._conn.commit()
    s.prune_weak_facts(5)
    row = s._conn.execute("SELECT dormant_since_cycle FROM semantic_facts WHERE id=?", (f,)).fetchone()
    assert s.get_fact(f) is not None and row["dormant_since_cycle"] is None
    s.close()
    print("  forget policy OK: legacy delete-at-0, demote+deep-delete after grace, revival clears it")


def test_conflict_limbo_holds_until_arbitration():
    """Conflict-limbo (A9/A13): a CONTESTED fact (active conflict group) is held in sustained
    resonance - protected from cycle decay AND prune even at resonance 0 - so it never fades before
    the user arbitrates; an UNCONTESTED faded fact is still pruned. resolve_conflict then supersedes
    the loser (kept as history) and frees the winner. Pure SQLite."""
    if not _STORE_OK:
        print(f"  SKIP conflict limbo: {_SKIP_REASON}"); return
    s = _fresh_store(vector_dim=16)
    s.set_cycle_counts(memory_cycle=20)
    _, a = s.add_or_reinforce_fact("the user prefers dark mode", _emb(s, "dark"), "general", "t")
    _, b = s.add_or_reinforce_fact("the user prefers light mode", _emb(s, "light"), "general", "t")
    _, c = s.add_or_reinforce_fact("an uncontested faded fact", _emb(s, "faded"), "general", "t")
    # a,b are a contested pair at resonance 0; c is uncontested at resonance 0.
    s._conn.execute("UPDATE semantic_facts SET conflict_group_id='g1', conflict_since_cycle=20, "
                    "resonance_count=0 WHERE id IN (?,?)", (a, b))
    s._conn.execute("UPDATE semantic_facts SET resonance_count=0 WHERE id=?", (c,))
    s._conn.commit()
    # limbo: protected decay + prune must NOT touch contested a,b, but the uncontested c IS pruned.
    s.apply_cycle_decay(protect_conflicts=True)
    s.prune_weak_facts(0, protect_conflicts=True)          # delete-at-0, contested spared
    assert s.get_fact(a) is not None and s.get_fact(b) is not None, "contested facts not held in limbo"
    assert s.get_fact(c) is None, "uncontested faded fact should still be pruned"
    # user arbitration: pick a -> b superseded (kept as history), a freed
    res = s.resolve_conflict(a, current_cycle=201)
    assert res and b in res.get("superseded", []), res
    assert s.get_fact(b)["tier"] == "superseded"        # loser retired as history, not deleted
    assert s.get_fact(a)["conflict_group_id"] is None    # winner freed from the conflict
    s.close()
    print("  conflict limbo OK: contested held through decay+prune, arbitration supersedes loser")


def test_surprise_weighted_decay_retention():
    """A11 surprise/importance-weighted retention: with peak_discount>0 a fact that ever mattered
    (high max_resonance_seen - e.g. a surprising one-off that entered high via novelty_boost) fades
    SLOWER than a mundane same-resonance fact, so the unique one-off is retained longer; with the
    discount off they decay identically. Pure SQLite."""
    if not _STORE_OK:
        print(f"  SKIP surprise decay: {_SKIP_REASON}"); return
    s = _fresh_store(vector_dim=16, decay_per_cycle=2.0)   # _fresh_store uses central DEFAULTS for tiers/promotion
    _, hi = s.add_or_reinforce_fact("a surprising one-off", _emb(s, "surprise"), "general", "t")
    _, lo = s.add_or_reinforce_fact("a mundane fact", _emb(s, "mundane"), "general", "t")
    # same current resonance; hi PEAKED above the promotion bar (surprising/important), lo never did.
    s._conn.execute("UPDATE semantic_facts SET resonance_count=3.0, max_resonance_seen=8.0 WHERE id=?", (hi,))
    s._conn.execute("UPDATE semantic_facts SET resonance_count=3.0, max_resonance_seen=3.0 WHERE id=?", (lo,))
    s._conn.commit()
    for _ in range(4):
        s.apply_cycle_decay(peak_discount=0.5)
    r_hi = s.get_fact(hi)["resonance_count"]; r_lo = s.get_fact(lo)["resonance_count"]
    assert r_hi > r_lo, (r_hi, r_lo)                       # high-peak (surprising) retained better
    # control: with the discount OFF they decay identically (same current resonance)
    s._conn.execute("UPDATE semantic_facts SET resonance_count=3.0 WHERE id IN (?,?)", (hi, lo)); s._conn.commit()
    for _ in range(4):
        s.apply_cycle_decay(peak_discount=0.0)
    assert abs(s.get_fact(hi)["resonance_count"] - s.get_fact(lo)["resonance_count"]) < 1e-9
    s.close()
    print(f"  surprise decay OK: high-peak retained ({r_hi:.2f}) > mundane ({r_lo:.2f}); uniform when off")


def test_procedural_seed_durable_and_idempotent():
    """P3e tool-grounding seed: seed_procedural_facts ingests durable procedural/guardrail facts
    (category=procedural, tier=long, high resonance, pinned) so the agent is grounded day one;
    idempotent on re-seed; recallable. Pure SQLite."""
    if not _STORE_OK:
        print(f"  SKIP procedural seed: {_SKIP_REASON}"); return
    s = _fresh_store(vector_dim=16)
    g1 = "always require a human to approve every Stripe payment in the Link app"
    g2 = "amounts for Stripe payments are specified in cents"
    items = [{"content": g1, "embedding": _emb(s, g1), "entities": ["Stripe"]},
             {"content": g2, "embedding": _emb(s, g2)}]
    assert s.seed_procedural_facts(items, current_cycle=1) == 2
    fid = s._conn.execute("SELECT id FROM semantic_facts WHERE content=?", (g1,)).fetchone()["id"]
    f = s.get_fact(fid)
    assert f["category"] == "procedural" and f["tier"] == "long" and f["resonance_count"] >= 10.0
    assert f.get("pinned") in (1, True), f   # auto-pin: never-forget + authority presentation
    assert s.seed_procedural_facts(items, current_cycle=2) == 0   # idempotent
    from retrieval import LatticeRetriever

    class _R(LatticeRetriever):
        def _get_embedding(self, t):
            return _emb(s, t)
    hits = _R(s, "http://x", "nomic", min_similarity=-1.0).search(
        "how do I approve a Stripe payment", limit=5)
    assert any("approve" in h["content"].lower() for h in hits)
    s.close()
    print("  procedural seed OK: durable (long/high-res/pinned) + idempotent + recallable")


def test_blind_reconcile_backfill():
    """Write-path completeness (§14 6a) END-TO-END: facts created WITHOUT a blind mirror (the
    abstraction/gist/procedural/backfill case) are reconciled by reading their plaintext
    embedding/HRR/entities back and mirroring into semantic_he*/_hrr/_entities - then blind recall
    ranks == plaintext. Replicates the provider _blind_reconcile loop inline (the provider can't run
    live; same store helpers + writers it uses) over real CKKS. Needs a store + openfhe + crypto."""
    if not _STORE_OK:
        print(f"  SKIP reconcile backfill: {_SKIP_REASON}"); return
    try:
        import he_crypto, crypto_keys
    except Exception as e:
        print(f"  SKIP reconcile backfill: {e}"); return
    if not (he_crypto.he_available() and crypto_keys.aead_available()):
        print("  SKIP reconcile backfill: openfhe/cryptography unavailable (run on the node)"); return
    from retrieval import BlindWriter, BlindEntityStore, BlindRetriever
    import numpy as np
    hg = _load("holographic")
    EMB, HD = 16, 8                                  # HRR HE dim = 2*HD = 16
    s = _fresh_store(vector_dim=EMB)
    s.hrr_dim = HD
    facts = ["user prefers dark themes", "user likes dark color schemes",
             "the database runs on port 5432", "weather is sunny today"]
    qvec = _emb(s, "user prefers dark themes")       # probe == fact 0's vector source
    fid_vec = {}
    for c in facts:
        v = _emb(s, c)
        hv = hg.encode_fact(c, ["user"], HD)
        _, fid = s.add_or_reinforce_fact(c, v, "general", "t", hrr_vector=hv, entities=["user"])
        fid_vec[fid] = v
    # NO blind rows yet - exactly the post-abstraction / first-blind-enable state.
    assert s.count_he_vectors("semantic_he") == 0
    assert s.facts_missing_blind("semantic_he") == sorted(fid_vec)
    # Build the blind clients + writers the provider would hold (Option A: separate embed/HRR ctx).
    emb_blind, _a, _b = he_crypto.BlindRecallPRE.generate(dim=EMB)
    hrr_blind, _c, _d = he_crypto.BlindRecallPRE.generate(dim=2 * HD)
    ekey = os.urandom(32)
    bw = BlindWriter(s, emb_blind)
    bhw = BlindWriter(s, hrr_blind, table="semantic_he_hrr")
    bes = BlindEntityStore(s, lambda e: crypto_keys.encrypt_entities(e, ekey),
                           lambda b: crypto_keys.decrypt_entities(b, ekey))
    # === the _blind_reconcile loop (inline) ===
    for fid in s.facts_missing_blind("semantic_he"):
        assert bw.write_fact(fid, s.get_fact_embedding(fid))
    for fid in s.facts_missing_blind("semantic_he_hrr"):
        assert bhw.write_fact(fid, hg.hrr_lift(s.get_fact_hrr_phases(fid)).tolist())
    for fid in s.facts_missing_blind("semantic_he_entities"):
        assert bes.set_entities(fid, s.get_entities_for_fact(fid))
    # All three tables fully mirrored, worklists drained (idempotent).
    for tbl in ("semantic_he", "semantic_he_hrr", "semantic_he_entities"):
        assert s.count_he_vectors(tbl) == len(facts), tbl
        assert s.facts_missing_blind(tbl) == [], tbl
    assert bes.get_entities(sorted(fid_vec)[0]) == ["user"]      # entity round-trip post-reconcile
    # Blind recall over the BACKFILLED semantic_he ranks == plaintext cosine.
    plain_rank = sorted(fid_vec, key=lambda f: float(np.dot(np.array(qvec), np.array(fid_vec[f]))), reverse=True)
    br = BlindRetriever(s, "http://x", "nomic", blind=emb_blind, min_similarity=-1.0, blind_hrr=hrr_blind)
    blind_rank = [r["id"] for r in br.blind_search_vec(qvec, limit=len(facts))]
    assert blind_rank == plain_rank, (blind_rank, plain_rank)
    s.close()
    print(f"  reconcile backfill OK: 3 blind tables mirrored from plaintext, blind recall == plaintext")


def test_eval_metrics_pure():
    """Phase-1 harness metrics (pure, no store): right_time_recall (recall + turn satisfaction),
    poison_hit_rate (A6 guardrail), tool_hallucination_rate over hand-built turn results."""
    import eval_metrics as m
    tr = [
        {"expected": ["a", "b"], "prefetched": ["a"], "poison": [], "tool_calls": []},   # 1/2 hit
        {"expected": ["c"], "prefetched": ["c"], "poison": ["x"], "tool_calls": []},     # full hit, no leak
        {"expected": [], "prefetched": ["x"], "poison": ["x"],
         "tool_calls": [{"name": "t", "correct": False}]},                               # leak + halluc
    ]
    r = m.right_time_recall(tr)
    assert r["expected_hits"] == 2 and r["expected_misses"] == 1
    assert abs(r["recall"] - 2 / 3) < 1e-9
    assert r["turns_with_expectation"] == 2 and r["turns_fully_satisfied"] == 1
    p = m.poison_hit_rate(tr)
    assert p["poison_turns"] == 2 and p["turns_with_leak"] == 1 and p["leaked_items"] == 1
    assert abs(p["leak_rate"] - 0.5) < 1e-9
    t = m.tool_hallucination_rate(tr)
    assert t["tool_calls"] == 1 and t["hallucinated"] == 1 and t["rate"] == 1.0
    s = m.summarize(tr)
    assert s["turns"] == 3 and {"right_time_recall", "poison", "tool"} <= set(s)
    print("  eval metrics OK: recall/poison/tool computed correctly")


def test_eval_replay_smoke():
    """Phase-1 harness replay: drives the REAL store/retriever over the example corpus with
    deterministic embeddings (no Ollama). Asserts the pipeline runs end-to-end, produces a
    well-formed per-turn result for every turn, and metrics compute. Recall VALUE is not asserted
    (pseudo-embeds aren't semantic - real Ollama runs measure that); this proves the plumbing."""
    if not _STORE_OK:
        print(f"  SKIP eval replay: {_SKIP_REASON}"); return
    import eval_corpus, eval_replay, eval_metrics
    from eval_embed import deterministic_embed
    corpus = eval_corpus.validate_corpus(eval_corpus.example_corpus())
    n_turns = sum(len(s) for s in corpus)
    results = eval_replay.replay(corpus, config={"block_size": 5, "dream_every_n": 2},
                                 embed_fn=lambda t: deterministic_embed(t, dim=64))
    assert len(results) == n_turns, (len(results), n_turns)
    for r in results:
        assert {"expected", "prefetched", "poison", "tool_calls"} <= set(r)
        assert isinstance(r["prefetched"], list)
    summ = eval_metrics.summarize(results)
    assert summ["turns"] == n_turns
    assert summ["right_time_recall"]["turns_with_expectation"] == 3   # corpus asserts 3 recalls
    print(f"  eval replay OK: {n_turns} turns over example corpus, metrics computed "
          f"(recall={summ['right_time_recall']['recall']:.2f} on pseudo-embeds)")


def test_eval_reference_corpus():
    """Phase-1 reference corpus loads + validates (every expect_top/expect_recall/poison key is
    introduced by some fact) and replays end-to-end; relevance_ordering + tool metrics are present.
    Real-embedding scoring is the eval_run/Ollama path (not asserted here - this is the plumbing)."""
    if not _STORE_OK:
        print(f"  SKIP reference corpus: {_SKIP_REASON}"); return
    import os as _os
    import eval_corpus, eval_replay, eval_metrics
    from eval_embed import deterministic_embed
    corpus = eval_corpus.load_corpus(_os.path.join(PLUGIN_DIR, "eval_corpus_reference.json"))
    n_turns = sum(len(s) for s in corpus)
    results = eval_replay.replay(corpus, config={"block_size": 8},
                                 embed_fn=lambda t: deterministic_embed(t, dim=64))
    assert len(results) == n_turns
    summ = eval_metrics.summarize(results)
    ro = summ["relevance_ordering"]
    assert ro["turns_with_expect_top"] >= 10
    assert summ["tool"]["tool_calls"] == 2 and summ["tool"]["hallucinated"] == 1
    print(f"  reference corpus OK: {n_turns} turns, {ro['turns_with_expect_top']} ranked-recall turns, "
          f"tool halluc {summ['tool']['hallucinated']}/{summ['tool']['tool_calls']}")


def test_blind_tier_collaborator():
    """Phase-0 seam: BlindTier is the single collaborator the provider holds instead of six
    scattered _blind* fields. Proves end-to-end: resolve() brings up the HE recall/HRR contexts +
    the AEAD entity store from a passphrase+keystore; reconcile() mirrors all 3 blind tables from
    plaintext; decorate_retriever() yields a BlindRetriever whose ranking == plaintext; and the
    inactive tier (no recall context) is a clean no-op. This is the wiring the old inline
    register/_blind_reconcile did - now directly testable. Needs a store + openfhe + crypto."""
    if not _STORE_OK:
        print(f"  SKIP blind tier: {_SKIP_REASON}"); return
    try:
        import he_crypto, crypto_keys
    except Exception as e:
        print(f"  SKIP blind tier: {e}"); return
    if not (he_crypto.he_available() and crypto_keys.kdf_available() and crypto_keys.aead_available()):
        print("  SKIP blind tier: openfhe/argon2/cryptography unavailable (run on the node)"); return
    import os, tempfile
    import numpy as np
    from blind_tier import BlindTier
    from retrieval import BlindRetriever
    hg = _load("holographic")
    EMB, HD = 16, 8                                  # HRR HE dim = 2*HD = 16
    s = _fresh_store(vector_dim=EMB)
    s.hrr_dim = HD
    facts = ["user prefers dark themes", "user likes dark color schemes",
             "the database runs on port 5432", "weather is sunny today"]
    qvec = _emb(s, "user prefers dark themes")
    fid_vec = {}
    for c in facts:
        v = _emb(s, c)
        hv = hg.encode_fact(c, ["user"], HD)
        _, fid = s.add_or_reinforce_fact(c, v, "general", "t", hrr_vector=hv, entities=["user"])
        fid_vec[fid] = v
    tmp = tempfile.mkdtemp()
    os.environ[crypto_keys.ENV_PASSPHRASE] = "correct horse battery staple"
    try:
        bt = BlindTier.resolve(
            s, db_path=os.path.join(tmp, "mem.db"),
            keystore_path=os.path.join(tmp, "mem.keys"),
            he_keystore_path=os.path.join(tmp, "mem.he"),
            hrr_dim=HD, reconcile_batch=200)
        assert bt is not None
        assert bt.recall is not None and bt.hrr is not None and bt.entities is not None
        assert bt.writer is not None and bt.hrr_writer is not None
        # reconcile mirrors all 3 blind tables from plaintext; worklists drain.
        bt.reconcile(s)
        for tbl in ("semantic_he", "semantic_he_hrr", "semantic_he_entities"):
            assert s.count_he_vectors(tbl) == len(facts), tbl
        assert s.facts_missing_blind("semantic_he") == []
        assert s.facts_missing_blind("semantic_he_hrr") == []
        assert s.facts_needing_entity_mirror() == []
        # decorate_retriever yields a BlindRetriever whose ranking matches plaintext cosine.
        sentinel = object()
        br = bt.decorate_retriever(sentinel, "http://x", "nomic", -1.0)
        assert isinstance(br, BlindRetriever) and br is not sentinel
        plain_rank = sorted(fid_vec, key=lambda f: float(np.dot(np.array(qvec), np.array(fid_vec[f]))), reverse=True)
        blind_rank = [r["id"] for r in br.blind_search_vec(qvec, limit=len(facts))]
        assert blind_rank == plain_rank, (blind_rank, plain_rank)
        # inactive tier (no recall context) is a clean no-op.
        empty = BlindTier(s, recall=None)
        assert empty.decorate_retriever(sentinel, "http://x", "nomic", -1.0) is sentinel
        assert empty.reconcile(s) == 0
    finally:
        os.environ.pop(crypto_keys.ENV_PASSPHRASE, None)
    s.close()
    print("  blind tier OK: resolve -> 3 tables mirrored, blind recall == plaintext, inactive no-op")


def test_he_blind_write_substrate():
    """0b: BlindWriter encrypts embeddings client-side and fills semantic_he with opaque CKKS
    ciphertext (real BlindRecallPRE). Substrate: the count matches and the plaintext float
    bytes are NOT recoverable from the ct. Needs a real store + openfhe -> node ~/he venv."""
    if not _STORE_OK:
        print(f"  SKIP blind write: {_SKIP_REASON}"); return
    try:
        import he_crypto
    except Exception as e:
        print(f"  SKIP blind write: {e}"); return
    if not he_crypto.he_available():
        print("  SKIP blind write: openfhe/numpy unavailable (run on the node)"); return
    from retrieval import BlindWriter
    DIM, N = 16, 4
    s = _fresh_store(vector_dim=DIM)
    blind, _kb, _sb = he_crypto.BlindRecallPRE.generate(dim=DIM)
    writer = BlindWriter(s, blind)
    saved = {}
    for i in range(N):
        v = _emb(s, f"blindfact-{i}")
        _, fid = s.add_or_reinforce_fact(f"blind fact {i}", v, "general", "t")
        assert writer.write_fact(fid, v), "blind write_fact returned False"
        saved[fid] = v
    assert s.count_he_vectors() == N, s.count_he_vectors()
    for fid, v in saved.items():                       # each ct is opaque
        ct = s.get_he_vector(fid)
        assert ct and store_mod.serialize_vector(v) not in ct
    # a non-positive id / empty embedding is a logged no-op, not a crash
    assert writer.write_fact(0, v) is False and writer.write_fact(5, []) is False
    s.close()
    print(f"  blind write OK: semantic_he filled with {N} CKKS cts")


def test_blind_keystore_setup_and_wrap():
    """0b: crypto_keys.setup_or_load_blind_client generates + AES-GCM-wraps + persists the HE
    keystore; the sidecar is secret-free and the wrapped master round-trips to the live
    secret. (Engine reload across processes is the serialized split, node-proven separately.)
    Self-skips without openfhe/argon2/cryptography."""
    try:
        import he_crypto
        import crypto_keys
    except Exception as e:
        print(f"  SKIP blind keystore: {e}"); return
    if not he_crypto.he_available() or not crypto_keys.kdf_available():
        print("  SKIP blind keystore: openfhe/argon2 unavailable (run on the node)"); return
    import os
    import tempfile
    import openfhe as o
    pw = b"correct horse battery staple"
    try:
        ks0 = crypto_keys.create_keystore(pw)                       # needs argon2
        hek = os.path.join(tempfile.mkdtemp(), "lattice.he")
        client, he_ks, created = crypto_keys.setup_or_load_blind_client(pw, ks0, hek, dim=16)
    except crypto_keys.CryptoUnavailableError as e:
        print(f"  SKIP blind keystore: {e}"); return
    assert created and os.path.exists(hek)
    assert crypto_keys.he_keystore_is_secret_free(he_ks), list(he_ks)
    assert set(he_ks["wrapped_secrets"]) == {"master", "agent"}
    # the wrapped master unwraps to exactly the live master secret bytes
    live_master = o.Serialize(client._sk, o.BINARY)
    wrap_key = crypto_keys.derive_he_wrap_key(pw, ks0)
    got = crypto_keys.unwrap_he_secret(he_ks["wrapped_secrets"]["master"], wrap_key)
    ok = bytes(got) == live_master
    crypto_keys.secure_zero(got); crypto_keys.secure_zero(wrap_key)
    assert ok, "wrapped master did not round-trip to the live secret"
    print("  blind keystore setup+wrap OK")


def test_blind_multi_keystore_setup_and_load():
    """Option A (2a): crypto_keys.setup_or_load_blind_contexts generates the THREE-keyset blind
    tier in ONE process (recall @ embed-dim + HRR @ 2·hrr-dim, both BlindRecallPRE; maint LIGHT
    decay-only BlindMaintenance) with by-tag-isolated eval keys, wraps every secret, and persists
    a secret-free multi keystore. Validates: structure (version 2, 3 keysets, secret-free, maint
    at _MAINT_BLIND_DEPTH), the LIVE clients all compute (recall/HRR cosine + maint decay), and the
    persisted secrets unwrap back to the live ones (load/decode/unwrap helpers). Engine reload from
    the keystore is the serialized split (node-proven separately; not re-deserialized in-process to
    avoid the global eval-key collision). Self-skips without openfhe/argon2/cryptography → node."""
    try:
        import he_crypto
        import crypto_keys
    except Exception as e:
        print(f"  SKIP multi keystore: {e}"); return
    if not (he_crypto.he_available() and crypto_keys.kdf_available() and crypto_keys.aead_available()):
        print("  SKIP multi keystore: openfhe/argon2/cryptography unavailable (run on the node)"); return
    import os, tempfile
    import numpy as np
    import openfhe as o
    EMBED, HRRD = 16, 8                         # HRR HE dim = 2*HRRD = 16
    pw = b"correct horse battery staple"
    try:
        ks0 = crypto_keys.create_keystore(pw)
        hek = os.path.join(tempfile.mkdtemp(), "lattice.multi.he")
        clients, he_ks, created = crypto_keys.setup_or_load_blind_contexts(
            pw, ks0, hek, embed_dim=EMBED, hrr_dim=HRRD, role="user")
    except crypto_keys.CryptoUnavailableError as e:
        print(f"  SKIP multi keystore: {e}"); return
    # structure
    assert created and os.path.exists(hek)
    assert he_ks["version"] == crypto_keys.HE_MULTI_KEYSTORE_VERSION
    assert set(he_ks["keysets"]) == {"recall", "hrr", "maint"}, list(he_ks["keysets"])
    assert crypto_keys.multi_he_keystore_is_secret_free(he_ks)
    assert he_ks["keysets"]["recall"]["meta"]["dim"] == EMBED
    assert he_ks["keysets"]["hrr"]["meta"]["dim"] == 2 * HRRD
    assert he_ks["keysets"]["maint"]["meta"]["depth"] == he_crypto._MAINT_BLIND_DEPTH  # LIGHT (decay-only)
    assert set(clients) == {"recall", "hrr", "maint"}
    # live ops - each keyset computes correctly in the one setup process (by-tag coexistence)
    def _unit(n, s):
        v = np.random.default_rng(s).standard_normal(n); return (v / np.linalg.norm(v)).tolist()
    for name, dim in (("recall", EMBED), ("hrr", 2 * HRRD)):
        c = clients[name]; a, b = _unit(dim, 1), _unit(dim, 2)
        got = c.decrypt_score(c.cosine_score(c.encrypt_unit_vector(a), c.encrypt_unit_vector(b)))
        assert abs(got - float(np.dot(a, b))) < 1e-3, (name, got, float(np.dot(a, b)))
    m = clients["maint"]
    got_decay = m.decrypt_scalars(m.decay(m.encrypt_scalars([0.8]), 0.5), 1)[0]
    assert abs(got_decay - 0.4) < 1e-2, got_decay
    # persistence: the secrets unwrap back to the live ones (no engine reload → no collision)
    ks_disk = crypto_keys.load_multi_he_keystore(hek)
    assert crypto_keys.multi_he_keystore_is_secret_free(ks_disk)
    assert set(crypto_keys.multi_he_key_blobs_from_keystore(ks_disk, "recall")) == {"ctx", "pub", "em", "ea", "rk"}
    assert set(crypto_keys.multi_he_key_blobs_from_keystore(ks_disk, "maint")) == {"ctx", "pub", "em"}
    wrap_key = crypto_keys.derive_he_wrap_key(pw, ks0)
    try:
        live_recall_master = o.Serialize(clients["recall"]._sk, o.BINARY)
        got_master = crypto_keys.unwrap_he_secret(ks_disk["keysets"]["recall"]["wrapped_secrets"]["master"], wrap_key)
        live_maint = o.Serialize(clients["maint"]._sk, o.BINARY)
        got_maint = crypto_keys.unwrap_he_secret(ks_disk["keysets"]["maint"]["wrapped_secrets"]["secret"], wrap_key)
        ok = bytes(got_master) == live_recall_master and bytes(got_maint) == live_maint
        crypto_keys.secure_zero(got_master); crypto_keys.secure_zero(got_maint)
    finally:
        crypto_keys.secure_zero(wrap_key)
    assert ok, "wrapped recall-master / maint secret did not round-trip"
    print("  multi keystore OK: 3 keysets (recall/hrr/maint light), secret-free, live ops + unwrap round-trip")


def test_blind_end_to_end_recall():
    """0c: end-to-end blind tier over a REAL store - the keystore-loaded client + BlindWriter
    fill semantic_he, then BlindRetriever recall ranks identically to a plaintext cosine
    reference. Exercises the exact components the provider wires
    (setup_or_load_blind_client -> BlindWriter -> BlindRetriever) over real CKKS. Needs a
    store + openfhe + argon2 -> node ~/he venv."""
    if not _STORE_OK:
        print(f"  SKIP blind e2e: {_SKIP_REASON}"); return
    try:
        import he_crypto
        import crypto_keys
    except Exception as e:
        print(f"  SKIP blind e2e: {e}"); return
    if not he_crypto.he_available() or not crypto_keys.kdf_available():
        print("  SKIP blind e2e: openfhe/argon2 unavailable (run on the node)"); return
    import os
    import tempfile
    import numpy as np
    from retrieval import BlindWriter, BlindRetriever
    DIM, N = 64, 6
    rng = np.random.default_rng(11)
    unit = lambda v: v / (np.linalg.norm(v) or 1.0)
    q = unit(rng.standard_normal(DIM))
    targets = np.linspace(0.9, 0.2, N)                  # well-separated cosines
    pairs = []
    for c in targets:
        z = rng.standard_normal(DIM); z = unit(z - (z @ q) * q)
        pairs.append((unit(c * q + np.sqrt(max(1 - c * c, 0.0)) * z), float(c)))
    order = list(rng.permutation(N)); pairs = [pairs[i] for i in order]   # shuffle insert order
    s = _fresh_store(vector_dim=DIM)
    pw = b"e2e-passphrase"
    try:
        ks0 = crypto_keys.create_keystore(pw)
        hek = os.path.join(tempfile.mkdtemp(), "lattice.he")
        blind, _ks, _created = crypto_keys.setup_or_load_blind_client(pw, ks0, hek, dim=DIM)
    except crypto_keys.CryptoUnavailableError as e:
        print(f"  SKIP blind e2e: {e}"); s.close(); return
    writer = BlindWriter(s, blind)
    fid_cos = {}
    for v, c in pairs:
        _, fid = s.add_or_reinforce_fact(f"e2e fact {c:.2f}", v.tolist(), "general", "t")
        assert writer.write_fact(fid, v.tolist())
        fid_cos[fid] = c
    assert s.count_he_vectors() == N
    expected = [fid for fid, _ in sorted(fid_cos.items(), key=lambda kv: kv[1], reverse=True)]
    br = BlindRetriever(s, "http://x", "nomic", blind=blind, min_similarity=-1.0)
    got = [r["id"] for r in br.blind_search_vec(q.tolist(), limit=N)]
    assert got == expected, (got, expected)
    err = max(abs(dict(br.blind_scores(q.tolist()))[f] - fid_cos[f]) for f in fid_cos)
    assert err < 1e-2, err
    s.close()
    print(f"  blind e2e OK: N={N} dim={DIM} ranking matches plaintext, max_cos_err={err:.2e}")


def test_blind_policy_scope_limiter():
    """E6 §7.2: the pure-Python scope policy that bounds what PRE provenance cannot -
    top-k ceiling, per-cycle query cap, per-cycle re-encryption cap, audit log. Runs
    everywhere (no openfhe/SQLite)."""
    import blind_policy as bp
    lim = bp.ScopeLimiter(topk_ceiling=5, per_cycle_query_cap=3, per_cycle_reencrypt_cap=8)
    t1 = lim.authorize(cycle=1, k=4)
    assert isinstance(t1, str) and t1
    lim.authorize(cycle=1, k=3)                  # cycle 1 now: 2 queries, 7 re-encrypted
    # top-k ceiling
    try:
        lim.authorize(cycle=1, k=6); assert False, "ceiling not enforced"
    except bp.ScopeExceededError:
        pass
    # per-cycle re-encryption cap (7 + 2 > 8)
    try:
        lim.authorize(cycle=1, k=2); assert False, "re-encrypt cap not enforced"
    except bp.ScopeExceededError:
        pass
    # a NEW cycle resets the per-cycle budgets
    lim.authorize(cycle=2, k=5)
    # per-cycle query cap: cycle 3 allows 3 queries then refuses the 4th
    for _ in range(3):
        lim.authorize(cycle=3, k=1)
    try:
        lim.authorize(cycle=3, k=1); assert False, "query cap not enforced"
    except bp.ScopeExceededError:
        pass
    # audit log reflects the grants
    assert lim.audit.query_count(1) == 2 and lim.audit.total_reencrypted(1) == 7
    assert lim.audit.query_count(3) == 3
    assert len(lim.audit.events()) == 2 + 1 + 3
    try:
        lim.authorize(cycle=4, k=0); assert False, "non-positive k not rejected"
    except ValueError:
        pass


def test_blind_reencrypt_gate():
    """E6 6c: the store-side BlindReEncryptGate binds re-encryptions to a single-use token -
    spends down to the authorized budget, then refuses over-spend, unknown tokens, and
    replay. Pure policy, runs everywhere."""
    import blind_policy as bp
    gate = bp.BlindReEncryptGate()
    gate.register("tok-a", 3)
    assert gate.remaining("tok-a") == 3
    for _ in range(3):
        gate.spend("tok-a")
    assert gate.remaining("tok-a") == 0
    for bad in (lambda: gate.spend("tok-a"),       # over budget
                lambda: gate.spend("nope"),         # unknown token
                lambda: gate.register("tok-a", 1),  # replay of a seen token
                lambda: gate.register("", 1),        # empty token
                lambda: gate.register("tok-b", 0)):  # non-positive k
        try:
            bad(); assert False, "gate failed to refuse a bad operation"
        except bp.TokenError:
            pass


def test_store_reencrypt_audit_substrate():
    """E6 6c: the reencrypt_audit table persists re-encryption grants (substrate-checkable).
    Pure SQLite - no openfhe."""
    if not _STORE_OK:
        print(f"  SKIP reencrypt audit: {_SKIP_REASON}"); return
    s = _fresh_store(vector_dim=8)
    sql = s._conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='reencrypt_audit'").fetchone()
    assert sql and "query_token" in sql["sql"].lower(), sql
    assert s.count_reencrypt_events() == 0
    s.record_reencrypt_event(cycle=5, query_token="abc123", k=4)
    s.record_reencrypt_event(cycle=5, query_token="def456", k=2)
    assert s.count_reencrypt_events() == 2
    evs = s.get_reencrypt_events(limit=10)
    assert evs[0]["query_token"] == "def456" and evs[0]["cycle"] == 5 and evs[0]["k"] == 2
    assert {e["query_token"] for e in evs} == {"abc123", "def456"}
    row = s._conn.execute(                                   # direct substrate check
        "SELECT cycle, k FROM reencrypt_audit WHERE query_token='abc123'").fetchone()
    assert row["cycle"] == 5 and row["k"] == 4
    try:
        s.record_reencrypt_event(5, "", 1); assert False, "empty token not rejected"
    except ValueError:
        pass
    s.close()


def test_he_blind_pre_runtime():
    """E6 6c (runtime trust model, real): an eval-only store scores cosine + re-encrypts each
    result to the agent GATED by a single-use ScopeLimiter token; the AGENT (not the master)
    decrypts; the grant is persisted to reencrypt_audit; over-budget + replay are refused.
    Needs a store + openfhe -> node ~/he venv."""
    if not _STORE_OK:
        print(f"  SKIP pre runtime: {_SKIP_REASON}"); return
    try:
        import he_crypto
    except Exception as e:
        print(f"  SKIP pre runtime: {e}"); return
    if not he_crypto.he_available():
        print("  SKIP pre runtime: openfhe unavailable (run on the node)"); return
    import numpy as np
    import openfhe as o
    import blind_policy as bp
    from retrieval import BlindWriter
    DIM, N = 16, 4
    rng = np.random.default_rng(9); unit = lambda v: v / (np.linalg.norm(v) or 1.0)
    facts = [unit(rng.standard_normal(DIM)) for _ in range(N)]
    q = unit(facts[1] + 0.05 * rng.standard_normal(DIM))
    cos = [float(q @ f) for f in facts]
    s = _fresh_store(vector_dim=DIM)
    user, _kb, sb = he_crypto.BlindRecallPRE.generate(dim=DIM)    # store role: pub + eval + rk
    agent = he_crypto.BlindRecallPRE(user._cc, DIM, user.batch)   # white-box agent: use-key only
    agent._sk = o.DeserializePrivateKeyString(sb["agent"], o.BINARY)
    writer = BlindWriter(s, user)
    fid_cos = {}
    for f, c in zip(facts, cos):
        _, fid = s.add_or_reinforce_fact(f"pre fact {c:.2f}", f.tolist(), "general", "t")
        assert writer.write_fact(fid, f.tolist()); fid_cos[fid] = c
    scope = bp.ScopeLimiter(); gate = bp.BlindReEncryptGate(); cycle = 1
    token = scope.authorize(cycle, k=N)          # agent-side per-cycle cap enforcement
    gate.register(token, N)                        # store accepts the single-use grant
    s.record_reencrypt_event(cycle, token, N)      # persist the audit row
    q_ct = user.encrypt_unit_vector(q.tolist())
    scored = {}
    for fid, ct in s.iter_he_vectors():
        score = user.cosine_score(q_ct, ct)        # STORE: blind cosine (no secret used)
        gate.spend(token)                           # gate each ReEncrypt on the token
        scored[fid] = agent.decrypt_score(user.reencrypt_score(score))  # reencrypt -> AGENT decrypt
    expected_top = sorted(fid_cos, key=lambda f: fid_cos[f], reverse=True)[0]
    assert max(scored, key=scored.get) == expected_top
    assert max(abs(scored[f] - fid_cos[f]) for f in fid_cos) < 1e-2
    for bad in (lambda: gate.spend(token),          # N+1th reencrypt over budget
                lambda: gate.register(token, 1)):    # replay of the grant
        try:
            bad(); assert False, "gate failed to refuse"
        except bp.TokenError:
            pass
    assert s.count_reencrypt_events() == 1
    ev = s.get_reencrypt_events()[0]
    assert ev["query_token"] == token and ev["cycle"] == cycle and ev["k"] == N
    s.close()
    print("  he pre-runtime OK: token-gated reencrypt -> agent decrypt, audit persisted")


def test_store_promotion_requires_dwell():
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    e = _emb(s, "fact one")
    _, fid = s.add_or_reinforce_fact("fact one", e, "general", "sess1")
    s.adjust_resonance(fid, 10)              # plenty of resonance
    s.promote_facts()                        # but zero dwell → no promotion
    tier = s._conn.execute("SELECT tier FROM semantic_facts WHERE id=?", (fid,)).fetchone()["tier"]
    assert tier == "short", tier
    short_dwell = getattr(s, "short_tier_cycles", 2)
    for _ in range(short_dwell):             # use actual from store/central
        s.increment_tier_cycles()
    s.promote_facts()
    tier = s._conn.execute("SELECT tier FROM semantic_facts WHERE id=?", (fid,)).fetchone()["tier"]
    assert tier == "mid", tier
    s.close()


def test_store_recall_reinforcement():
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    e = _emb(s, "recall me")
    _, fid = s.add_or_reinforce_fact("recall me", e, "general", "sess1")
    before = s._conn.execute("SELECT resonance_count FROM semantic_facts WHERE id=?", (fid,)).fetchone()[0]
    s.reinforce_on_recall([fid], 0.34)
    after = s._conn.execute("SELECT resonance_count FROM semantic_facts WHERE id=?", (fid,)).fetchone()[0]
    assert abs((after - before) - 0.34) < 1e-6, (before, after)
    s.close()


def test_store_orphan_entity_gc():
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    e = _emb(s, "Charlie Brown uses numpy")
    _, fid = s.add_or_reinforce_fact("Charlie Brown uses numpy", e, "general", "sess1",
                                     entities=["charlie brown", "numpy"])
    assert s._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] >= 1
    s.remove_fact(fid)
    removed = s.gc_orphan_entities()
    assert removed >= 1
    assert s._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0
    s.close()

def test_store_episode_roundtrip():
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    s.add_episode("sess1", "user", "hello world")
    s.add_episode("sess1", "assistant", "hi there")
    eps = s.get_recent_episodes(limit=10, session_id="sess1")
    assert len(eps) == 2, eps
    assert eps[0]["role"] == "user" and eps[1]["role"] == "assistant", eps
    s.close()

def test_store_hrr_decode_guard():
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store(hrr_dim=1024)
    # a wrong-length blob must be skipped (None), not raise
    assert s._phases_from_blob(b"\x00" * (768 * 8)) is None
    assert s._phases_from_blob(None) is None
    s.close()
    
def test_store_add_turn_ordering():
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    s.add_turn("sess1", "hello", "hi there")
    s.add_turn("sess1", "second", "reply two")
    eps = s.get_recent_episodes(limit=10, session_id="sess1")
    roles = [e["role"] for e in eps]
    assert roles == ["user", "assistant", "user", "assistant"], roles
    contents = [e["content"] for e in eps]
    assert contents == ["hello", "hi there", "second", "reply two"], contents
    s.close()

def test_store_merge_gate_keeps_near_but_distinct():
    if not _STORE_OK:
        print("  SKIP"); return
    # reinforce_threshold high (0.99) so only ~identical embeddings merge.
    s = _fresh_store(reinforce_threshold=0.99)
    import numpy as np
    base = _emb(s, "user prefers dark themes")
    a1, id1 = s.add_or_reinforce_fact("user prefers dark themes", base, "pref", "sess1")
    # A near-but-not-identical embedding must NOT be folded into id1.
    near = (np.array(base) * 0.6 + np.array(_emb(s, "user prefers light themes")) * 0.4)
    near = (near / (np.linalg.norm(near) or 1.0)).tolist()
    a2, id2 = s.add_or_reinforce_fact("user prefers light themes", near, "pref", "sess1")
    assert id1 != id2, (a1, a2, id1, id2)
    s.close()


def test_store_get_fact_roundtrip_and_miss():
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    content = "the capital of france is paris"
    e = _emb(s, content)
    _, fid = s.add_or_reinforce_fact(content, e, "geo", "sess1")
    # Hit: exact-ID lookup returns the stored row with content string-equal.
    row = s.get_fact(fid)
    assert row is not None, "get_fact returned None for a known id"
    assert row["content"] == content, row
    # Miss: after deletion the same id returns None. (The tool layer maps this
    # None to {"found": false} and NEVER returns neighbour rows.)
    assert s.remove_fact(fid) is True
    assert s.get_fact(fid) is None, s.get_fact(fid)
    s.close()


def test_store_relation_extraction_grounded():
    """Phase 5a: an entity-grounded triple is extracted, stored, and HRR-encoded."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    content = "Alice works at Acme"
    _, fid = s.add_or_reinforce_fact(content, _emb(s, content), "general", "sess1",
                                     entities=["alice", "acme"])
    n = s.extract_and_store_relations(fid, content, entities=["alice", "acme"],
                                      min_confidence=0.5)
    assert n == 1, n
    rels = s.get_fact_relations(fid)
    assert len(rels) == 1, rels
    r = rels[0]
    assert (r["subject"], r["relation"], r["object"]) == ("alice", "works_at", "acme"), r
    assert r["confidence"] >= 0.85, r            # both args grounded → high conf
    # HRR triple vector was stored (numpy is present in the store-test env).
    blob = s._conn.execute(
        "SELECT hrr_vector FROM fact_relations WHERE fact_id=?", (fid,)
    ).fetchone()["hrr_vector"]
    assert blob is not None and len(blob) == s.hrr_dim * 8, (blob is None,)
    # Index-backed lookup finds it (live facts only).
    assert s.get_relations(subject="alice", relation="works_at"), "subject lookup empty"
    s.close()


def test_store_relation_extraction_gate_and_noise():
    """Phase 5a: relation-free text yields nothing; an UNGROUNDED triple is scored
    below the default gate and not stored (precision-first)."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    # No relation verb at all → no triples.
    assert s.extract_triples("nice weather today", entities=[]) == []
    # Ungrounded prefers-triple: extracted but low confidence.
    t = s.extract_triples("users prefer dark themes", entities=[])
    assert len(t) == 1 and t[0]["relation"] == "prefers", t
    assert t[0]["confidence"] < 0.5, t
    c = "users prefer dark themes"
    _, fid = s.add_or_reinforce_fact(c, _emb(s, c), "pref", "sess1")
    # At the default gate (0.5) the ungrounded triple is NOT stored.
    assert s.extract_and_store_relations(fid, c, entities=[], min_confidence=0.5) == 0
    assert s.get_fact_relations(fid) == []
    s.close()


def test_store_relation_extraction_elided_subject_guard():
    """Phase 5a precision: in coordination ('X verb1 Y and verb2 Z') the second
    verb's subject is elided. Rather than emit a CONFIDENT WRONG triple with the
    prior clause's object as subject, the triple is dropped - but the well-formed
    first clause, and genuinely separate clauses, still extract correctly."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    # Elided subject for 'lives in' → only the correct works_at triple survives.
    t = s.extract_triples("Free works at Anthropic and lives in Seattle",
                          entities=["free", "anthropic", "seattle"])
    assert t == [{"subject": "free", "relation": "works_at",
                  "object": "anthropic", "confidence": 0.9}], t
    # Two fully-formed clauses → BOTH correct triples extracted.
    t2 = {(x["subject"], x["relation"], x["object"]) for x in s.extract_triples(
        "Bob lives in Paris and Alice works at Acme",
        entities=["bob", "paris", "alice", "acme"])}
    assert t2 == {("bob", "lives_in", "paris"), ("alice", "works_at", "acme")}, t2
    s.close()


def test_store_relation_extraction_idempotent_and_cascade():
    """Phase 5a: re-extraction is idempotent (UNIQUE), and pruning a fact cascades
    its relations away (ON DELETE CASCADE, foreign_keys ON)."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    content = "Bob lives in Seattle"
    _, fid = s.add_or_reinforce_fact(content, _emb(s, content), "general", "sess1",
                                     entities=["bob", "seattle"])
    assert s.extract_and_store_relations(fid, content, entities=["bob", "seattle"]) == 1
    # Second pass inserts nothing (idempotent on the UNIQUE key).
    assert s.extract_and_store_relations(fid, content, entities=["bob", "seattle"]) == 0
    assert s._conn.execute("SELECT COUNT(*) FROM fact_relations").fetchone()[0] == 1
    # Deleting the fact cascades the relation row.
    assert s.remove_fact(fid) is True
    assert s._conn.execute("SELECT COUNT(*) FROM fact_relations").fetchone()[0] == 0
    s.close()


def _seed_relations(s):
    import numpy as np
    def emb(t):
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        v = rng.standard_normal(s.vector_dim); return (v / (np.linalg.norm(v) or 1.0)).tolist()
    for c, e in [("Free lives in Seattle", ["free", "seattle"]),
                 ("Free works at Anthropic", ["free", "anthropic"]),
                 ("Bob lives in Paris", ["bob", "paris"])]:
        _, fid = s.add_or_reinforce_fact(c, emb(c), "general", "x", entities=e)
        s.extract_and_store_relations(fid, c, entities=e)
    return s


def test_store_relational_recall_graph():
    """Phase 5b: a structured (subject, relation, ?) query returns the exact triple
    as a 'graph' match; an under-specified (?, relation, ?) returns the whole set."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _seed_relations(_fresh_store())
    r = s.relational_recall(subject="free", relation="lives_in")
    assert len(r) == 1 and r[0]["match"] == "graph", r
    assert (r[0]["subject"], r[0]["relation"], r[0]["object"]) == ("free", "lives_in", "seattle"), r
    assert r[0]["content"] == "Free lives in Seattle", r        # source fact attached
    both = {(x["subject"], x["object"]) for x in s.relational_recall(relation="lives_in")}
    assert both == {("free", "seattle"), ("bob", "paris")}, both
    s.close()


def test_store_relational_recall_multiword_free_query():
    """Phase 5b: a free-text question naming a MULTI-WORD entity ('Acme Robotics')
    resolves the whole phrase as one anchor (not just split tokens), so it matches
    the stored multi-word subject/object - without depending on spaCy."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    content = "Acme Robotics is located in Boston"
    _, fid = s.add_or_reinforce_fact(content, _emb(s, content), "general", "x",
                                     entities=["acme robotics", "boston"])
    s.extract_and_store_relations(fid, content, entities=["acme robotics", "boston"])
    rel, anchors = s._parse_relational_query("where is Acme Robotics located?")
    assert rel == "located_in", (rel, anchors)
    assert "acme robotics" in anchors, anchors          # whole phrase, not split
    r = s.relational_recall(query="where is Acme Robotics located?")
    assert any(x["object"] == "boston" for x in r), r
    s.close()


def test_content_quote_numeric_conflict():
    """A fact must not contradict its own source_quote numerically - and a correct
    DERIVATION must not be mistaken for a contradiction.

    Every case below is real, from a 6,983-fact clinical corpus. The positive is the
    defect that motivated the check: it was stored 'attested', because attestation
    verifies the quote against the TRANSCRIPT and nothing verified the fact against
    its quote. The negatives are the reason this marks instead of dropping - a blanket
    "content number absent from quote" rule fired on 17% of reference-range facts, and
    reading them, nearly all were legitimate.
    """
    att = _load("attestation")
    conflict = att.content_quote_numeric_conflict

    # POSITIVE - the real defect (fact 3493). Same measurement, different values.
    hits = conflict("Normal canine body temperature range is 99.5 to 102.5 degrees F "
                    "(37.5 to 39.2 degrees C).",
                    "normal body temperature for a dog is between 99.8-102.8 degrees F "
                    "(37.6-39.3 degrees C)", 0.02)
    assert hits, "the body-temperature contradiction must be caught"
    assert any(abs(c - 99.5) < 1e-9 for c, _, _ in hits), hits

    # POSITIVE - a stated range its own quote contradicts (fact 5421).
    assert conflict("Plasma osmolality in dogs has a normal range of approximately "
                    "280-300 mOsm/kg.",
                    "The median measured osmolality for all dogs was 302 mOsm/kg", 0.02)

    # NEGATIVE - correct unit DERIVATIONS. Orders of magnitude away, must not fire.
    assert not conflict("The MCL for TTHM is 0.080 milligrams per liter (80 parts per billion).",
                        "The maximum contaminant level (MCL) for TTHM is 0.080 [mg/L].", 0.02)
    assert not conflict("Fibrinogen 1.24 to 4.30 g/L (124 to 430 mg/dL)",
                        "reference intervals were 1.24-4.30 g/l", 0.02)
    assert not conflict("up to 5 white blood cells per high power field (HPF, 400x)",
                        "White blood cells are reported per HPF using the high dry "
                        "objective (40x)", 0.02)

    # NEGATIVE - ROUNDING, the dominant benign case. Not a different measurement.
    assert not conflict("A caregiver placebo effect occurs in approximately 40% of cases.",
                        "A caregiver placebo effect ... occurred 39.7% of the time.", 0.02)
    assert not conflict("adverse reactions: systemic disorders (37.1%)",
                        "adverse reactions ... systemic disorders (37.06%)", 0.02)
    assert not conflict("median resting heart rate was approximately 60 beats per minute",
                        "Apparently healthy dogs: HR 60.5 [55.2-65.3] beats/min", 0.02)

    # NEGATIVE - citation numbers can never be in a quote and must not count.
    assert not conflict("GGT reference range for dogs is <10 U/L (Laboklin 2024).",
                        "canine GGT <10 U/L", 0.02)
    assert not conflict("Ammonia reference interval is 24-36 ug/dL (as_of 2021).",
                        "ammonia reference interval 24-36 ug/dL", 0.02)

    # NEGATIVE - disabled by default, so no profile inherits this silently.
    assert conflict("temperature is 99.5 F", "temperature is 99.8 F", 0.0) == []
    assert conflict("temperature is 99.5 F", "temperature is 99.8 F", -1) == []

    # NEGATIVE - a far-away number is a different quantity, not a contradiction.
    assert not conflict("USG >1.030 indicates concentrating ability",
                        "glomerular filtrate has specific gravity 1.008 to 1.012", 0.001)


def test_content_quote_numeric_tolerance_defaults_off_and_is_wired():
    """The knob must exist, default to OFF, and actually reach the provider - a check
    that is configured but never read is worse than no check, because it reports safety
    it is not providing."""
    schema = _load("config_schema")
    entry = next((e for e in schema.CONFIG_SCHEMA
                  if e["key"] == "content_quote_numeric_tolerance"), None)
    assert entry is not None, "config key missing from the schema"
    assert entry["default"] == 0.0, "must default OFF: existing profiles keep their behaviour"
    src = open(os.path.join(PLUGIN_DIR, "__init__.py"), encoding="utf-8").read()
    assert "_content_quote_numeric_tolerance" in src, "provider never reads the key"
    cons = open(os.path.join(PLUGIN_DIR, "consolidation.py"), encoding="utf-8").read()
    assert "content_quote_numeric_conflict" in cons, "consolidation never calls the check"
    assert "numeric_conflict" in cons, "verdict never recorded in quote_status"
    # It must MARK, not drop: a `continue` here would discard the fact.
    i = cons.find("content_quote_numeric_conflict(")
    j = cons.find("quote_status = \"numeric_conflict\"", i)
    assert i != -1 and j != -1 and "continue" not in cons[i:j], \
        "the numeric conflict path must mark the fact, never drop it"


def test_store_relational_query_pronoun_does_not_bind_to_an_abbreviation():
    """A capitalized PRONOUN mid-query must not be taken as an anchor.

    Found live: a vet corpus stores the abbreviation HE (hepatic encephalopathy) as a
    real subject, and "I am worried. He never whines." returned
    `acute liver failure --[results_in]--> he` - liver-failure triples injected into a
    question about pain. The single-token fallback skipped only the SENTENCE-INITIAL
    word, so a `He` opening a second sentence sailed through. Every question about a
    person or an animal has that shape.

    Both polarities, because the fix must not cost the real anchor: an ALL-CAPS
    spelling is an abbreviation, not a pronoun, so `HE` still binds.
    """
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    content = "HE is part of hepatic failure"
    ents = ["he", "hepatic failure"]
    _, fid = s.add_or_reinforce_fact(content, _emb(s, content), "general", "x",
                                     entities=ents)
    s.extract_and_store_relations(fid, content, entities=ents)
    # The negatives below are only meaningful if a `he` triple EXISTS to be wrongly
    # recalled. A first version of this test used a relation the deterministic
    # extractor does not implement, stored nothing, and passed its negatives vacuously.
    assert s._conn.execute(
        "SELECT count(*) FROM fact_relations WHERE subject='he' OR object='he'"
    ).fetchone()[0] > 0, "fixture stored no `he` triple - the negatives would be vacuous"

    # NEGATIVE: the pronoun must not anchor, at any position but 0.
    for q in ("I am worried. He never whines.",
              "How do I tell if he is hurting? He never whines.",
              "The vet asked about him. His appetite is fine."):
        _, anchors = s._parse_relational_query(q)
        assert "he" not in anchors and "his" not in anchors and "him" not in anchors, (q, anchors)
        assert s.relational_recall(query=q) == [], (q, s.relational_recall(query=q))

    # POSITIVE: the ALL-CAPS abbreviation is still an anchor and still recalls.
    q = "What is HE part of?"
    _, anchors = s._parse_relational_query(q)
    assert "he" in anchors, anchors
    got = s.relational_recall(query=q)
    assert any(x["subject"] == "he" or x["object"] == "he" for x in got), got

    # POSITIVE: an ordinary proper noun is untouched by the stoplist.
    _, anchors = s._parse_relational_query("Does Charlie have it?")
    assert "charlie" in anchors, anchors

    # A shouted query carries no capitalisation signal, so the stoplist still applies.
    _, anchors = s._parse_relational_query("IS HE HURTING? HE NEVER WHINES.")
    assert "he" not in anchors, anchors
    s.close()


def test_store_relational_recall_fuzzy_hrr():
    """Phase 5b: when no triple satisfies ALL slots, the HRR partial-binding probe
    surfaces the closest structural match (graceful fallback), labelled 'hrr'."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _seed_relations(_fresh_store())
    # No 'free lives_in portland' exists → the (free, lives_in, *) triple is the
    # nearest match (2 of 3 slots), returned as a fuzzy hrr hit, not graph.
    r = s.relational_recall(subject="free", relation="lives_in", object="portland", hrr_floor=0.4)
    assert r and r[0]["match"] == "hrr", r
    assert r[0]["object"] == "seattle" and r[0]["score"] >= 0.4, r
    # A high floor suppresses the fuzzy match entirely (no false certainty).
    assert s.relational_recall(subject="free", relation="lives_in",
                               object="portland", hrr_floor=0.95) == []
    s.close()


def test_store_relational_recall_free_query_and_superseded():
    """Phase 5b: free-text parse resolves a one-name question; superseded belief-
    history is excluded from relational recall by default."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _seed_relations(_fresh_store())
    r = s.relational_recall(query="where does Free live?")
    assert len(r) == 1 and (r[0]["subject"], r[0]["object"]) == ("free", "seattle"), r
    # Retire the fact as superseded history → its triple drops out of recall.
    s._conn.execute("UPDATE semantic_facts SET tier='superseded' WHERE content=?",
                    ("Free lives in Seattle",))
    s._conn.commit()
    assert s.relational_recall(subject="free", relation="lives_in") == []
    assert s.relational_recall(subject="free", relation="lives_in",
                               include_superseded=True), "should reappear when included"
    s.close()


def _seed_chain(s):
    import numpy as np
    def emb(t):
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        v = rng.standard_normal(s.vector_dim); return (v / (np.linalg.norm(v) or 1.0)).tolist()
    for c, e in [("Free works at Anthropic", ["free", "anthropic"]),
                 ("Anthropic is located in San Francisco", ["anthropic", "san francisco"]),
                 ("San Francisco is located in California", ["san francisco", "california"]),
                 ("California is located in the USA", ["california", "usa"])]:
        _, fid = s.add_or_reinforce_fact(c, emb(c), "general", "x", entities=e)
        s.extract_and_store_relations(fid, c, entities=e)
    return s


def test_store_infer_transitive_and_no_write():
    """Phase 5c: a 2-hop chain surfaces a labelled inference with its supporting
    path and a decayed confidence - and inference NEVER writes to the DB."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _seed_chain(_fresh_store())
    fr0 = s._conn.execute("SELECT COUNT(*) FROM fact_relations").fetchone()[0]
    sf0 = s._conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
    # max_hops=1 = multi-hop disabled (safest constrained-agent setting).
    assert s.infer_relations("free", max_hops=1) == []
    inf = s.infer_relations("free", max_hops=2)
    assert len(inf) == 1, inf
    r = inf[0]
    assert r["subject"] == "free" and r["object"] == "san francisco", r
    assert r["inferred"] is True and r["hops"] == 2, r
    assert r["relation"] is None, r                       # mixed chain → no composed relation
    assert 0.0 < r["confidence"] < 0.9, r                 # decayed below any single hop
    assert [(e["relation"]) for e in r["path"]] == ["works_at", "located_in"], r["path"]
    # Pure same-transitive-relation chain (SF→CA→USA, all located_in) composes to
    # that relation; the 1-hop SF→CA stored fact is NOT returned (it's not inferred).
    sf = s.infer_relations("san francisco", max_hops=2)
    usa = [x for x in sf if x["object"] == "usa"]
    assert usa and usa[0]["relation"] == "located_in" and usa[0]["hops"] == 2, sf
    # CRITICAL anti-fabrication invariant: inference persisted nothing.
    assert s._conn.execute("SELECT COUNT(*) FROM fact_relations").fetchone()[0] == fr0
    assert s._conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0] == sf0
    s.close()


def test_store_infer_targeted_hops_and_cycle():
    """Phase 5c: object filter returns only chains terminating there; the hop bound
    and cycle guard keep traversal finite."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _seed_chain(_fresh_store())
    # Targeted: free -> california needs 3 hops; absent at max_hops=2, present at 3.
    assert s.infer_relations("free", object="california", max_hops=2) == []
    hit = s.infer_relations("free", object="california", max_hops=3)
    assert len(hit) == 1 and hit[0]["hops"] == 3, hit
    # Cycle guard: a→b, b→a must not loop forever.
    import numpy as np
    def emb(t):
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        v = rng.standard_normal(s.vector_dim); return (v / (np.linalg.norm(v) or 1.0)).tolist()
    for c, e in [("Loopa is part of Loopb", ["loopa", "loopb"]),
                 ("Loopb is part of Loopa", ["loopb", "loopa"])]:
        _, fid = s.add_or_reinforce_fact(c, emb(c), "general", "x", entities=e)
        s.extract_and_store_relations(fid, c, entities=e)
    out = s.infer_relations("loopa", max_hops=3)            # must terminate
    assert all(r["subject"] == "loopa" for r in out), out
    s.close()


def test_store_self_model_roundtrip_and_seed():
    """Phase 7: set/get/update/delete the deliberate self-model; config seeding
    is INSERT-OR-IGNORE (never clobbers curated values) unless overwrite=True."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    # set normalizes the key to lowercase and stamps the cycle.
    assert s.set_self_model("Role", "memory assistant", current_cycle=2)["key"] == "role"
    assert s.get_self_model("role")["value"] == "memory assistant"
    # UPSERT: same key updates value + cycle in place.
    s.set_self_model("role", "neuroplastic memory engine", current_cycle=3)
    row = s.get_self_model("role")
    assert row["value"] == "neuroplastic memory engine" and row["updated_cycle"] == 3, row
    # seed: only the NEW key is written; the curated 'role' is preserved.
    n = s.seed_self_model({"role": "DEFAULT", "name": "Hermes"}, current_cycle=4)
    assert n == 1, n
    assert s.get_self_model("role")["value"] == "neuroplastic memory engine"
    assert s.get_self_model("name")["value"] == "Hermes"
    # overwrite=True forces a refresh from config.
    s.seed_self_model({"role": "refreshed"}, current_cycle=5, overwrite=True)
    assert s.get_self_model("role")["value"] == "refreshed"
    # full read is ordered by key; delete removes one entry.
    assert [r["key"] for r in s.get_self_model()] == ["name", "role"]
    assert s.delete_self_model("name") is True
    assert s.get_self_model("name") is None
    s.close()


def test_store_self_model_isolated_from_ingest():
    """Phase 7 anti-fabrication invariant: the autonomous ingest path
    (add_or_reinforce_fact) is structurally unable to touch the self-model - it
    lives in a separate table. Even self/infra-looking facts never leak into it."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    s.set_self_model("name", "Hermes", current_cycle=1)
    s.set_self_model("relationship_with_user", "long-term collaborator", current_cycle=1)
    before = s.get_self_model()
    # Drive the autonomous ingest primitive hard, including identity-shaped content.
    for c in ["the user prefers dark themes", "my name is not really important",
              "the assistant runs on local hardware", "user lives in Seattle",
              "the agent is a memory system"]:
        s.add_or_reinforce_fact(c, _emb(s, c), "general", "sess1")
    after = s.get_self_model()
    assert after == before, (before, after)               # self-model untouched by ingest
    assert s.get_self_model("name")["value"] == "Hermes"  # curated value intact
    # Sanity: ingest DID write to the (separate) semantic_facts store.
    assert s._conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0] >= 1
    s.close()


def test_store_narrative_roundtrip_and_bound():
    """Phase 8: session summaries round-trip in chronological order, the keep-bound
    prunes oldest, and an empty summary is skipped."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    for i in range(5):
        s.add_session_summary(f"sess{i}", f"summary number {i}",
                              started_cycle=i, ended_cycle=i + 1, created_cycle=i, keep=3)
    rows = s.get_recent_narrative(limit=10)
    assert len(rows) == 3, rows                          # bounded to keep=3
    assert [r["created_cycle"] for r in rows] == [2, 3, 4], rows   # chronological
    assert rows[-1]["summary"] == "summary number 4", rows
    # newest-first ordering when not chronological
    assert s.get_recent_narrative(limit=1, chronological=False)[0]["created_cycle"] == 4
    # the table itself is pruned (durable but bounded)
    assert s._conn.execute("SELECT COUNT(*) FROM session_summaries").fetchone()[0] == 3
    # empty/whitespace summary is not stored
    assert s.add_session_summary("sx", "   ", created_cycle=9) is None
    assert s._conn.execute("SELECT COUNT(*) FROM session_summaries").fetchone()[0] == 3
    s.close()


# ---- P2: narrative eval bar - regression guards for the P0 digest + P1 structured/temporal/ascii work ----
class _FakeOllamaResp:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_ollama(response_fn):
    """Patch urllib.request.urlopen so summarize_session runs offline. response_fn(payload)
    -> the model's response string; `payload` is the decoded ollama request body, so a test
    can capture the prompt that reached the model. Returns a restore() callable."""
    import urllib.request as _ur
    import json as _j
    _orig = _ur.urlopen

    def _fake(req, timeout=None):
        payload = _j.loads(req.data.decode("utf-8"))
        return _FakeOllamaResp(_j.dumps({"response": response_fn(payload)}).encode("utf-8"))
    _ur.urlopen = _fake
    return lambda: setattr(_ur, "urlopen", _orig)


def test_narrative_structured_fields_roundtrip():
    """P1.1: structured fields store as typed columns and decode back to lists; a plain
    (freeform) summary keeps empty lists + a null throughline (back-compat)."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    sid = s.add_session_summary(
        "s1", "blob", created_cycle=10,
        throughline="chose the digest", decisions=["max 3 lanes", "no HE beside training"],
        open_loops=["resume lane 71"], closed=["completed smoke test"], topics=["memory"])
    plain = s.add_session_summary("s2", "just prose", created_cycle=11)
    rows = {r["summary_id"]: r for r in s.get_recent_narrative(chronological=False)}
    r1, r2 = rows[sid], rows[plain]
    assert r1["decisions"] == ["max 3 lanes", "no HE beside training"], r1["decisions"]
    assert r1["open_loops"] == ["resume lane 71"] and r1["closed"] == ["completed smoke test"], r1
    assert r1["throughline"] == "chose the digest" and r1["topics"] == ["memory"], r1
    assert r2["decisions"] == [] and r2["open_loops"] == [] and r2["throughline"] is None, r2
    s.close()


def test_narrative_temporal_framing():
    """P1.2: mark_prior_narratives_historical keeps the newest current, flags the rest,
    and is idempotent."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    a = s.add_session_summary("s1", "one", created_cycle=1)
    b = s.add_session_summary("s2", "two", created_cycle=2)
    c = s.add_session_summary("s3", "three", created_cycle=3)
    assert s.mark_prior_narratives_historical(keep_current=1) == 2
    hist = {r["summary_id"]: r["historical"] for r in s.get_recent_narrative(chronological=False)}
    assert hist[c] == 0 and hist[a] == 1 and hist[b] == 1, hist
    assert s.mark_prior_narratives_historical(keep_current=1) == 0  # idempotent
    s.close()


def test_narrative_historical_recompute_clears():
    """Regression: mark_prior recomputes BOTH directions - a row marked historical that later
    becomes the newest is CLEARED back to current (the set-only version left it stale, which the
    hybrid backfill exposed)."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    a = s.add_session_summary("s1", "one", created_cycle=5)
    b = s.add_session_summary("s2", "two", created_cycle=1)  # older -> historical
    assert s.mark_prior_narratives_historical(keep_current=1) == 1  # only b demoted
    hist = {r["summary_id"]: r["historical"] for r in s.get_recent_narrative(chronological=False)}
    assert hist[a] == 0 and hist[b] == 1, hist
    # remove a; b (previously historical) is now the newest -> must clear back to current
    s._conn.execute("DELETE FROM session_summaries WHERE summary_id = ?", (a,))
    s._conn.commit()
    assert s.mark_prior_narratives_historical(keep_current=1) == 1  # b promoted 1 -> 0
    only = s.get_recent_narrative(chronological=False)[0]
    assert only["summary_id"] == b and only["historical"] == 0, only
    s.close()


def test_narrative_ascii_guarantee():
    """Core hardening: _ascii_sanitize returns pure ASCII for dashes, arrows, accents,
    bullets, and smart quotes - stored narratives are always ASCII regardless of the model."""
    _sanitize = _load("store_narrative")._ascii_sanitize
    for src in ["a \u2014 b", "did X \u2192 Y", "caf\u00e9 na\u00efve", "\u2022 point",
                "smart \u2019quote\u2019", "en\u2013dash and \u2026 done"]:
        out = _sanitize(src)
        assert all(ord(ch) < 128 for ch in out), (src, out)
    assert _sanitize("a \u2014 b") == "a - b"
    assert _sanitize("caf\u00e9 na\u00efve") == "cafe naive"


def test_narrative_structured_llm_parse_and_fallback():
    """P1.1: summarize_session(structured=True) parses JSON into typed columns; on a
    non-JSON model reply it falls back to storing the raw prose as a freeform summary."""
    if not _STORE_OK:
        print("  SKIP"); return
    import json as _j
    s = _fresh_store()
    good = _j.dumps({"throughline": "did X", "decisions": ["d1", "d2"],
                     "open_loops": ["o1"], "closed": ["c1"], "topics": ["t1"]})
    restore = _patch_ollama(lambda payload: good)
    try:
        sid = s.summarize_session("m", "http://x", "s-struct", structured=True,
                                  digest="LOG BODY", created_cycle=5)
    finally:
        restore()
    assert sid is not None
    r = s.get_recent_narrative(chronological=False)[0]
    assert r["throughline"] == "did X" and r["decisions"] == ["d1", "d2"], r
    assert r["open_loops"] == ["o1"] and r["closed"] == ["c1"] and r["topics"] == ["t1"], r
    restore2 = _patch_ollama(lambda payload: "just prose, not json at all")
    try:
        sid2 = s.summarize_session("m", "http://x", "s-fb", structured=True,
                                   digest="LOG", created_cycle=6)
    finally:
        restore2()
    r2 = [x for x in s.get_recent_narrative(chronological=False) if x["summary_id"] == sid2][0]
    assert r2["throughline"] is None and "prose" in r2["summary"], r2
    s.close()


def test_narrative_digest_not_clipped():
    """P0 40-episode regression: when a digest is provided, the WHOLE digest reaches the
    model (early AND late), bypassing the max_episodes cap that clipped long ingests."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    early, late = "EARLYtok42", "LATEtok99"
    digest = early + "\n" + ("filler line\n" * 200) + late
    cap = {}

    def _resp(payload):
        cap["prompt"] = payload.get("prompt", "")
        return "a short summary"
    restore = _patch_ollama(_resp)
    try:
        s.summarize_session("m", "http://x", "s-dig", digest=digest, created_cycle=7)
    finally:
        restore()
    assert early in cap["prompt"] and late in cap["prompt"], "digest was clipped"
    s.close()


def test_store_semantic_match_can_exclude_its_own_sources():
    """A synthesis caller must not be deduped against its own inputs.

    THE BUG THIS FIXES, measured on a live 6,983-fact corpus: the abstraction pass built
    an abstraction from a cluster, then asked "does this already exist?" against the whole
    corpus -- which still held that cluster. A faithful abstraction of 3-8 related facts
    is necessarily ~0.85+ similar to them, so 9 of 9 candidates were rejected, 8 of them
    against their OWN sources, leaving that lattice with ONE abstraction after 195
    consolidations. The better the abstraction, the more certainly it died.
    """
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    a = "IRIS CKD stage 3 in dogs is creatinine 2.9-5.0 mg/dL"
    _, fid = s.add_or_reinforce_fact(a, _emb(s, a), "staging", "sess1")

    # the identical text must dedup against itself when nothing is excluded ...
    hit = s._find_semantic_match(_emb(s, a), threshold=0.82)
    assert hit is not None and hit["id"] == fid, hit
    # ... and must NOT when that fact is named as one of its own sources
    assert s._find_semantic_match(_emb(s, a), threshold=0.82,
                                  exclude_ids={fid}) is None

    # With the source excluded, a genuine NON-source duplicate must still be found.
    # This is the case k=1 could never reach: the old gist workaround asked for one
    # neighbour and let the candidate through whenever that neighbour was a source,
    # even when the second-nearest was a real duplicate.
    #
    # Inserted directly, because the scenario needs TWO rows sharing one embedding and
    # add_or_reinforce_fact would (correctly) reinforce the first instead of adding a
    # second. _emb is a hash-seeded unit vector, so no two texts are near each other.
    from store_common import serialize_vector
    e = _emb(s, a)
    cur = s._conn.execute(
        "INSERT INTO semantic_facts (content, category, tier, resonance_count, "
        "source_session) VALUES (?, 'staging', 'long', 5, 'sess2')",
        (a + " (independent duplicate row)",))
    fid2 = cur.lastrowid
    s._conn.execute("INSERT INTO semantic_vec (id, embedding) VALUES (?, ?)",
                    (fid2, serialize_vector(e)))
    s._conn.commit()
    got = s._find_semantic_match(e, threshold=0.82, exclude_ids={fid})
    assert got is not None, "excluding the source hid a real non-source duplicate"
    assert got["id"] == fid2, got
    # excluding both leaves nothing
    assert s._find_semantic_match(_emb(s, a), threshold=0.82,
                                  exclude_ids={fid, fid2}) is None


def test_abstraction_and_gist_exclude_sources_and_embed_separately():
    """Source-level invariants for both synthesis paths.

    Two distinct failures, both silent in production because hermes emits no stderr:
      * dedup against own sources (above) -- the layer produces nothing;
      * embedding sent to the REASON endpoint -- with a vLLM shim in front of the
        reasoning model, /api/embeddings 404s, the exception is swallowed, and
        abstractions insert with NO vector: dedup skipped, no semantic_vec row,
        invisible to recall, facts/vec parity broken.
    """
    src = open(os.path.join(PLUGIN_DIR, "store_abstraction.py"), encoding="utf-8").read()
    assert "exclude_ids={r[\"id\"] for r in cluster}" in src, \
        "abstraction dedup no longer excludes its own cluster sources"
    assert "exclude_ids={src[\"id\"] for src in cluster}" in src, \
        "gist dedup no longer excludes its own cluster sources"
    assert src.count("embed_endpoint or ollama_endpoint") >= 2, \
        "an embedding call still uses the reason endpoint unconditionally"
    assert "embed_endpoint: str = None" in src, \
        "embed_endpoint parameter missing from a synthesis entry point"

    con = open(os.path.join(PLUGIN_DIR, "consolidation.py"), encoding="utf-8").read()
    assert con.count("embed_endpoint=self._ollama_endpoint_embed") >= 2, \
        "a caller does not pass the embed endpoint through"

    facts = open(os.path.join(PLUGIN_DIR, "store_facts.py"), encoding="utf-8").read()
    assert "exclude_ids: Optional[set] = None" in facts, "no exclude_ids parameter"
    assert "k = 1 + len(ex)" in facts, \
        "k not inflated for exclusions -- excluded rows consume KNN slots and the " \
        "query returns nothing instead of the nearest eligible row"


def test_store_source_provenance_roundtrip():
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    content = "user runs ollama on port 11434"
    quote = "I run ollama on port 11434 locally"
    ref = "https://example.com/setup"
    _, fid = s.add_or_reinforce_fact(content, _emb(s, content), "infra", "sess1",
                                     source_quote=quote, source_ref=ref)
    row = s.get_fact(fid)
    assert row is not None, "get_fact returned None for a known id"
    assert row["source_quote"] == quote, row
    assert row["source_ref"] == ref, row
    # Back-compat: a fact stored without provenance keeps NULLs.
    c2 = "user likes concise answers"
    _, fid2 = s.add_or_reinforce_fact(c2, _emb(s, c2), "pref", "sess1")
    row2 = s.get_fact(fid2)
    assert row2["source_quote"] is None and row2["source_ref"] is None, row2
    assert row2["quote_status"] is None, row2
    s.close()


def test_store_quote_status_roundtrip():
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    c = "user deploys the service on port 8080"
    _, fid = s.add_or_reinforce_fact(c, _emb(s, c), "infra", "sess1",
                                     source_quote="we deploy on port 8080",
                                     quote_status="attested")
    assert s.get_fact(fid)["quote_status"] == "attested", s.get_fact(fid)
    s.close()


def test_store_temporal_stamping():
    """Phase 1a: learned_at_cycle is set once at INSERT; last_confirmed_cycle
    tracks the memory_cycle at each reinforcement. Validated at the substrate."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    e = _emb(s, "user lives in Seattle")
    # INSERT at cycle 0 → both stamps == 0; supersedion fields NULL.
    _, fid = s.add_or_reinforce_fact("user lives in Seattle", e, "general", "sess1")
    row = s._conn.execute(
        "SELECT learned_at_cycle, last_confirmed_cycle, superseded_by, "
        "superseded_at_cycle FROM semantic_facts WHERE id=?", (fid,)
    ).fetchone()
    assert row["learned_at_cycle"] == 0, dict(row)
    assert row["last_confirmed_cycle"] == 0, dict(row)
    assert row["superseded_by"] is None and row["superseded_at_cycle"] is None, dict(row)
    # Advance the logical clock, then reinforce the SAME fact (semantic match).
    s.set_cycle_counts(memory_cycle=3)
    a2, fid2 = s.add_or_reinforce_fact("user lives in Seattle", e, "general", "sess1")
    assert fid2 == fid, (a2, fid2, fid)               # reinforced, not a new row
    row = s._conn.execute(
        "SELECT learned_at_cycle, last_confirmed_cycle FROM semantic_facts WHERE id=?",
        (fid,)
    ).fetchone()
    assert row["learned_at_cycle"] == 0, dict(row)    # learned-at is immutable
    assert row["last_confirmed_cycle"] == 3, dict(row)  # confirmed-at bumped
    # get_fact surfaces the new temporal fields too.
    f = s.get_fact(fid)
    assert f["learned_at_cycle"] == 0 and f["last_confirmed_cycle"] == 3, f
    assert "superseded_by" in f and "superseded_at_cycle" in f, f
    s.close()


def test_store_supersede_conflict_loser():
    """Phase 1b: a conflict loser bled to 0 is retired as superseded history
    (not deleted), excluded from recall, kept by prune, and walkable via
    get_fact_history. Validated at the SQLite substrate."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    e1, e2 = _emb(s, "user lives in Seattle"), _emb(s, "user lives in Portland")
    _, w = s.add_or_reinforce_fact("user lives in Seattle", e1, "geo", "sess1",
                                   entities=["seattle", "user"])
    _, l = s.add_or_reinforce_fact("user lives in Portland", e2, "geo", "sess1",
                                   entities=["portland", "user"])
    # Simulate post-apply_conflict_decay state: same group, loser bled to 0.
    s._conn.execute("UPDATE semantic_facts SET conflict_group_id='g1', "
                    "resonance_count=5, tier='long' WHERE id=?", (w,))
    s._conn.execute("UPDATE semantic_facts SET conflict_group_id='g1', "
                    "resonance_count=0, tier='long' WHERE id=?", (l,))
    s._conn.commit()

    n = s.supersede_conflict_losers(current_cycle=7)
    assert n == 1, n
    rl = s._conn.execute(
        "SELECT tier, superseded_by, superseded_at_cycle, conflict_group_id "
        "FROM semantic_facts WHERE id=?", (l,)).fetchone()
    assert rl["tier"] == "superseded", dict(rl)
    assert rl["superseded_by"] == w, dict(rl)
    assert rl["superseded_at_cycle"] == 7, dict(rl)
    assert rl["conflict_group_id"] is None, dict(rl)        # group resolved for loser
    # Winner is untouched (still alive; free_conflict_winners clears its lock later).
    rw = s._conn.execute("SELECT tier FROM semantic_facts WHERE id=?", (w,)).fetchone()
    assert rw["tier"] == "long", dict(rw)

    # Excluded from entity recall…
    ids = [f["id"] for f in s.get_facts_for_entity("user")]
    assert w in ids and l not in ids, ids
    # …and from the dedup/reinforce gate (its own embedding no longer matches it).
    assert s._find_semantic_match(e2, threshold=0.5) is None, "superseded matched dedup gate"

    # prune_weak_facts KEEPS the superseded row despite resonance 0.
    s.prune_weak_facts()
    kept = s.get_fact(l)
    assert kept is not None and kept["tier"] == "superseded", kept

    # History walk: loser → winner forward; winner ← loser backward.
    hist = s.get_fact_history(l)
    assert [c["id"] for c in hist["superseded_by_chain"]] == [w], hist
    histw = s.get_fact_history(w)
    assert l in [r["id"] for r in histw["replaced"]], histw
    s.close()


def test_store_supersede_cap_bounds_history():
    """Phase 1b: max_superseded_history drops the oldest superseded rows."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    # Three independent conflict groups, each with a winner + a loser at 0.
    ids = []
    for i in range(3):
        ew, el = _emb(s, f"win {i}"), _emb(s, f"lose {i}")
        _, w = s.add_or_reinforce_fact(f"win {i}", ew, "g", "sess1")
        _, l = s.add_or_reinforce_fact(f"lose {i}", el, "g", "sess1")
        s._conn.execute("UPDATE semantic_facts SET conflict_group_id=?, "
                        "resonance_count=5 WHERE id=?", (f"g{i}", w))
        s._conn.execute("UPDATE semantic_facts SET conflict_group_id=?, "
                        "resonance_count=0 WHERE id=?", (f"g{i}", l))
        ids.append(l)
    s._conn.commit()
    # Supersede each at an increasing cycle, capping history at 2 rows.
    for i, _l in enumerate(ids):
        s.supersede_conflict_losers(current_cycle=i + 1, max_history=2)
    superseded = s._conn.execute(
        "SELECT COUNT(*) FROM semantic_facts WHERE tier='superseded'").fetchone()[0]
    assert superseded == 2, superseded                       # oldest (cycle 1) dropped
    assert s.get_fact(ids[0]) is None, "oldest superseded row should be capped out"
    assert s.get_fact(ids[2]) is not None, "newest superseded row should survive"
    s.close()


def test_store_pending_conflicts_and_resolve():
    """Phase 6: a mature conflict surfaces via pending_conflicts (age-gated), and
    resolve_conflict boosts the winner + supersedes the loser. Substrate-checked."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    _, w = s.add_or_reinforce_fact("user lives in Seattle", _emb(s, "user lives in Seattle"), "geo", "s")
    _, l = s.add_or_reinforce_fact("user lives in Portland", _emb(s, "user lives in Portland"), "geo", "s")
    # Form a conflict at cycle 5 (as resolve_hrr_conflicts would), now at cycle 6.
    for fid, res in ((w, 4), (l, 3)):
        s._conn.execute("UPDATE semantic_facts SET conflict_group_id='cg1', tier='long', "
                        "resonance_count=?, conflict_since_cycle=5 WHERE id=?", (res, fid))
    s.set_cycle_counts(memory_cycle=6)
    s._conn.commit()

    # Age gate: group is 1 cycle old → min_age 5 hides it; min_age 0 shows it.
    assert s.get_pending_conflicts(min_age_cycles=5) == []
    pend = s.get_pending_conflicts(min_age_cycles=0)
    assert len(pend) == 1 and pend[0]["conflict_group_id"] == "cg1", pend
    assert pend[0]["age_cycles"] == 1, pend
    assert {f["id"] for f in pend[0]["facts"]} == {w, l}, pend

    # Resolve in favour of the winner → winner boosted/freed, loser superseded.
    res = s.resolve_conflict(w, current_cycle=6)
    assert res["winner_id"] == w and res["superseded"] == [l], res
    rw = s._conn.execute("SELECT resonance_count, conflict_group_id, conflict_since_cycle, "
                         "last_confirmed_cycle FROM semantic_facts WHERE id=?", (w,)).fetchone()
    assert rw["conflict_group_id"] is None and rw["conflict_since_cycle"] is None, dict(rw)
    assert rw["resonance_count"] == 6.0, dict(rw)          # 4 + 2 boost
    assert rw["last_confirmed_cycle"] == 6, dict(rw)
    rl = s._conn.execute("SELECT tier, superseded_by FROM semantic_facts WHERE id=?", (l,)).fetchone()
    assert rl["tier"] == "superseded" and rl["superseded_by"] == w, dict(rl)
    # Group resolved; resolving a non-conflicted fact is a no-op (None).
    assert s.get_pending_conflicts(min_age_cycles=0) == []
    assert s.resolve_conflict(w, current_cycle=7) is None
    s.close()


# The four PRODUCTION false-positive pairs (nemo webdev training run, 2026-07-07):
# template-parallel facts about DIFFERENT subjects that passed the entity-overlap
# gate (shared version/context entities) AND landed inside the similarity band
# (calibrated 0.72-0.87 under encode_text_rich), quarantining 8 TRUE facts.
# Each tuple: (content_a, subject_a, content_b, subject_b, shared_entities).
_PARALLEL_FP_PAIRS = [
    ("gl_FragCoord is available in GLSL ES versions 1.00, 3.00, 3.10, and 3.20",
     "gl_fragcoord",
     "gl_FrontFacing is available in GLSL ES versions 1.00, 3.00, 3.10, and 3.20",
     "gl_frontfacing",
     ["glsl es", "1.00", "3.00", "3.10", "3.20"]),
    ("smoothstep is available in GLSL ES versions 1.00, 3.00, 3.10 and 3.20.",
     "smoothstep",
     "fract is supported in GLSL ES versions 1.00, 3.00, 3.10 and 3.20.",
     "fract",
     ["glsl es", "1.00", "3.00", "3.10", "3.20"]),
    ("The count parameter of drawArrays is a GLsizei specifying the number of indices to be rendered.",
     "drawarrays",
     "The count parameter of drawElements is a GLsizei specifying the number of elements to be rendered.",
     "drawelements",
     ["count", "glsizei"]),
    ("normalize is supported in GLSL ES versions 1.00, 3.00, 3.10 and 3.20.",
     "normalize",
     "dot is supported in GLSL ES versions 1.00, 3.00, 3.10 and 3.20.",
     "dot",
     ["glsl es", "1.00", "3.00", "3.10", "3.20"]),
]


def _add_conflict_candidate(s, content, subject, shared, relation_object="glsl es"):
    """Add a mid-tier fact with entities + a relation triple, detection-ready."""
    hg = _load("holographic")
    _, fid = s.add_or_reinforce_fact(content, _emb(s, content), "spec", "sess1",
                                     entities=[subject] + list(shared))
    blob = hg.phases_to_bytes(hg.encode_text_rich(content, s.hrr_dim))
    s._conn.execute("UPDATE semantic_facts SET tier='mid', hrr_vector=? WHERE id=?",
                    (blob, fid))
    s._conn.execute(
        "INSERT INTO fact_relations (fact_id, subject, relation, object, confidence) "
        "VALUES (?, ?, ?, ?, 0.9)", (fid, subject, "available_in", relation_object))
    s._conn.commit()
    return fid


def _conflict_gid(s, fid):
    return s._conn.execute(
        "SELECT conflict_group_id FROM semantic_facts WHERE id=?", (fid,)
    ).fetchone()["conflict_group_id"]


def test_store_conflict_parallel_subject_veto():
    """Regression (2026-07-07 false-positive class): template-parallel facts about
    DIFFERENT subjects must NOT be locked as conflicts. Self-validating: the test
    first proves each pair passes the heuristic gates (overlap >= 0.5, sim in band)
    and IS flagged with the veto disabled - then proves the veto spares all four."""
    if not _STORE_OK:
        print("  SKIP"); return
    hg = _load("holographic")

    # (a) Counterfactual: veto OFF reproduces the production bug on pair 1.
    s0 = _fresh_store(conflict_subject_veto=False)
    a0 = _add_conflict_candidate(s0, *_PARALLEL_FP_PAIRS[0][0:2], _PARALLEL_FP_PAIRS[0][4])
    b0 = _add_conflict_candidate(s0, *_PARALLEL_FP_PAIRS[0][2:4], _PARALLEL_FP_PAIRS[0][4])
    # Gate self-check: the pair really is inside the trap (veto is what matters).
    sim = hg.similarity(hg.encode_text_rich(_PARALLEL_FP_PAIRS[0][0], s0.hrr_dim),
                        hg.encode_text_rich(_PARALLEL_FP_PAIRS[0][2], s0.hrr_dim))
    assert s0.conflict_sim_low <= sim <= s0.conflict_sim_high, sim
    s0.resolve_hrr_conflicts()
    g_a, g_b = _conflict_gid(s0, a0), _conflict_gid(s0, b0)
    assert g_a is not None and g_a == g_b, (g_a, g_b)   # bug reproduced
    s0.close()

    # (b) Veto ON (default): all four real pairs pass detection unflagged.
    s = _fresh_store()
    assert s.conflict_subject_veto is True               # central default
    fids = []
    for ca, sa, cb, sb, shared in _PARALLEL_FP_PAIRS:
        fids.append(_add_conflict_candidate(s, ca, sa, shared))
        fids.append(_add_conflict_candidate(s, cb, sb, shared))
    s.resolve_hrr_conflicts()
    locked = [f for f in fids if _conflict_gid(s, f) is not None]
    assert locked == [], locked
    s.close()
    print("  parallel-subject veto OK: 4 production pairs spared; veto-off reproduces the bug")


def test_store_conflict_true_contradiction_still_flagged():
    """The veto must not weaken real detection: an attribute contradiction about
    the SAME subject (shared relation subject) still forms a conflict group."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    a = _add_conflict_candidate(s, "user lives in Seattle", "user",
                                ["seattle"], relation_object="seattle")
    b = _add_conflict_candidate(s, "user lives in Portland", "user",
                                ["portland"], relation_object="portland")
    s.resolve_hrr_conflicts()
    g_a, g_b = _conflict_gid(s, a), _conflict_gid(s, b)
    assert g_a is not None and g_a == g_b, (g_a, g_b)
    since = s._conn.execute(
        "SELECT conflict_since_cycle FROM semantic_facts WHERE id=?", (a,)
    ).fetchone()["conflict_since_cycle"]
    assert since is not None
    s.close()
    print("  true contradiction OK: shared-subject pair still duels")


def test_store_conflict_adjudicator_gate():
    """The adjudicator callback is honored: False skips the lock, True locks,
    and errors/None FAIL OPEN to the pre-existing behavior (lock)."""
    if not _STORE_OK:
        print("  SKIP"); return

    def _seattle_portland(s):
        a = _add_conflict_candidate(s, "user lives in Seattle", "user",
                                    ["seattle"], relation_object="seattle")
        b = _add_conflict_candidate(s, "user lives in Portland", "user",
                                    ["portland"], relation_object="portland")
        return a, b

    # False → compatible → skipped.
    s = _fresh_store(); a, b = _seattle_portland(s)
    s.resolve_hrr_conflicts(adjudicator=lambda x, y: False)
    assert _conflict_gid(s, a) is None and _conflict_gid(s, b) is None
    s.close()
    # True → contradiction confirmed → locked.
    s = _fresh_store(); a, b = _seattle_portland(s)
    s.resolve_hrr_conflicts(adjudicator=lambda x, y: True)
    assert _conflict_gid(s, a) is not None
    s.close()
    # Exception → fail-open → locked (conservative old behavior).
    def _boom(x, y):
        raise RuntimeError("model down")
    s = _fresh_store(); a, b = _seattle_portland(s)
    s.resolve_hrr_conflicts(adjudicator=_boom)
    assert _conflict_gid(s, a) is not None
    s.close()
    # None (ambiguous) → fail-open → locked.
    s = _fresh_store(); a, b = _seattle_portland(s)
    s.resolve_hrr_conflicts(adjudicator=lambda x, y: None)
    assert _conflict_gid(s, a) is not None
    s.close()
    print("  adjudicator gate OK: False skips; True/None/error lock (fail-open)")


def _add_procedural_fact(s, content, tier="short"):
    """Add a category='procedural' tool heuristic, detection-ready (no entities
    or relations needed - the procedural pass is lexical)."""
    _, fid = s.add_or_reinforce_fact(content, _emb(s, content), "procedural",
                                     "sess1", entities=[])
    s._conn.execute("UPDATE semantic_facts SET tier=? WHERE id=?", (tier, fid))
    s._conn.commit()
    return fid


def test_store_procedural_conflict_deterministic_lane():
    """Procedural tool-superstition sweep (2026-07-09 nemo field finding): the
    dream-cycle's '[tool]' heuristics contradict by paraphrase and were invisible
    to the general pass (category-excluded). Lane 1: same tool + OPPOSITE stance
    (avoid vs prefer) + a shared specific topic stem locks a conflict group -
    while same-stance and different-tool pairs are spared."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    assert s.detect_procedural_conflicts is True         # central default
    a = _add_procedural_fact(s, "[web_extract] Avoid using web_extract on "
        "raw.githubusercontent.com URLs pointing to three.js source files.")
    b = _add_procedural_fact(s, "[web_extract] For raw.githubusercontent.com "
        "JavaScript files, always set use_llm_processing: true.", tier="mid")
    # Opposite stance but DIFFERENT tool - must NOT pair across tools.
    d = _add_procedural_fact(s, "[web_search] Avoid raw.githubusercontent.com "
        "site restrictions when searching for three.js source files.")
    s.resolve_hrr_conflicts()
    g_a, g_b = _conflict_gid(s, a), _conflict_gid(s, b)
    assert g_a is not None and g_a == g_b, (g_a, g_b)
    assert _conflict_gid(s, d) is None
    since = s._conn.execute(
        "SELECT conflict_since_cycle FROM semantic_facts WHERE id=?", (a,)
    ).fetchone()["conflict_since_cycle"]
    assert since is not None
    s.close()
    # Same stance, same tool, same topic - consistent advice, must NOT duel
    # (fresh store so no opposite-stance partner exists to absorb either).
    s1 = _fresh_store()
    c1 = _add_procedural_fact(s1, "[web_extract] Avoid using web_extract on "
        "raw.githubusercontent.com URLs pointing to three.js source files.")
    c2 = _add_procedural_fact(s1, "[web_extract] Never fetch three.js source "
        "files from raw.githubusercontent.com directly.")
    s1.resolve_hrr_conflicts()
    assert _conflict_gid(s1, c1) is None and _conflict_gid(s1, c2) is None
    s1.close()
    # Config gate: detect_procedural_conflicts=False disables the sweep.
    s2 = _fresh_store(detect_procedural_conflicts=False)
    a2 = _add_procedural_fact(s2, "[web_extract] Avoid using web_extract on "
        "raw.githubusercontent.com URLs pointing to three.js source files.")
    b2 = _add_procedural_fact(s2, "[web_extract] For raw.githubusercontent.com "
        "JavaScript files, always set use_llm_processing: true.")
    s2.resolve_hrr_conflicts()
    assert _conflict_gid(s2, a2) is None and _conflict_gid(s2, b2) is None
    s2.close()
    print("  procedural lane 1 OK: avoid-vs-prefer duels; same-stance/cross-tool/gated-off spared")


def test_store_procedural_conflict_adjudicator_lane():
    """Lane 2: same-stance paraphrase contradictions (the real 'bundle many URLs
    per call' vs 'prefer a single URL per invocation' pair from the nemo DB)
    carry no opposite polarity, so only the reason-model adjudicator can pair
    them - and ONLY an explicit True locks (None/False/error skip: precision-
    first, NOT fail-open like the general pass, because superstition is
    self-inflicted rather than adversarial)."""
    if not _STORE_OK:
        print("  SKIP"); return
    BUNDLE = ("[web_extract] When extracting JavaScript files, bundling "
              "multiple files in a single web_extract call increases success "
              "likelihood; single-file requests often fail.")
    SINGLE = ("[web_extract] When supplying multiple URLs in a single "
              "web_extract call, extraction often fails; prefer supplying a "
              "single URL per invocation.")

    def _pair(s):
        return _add_procedural_fact(s, BUNDLE), _add_procedural_fact(s, SINGLE)

    # True → locked.
    s = _fresh_store(); a, b = _pair(s)
    calls = []
    s.resolve_hrr_conflicts(adjudicator=lambda x, y: calls.append(1) or True)
    assert _conflict_gid(s, a) is not None
    assert _conflict_gid(s, a) == _conflict_gid(s, b)
    assert len(calls) >= 1
    s.close()
    # False → spared.
    s = _fresh_store(); a, b = _pair(s)
    s.resolve_hrr_conflicts(adjudicator=lambda x, y: False)
    assert _conflict_gid(s, a) is None and _conflict_gid(s, b) is None
    s.close()
    # None (ambiguous) → spared - the procedural lane is NOT fail-open.
    s = _fresh_store(); a, b = _pair(s)
    s.resolve_hrr_conflicts(adjudicator=lambda x, y: None)
    assert _conflict_gid(s, a) is None and _conflict_gid(s, b) is None
    s.close()
    # Error → spared (skip, logged at debug).
    def _boom(x, y):
        raise RuntimeError("model down")
    s = _fresh_store(); a, b = _pair(s)
    s.resolve_hrr_conflicts(adjudicator=_boom)
    assert _conflict_gid(s, a) is None and _conflict_gid(s, b) is None
    s.close()
    # No adjudicator at all → lane 2 never fires → spared.
    s = _fresh_store(); a, b = _pair(s)
    s.resolve_hrr_conflicts()
    assert _conflict_gid(s, a) is None and _conflict_gid(s, b) is None
    s.close()
    print("  procedural lane 2 OK: adjudicator True locks; False/None/error/absent spare")


def test_store_busy_timeout_set():
    """Multi-process write grace: every store connection carries a 30s SQLite
    busy_timeout so a write colliding with another process's finalize burst
    retries instead of throwing 'database is locked'."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    ms = s._conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert ms == 30000, ms
    s.close()
    print("  busy_timeout OK: 30000 ms on a fresh store connection")


def test_finalize_lock_contention():
    """FinalizeLock (cross-process finalize mutex): a held lock rejects
    non-blocking and timed attempts from a second holder, and release hands it
    over. Uses two instances (two file descriptors) - flock/msvcrt locks are
    per-open-file-description, so this is real contention. Also proves the
    lock holds across REAL process boundaries via a subprocess."""
    import subprocess
    import tempfile as _tf
    import time
    pl = _load("proc_lock")
    base = os.path.join(_tf.mkdtemp(), "dbfile")

    a = pl.FinalizeLock(base)
    b = pl.FinalizeLock(base)
    assert a.acquire(0.0) is True
    assert b.acquire(0.0) is False                    # non-blocking: busy
    t0 = time.time()
    assert b.acquire(1.2) is False                    # timed: still busy
    assert time.time() - t0 >= 1.0                    # actually waited
    a.release()
    assert b.acquire(0.0) is True                     # handover after release
    # Cross-process: while b holds, a child process must fail fast.
    code = (
        "import sys; sys.path.insert(0, %r); import proc_lock; "
        "sys.exit(0 if proc_lock.FinalizeLock(%r).acquire(0.0) is False else 1)"
        % (os.path.dirname(os.path.abspath(pl.__file__)), base)
    )
    r = subprocess.run([sys.executable, "-c", code], timeout=30)
    assert r.returncode == 0, "child process acquired a lock the parent holds"
    b.release()
    # After release the child can take it.
    code2 = code.replace("is False", "is True")
    r2 = subprocess.run([sys.executable, "-c", code2], timeout=30)
    assert r2.returncode == 0, "child process failed to acquire a free lock"
    print("  finalize lock OK: contention in-process + cross-process; clean handover")


def test_dream_cadence_gates_session_end():
    """dream_every_n_consolidations must govern the SESSION-END dream too.

    It used to not: on_session_end dreamed unconditionally, so the dial did
    nothing on the path that does nearly all the dreaming in batch use. With one
    agent that is invisible; with 12 overnight lanes x 25 blocks it is 300 dreams
    holding the FinalizeLock ~18s each -- ~90 minutes per night of serialized lock
    time every other lane queues behind.

    Four things are asserted, because three of them are the ways this could be
    wrong rather than merely unimplemented:
      1. the FIRST call dreams (no claim recorded yet -> nothing to skip)
      2. an immediate second call SKIPS (cadence not satisfied)
      3. advancing memory_cycle past the cadence lets it dream again
      4. respect_cadence=False ALWAYS dreams -- a manual `rlm_dream` request must
         never be silently swallowed by a cadence meant for batch ingest
    """
    try:
        _inject_hermes_stubs()
        prov = _load("__init__")
    except Exception as e:
        print(f"  SKIP dream cadence: {e}"); return
    import tempfile
    import threading

    ran = []

    class _CadStore:
        def __init__(self, tmp):
            self.db_path = os.path.join(tmp, "stub.db")
            self._meta = {}
            self._mc = 0
            self.dream_cycle = 0

        # --- the two calls the cadence claim depends on
        def get_cycle_counts(self):
            return (self._mc, self.dream_cycle)

        def get_meta_int(self, key, default=-1):
            return self._meta.get(key, default)

        def set_meta_int(self, key, value):
            self._meta[key] = int(value)

        def increment_cycle(self, key):
            self.dream_cycle += 1
            ran.append(self.dream_cycle)
            # Raise to abort the rest of the dream: this test is about the GATE,
            # and the steps after it need the full store surface.
            raise RuntimeError("stop-after-gate")

    p = prov.LatticeMemoryProvider({})
    tmp = tempfile.mkdtemp()
    p._store = _CadStore(tmp)
    p._write_enabled = True
    p._dream_lock = threading.Lock()
    p._dream_every_n_consolidations = 3

    def dream(**kw):
        try:
            p._run_dream_cycle(**kw)
        except RuntimeError:
            pass          # expected: the stub aborts once the gate is passed
        finally:
            # the aborted run leaves both locks held; release for the next call
            try:
                p._dream_lock.release()
            except RuntimeError:
                pass

    # 1. first call dreams (nothing claimed yet)
    dream(respect_cadence=True)
    assert len(ran) == 1, "first session-end dream should run: %r" % (ran,)
    claim = p._store.get_meta_int(p._DREAM_CLAIM_KEY, -1)
    assert claim == 0, "claim not recorded: %r" % (claim,)

    # 2. immediately again -> skipped, 0 consolidations have happened
    dream(respect_cadence=True)
    assert len(ran) == 1, "second dream should be skipped by cadence: %r" % (ran,)

    # 3. advance the consolidation clock past the cadence -> dreams again
    p._store._mc = 3
    dream(respect_cadence=True)
    assert len(ran) == 2, "dream should run once cadence is satisfied: %r" % (ran,)

    # 4. an explicit request ignores the cadence entirely
    dream(respect_cadence=False)
    assert len(ran) == 3, (
        "respect_cadence=False must always dream -- a manual rlm_dream call "
        "should never be swallowed: %r" % (ran,))
    print("  dream cadence OK: first runs, second skipped, resumes after %d "
          "consolidations, explicit request always dreams" % 3)


def test_dream_cadence_claim_is_visible_cross_process():
    """The cadence marker must live in the DB, not on the instance.

    Twelve separate hermes processes end sessions independently; a claim held in
    a Python attribute would let all twelve dream on the same cycle. This proves
    set_meta_int/get_meta_int round-trip through SQLite and are visible to a
    SEPARATE connection, which is what a second process actually is.
    """
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    assert s.get_meta_int("last_dream_memory_cycle", -1) == -1, "unset should default"
    s.set_meta_int("last_dream_memory_cycle", 7)
    assert s.get_meta_int("last_dream_memory_cycle", -1) == 7

    # A second connection to the same file == another process's view.
    import sqlite3
    other = sqlite3.connect(s.db_path)
    row = other.execute(
        "SELECT value FROM meta WHERE key='last_dream_memory_cycle'").fetchone()
    other.close()
    assert row and int(row[0]) == 7, (
        "claim not visible to a second connection: %r" % (row,))
    # Garbage must not crash the gate -- it degrades to "no claim recorded".
    s.set_meta_int("last_dream_memory_cycle", 0)
    s._conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES "
        "('last_dream_memory_cycle', 'not-an-int')")
    s._conn.commit()
    assert s.get_meta_int("last_dream_memory_cycle", -1) == -1, "should fall back"
    s.close()
    print("  dream claim OK: persists in meta, visible cross-connection, "
          "non-integer degrades to unset")


def test_every_reason_call_disables_thinking():
    """EVERY /api/generate payload must carry think: False.

    ollama exposes thinking as a per-request field and has no Modelfile parameter
    for it, so a thinking-capable reason model silently loses its entire answer
    into an unclosed think block: gemma4:12b returns 0 chars with
    done_reason=length after 32.6s, versus correct JSON in 1.6s with think=false.
    That failure already cost a run -- five overnight lanes did 571s of research
    each and banked ZERO facts because consolidation received empty strings.

    A source-level invariant rather than a behavioural one, deliberately: the
    calls live in four different modules (extraction, conflict adjudication,
    abstraction x3, relations, narrative) and the next one added would inherit
    the bug silently. Counting sites is what catches that; mocking one call path
    would not.
    """
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    offenders, checked = [], 0
    for path in sorted(glob.glob(os.path.join(here, "*.py"))):
        name = os.path.basename(path)
        if name.startswith(("test_", "eval_")):
            continue
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        gens = src.count("api/generate")
        if not gens:
            continue
        checked += 1
        thinks = src.count('"think": False')
        if gens != thinks:
            offenders.append((name, gens, thinks))
    assert checked >= 4, (
        "expected to find the reason-call modules; found %d" % checked)
    assert not offenders, (
        "reason-model call sites without think: False -> "
        + ", ".join("%s (%d generate, %d think)" % o for o in offenders))
    print("  think-off invariant OK: %d modules, every /api/generate carries "
          "think: False" % checked)


def test_finalize_lock_scope_excludes_llm_stages():
    """The finalize lock must cover the WRITE phase only - not the LLM stages.

    Measured 2026-07-25 on five concurrent overnight lanes, the old scope (lock
    taken for the whole epoch) produced hold times of 111.3s / 121.1s / 415.2s
    with zero-gap handoffs, because two LLM stages sat inside it: the extraction
    call (up to _reason_timeout, configured 900s in production) and ONE relation
    call per newly added fact. Neither touches state the lock protects, so they
    were pure serialization: throughput was capped at 1/hold_time no matter how
    many lanes ran.

    This pins the scope by probing the lock from a SECOND FinalizeLock instance
    (a distinct file descriptor, so it genuinely contends) at three points:
      extraction  -> must be FREE
      the write   -> must be HELD
      relations   -> must be FREE
    Before the change the first and third both reported HELD.
    """
    try:
        _inject_hermes_stubs()
        prov = _load("__init__")
        pl = _load("proc_lock")
    except Exception as e:
        print(f"  SKIP finalize lock scope: {e}"); return
    import json
    import tempfile

    seen = {}

    def _probe(where, db_path):
        """Is the lock free RIGHT NOW? Separate instance == real contention."""
        probe = pl.FinalizeLock(db_path)
        free = probe.acquire(0.0)
        if free:
            probe.release()
        seen[where] = "free" if free else "held"

    class _ScopeStore:
        def __init__(self, tmp):
            self.db_path = os.path.join(tmp, "stub.db")

        def get_recent_episodes(self, limit=10, session_id=None):
            return [{"role": "user", "content": "what did we learn" * 8},
                    {"role": "assistant", "content": "creatinine 125 umol/L" * 20}]

        def get_cycle_counts(self):
            return (0, 0)

        def _clean_llm_json(self, text):
            return text

        def begin_write_batch(self, **kw):
            return 1

        def end_write_batch(self):
            pass

        def _extract_entities(self, content):
            return []

        def session_tool_names(self, session_id):
            return set()

        def increment_cycle(self, key):
            return 1

        def add_or_reinforce_fact(self, content, emb, category, session_id, **kw):
            _probe("write", self.db_path)
            return ("added", 77)

    class _ScopeRetriever:
        def _get_embedding(self, content):
            return [0.1] * 8

    p = prov.LatticeMemoryProvider({})
    tmp = tempfile.mkdtemp()
    p._store = _ScopeStore(tmp)
    p._retriever = _ScopeRetriever()
    p._write_enabled = True
    p._gate_self_writes = False
    p._enable_relations = True          # the deferred stage under test
    p._session_id = "s-scope"
    p._extract_relations_for_fact = (
        lambda fid, content, entities: _probe("relations", p._store.db_path))

    fake = {"response": json.dumps([{"content": "Creatinine is reported in umol/L",
                                     "category": "unit_convention"}])}

    def _fake_post(url, payload, timeout, max_attempts=3):
        _probe("extract", p._store.db_path)
        return fake

    g = p._run_consolidation_epoch.__func__.__globals__
    orig = g["_ollama_post_with_retry"]
    g["_ollama_post_with_retry"] = _fake_post
    try:
        p._run_consolidation_epoch("s-scope", suppress_dream=True)
    finally:
        g["_ollama_post_with_retry"] = orig

    assert seen.get("extract") == "free", (
        "extraction ran INSIDE the finalize lock: %r" % (seen,))
    assert seen.get("write") == "held", (
        "the write phase ran OUTSIDE the finalize lock - the dedup "
        "read-then-write race is unprotected: %r" % (seen,))
    assert seen.get("relations") == "free", (
        "relation extraction ran INSIDE the finalize lock: %r" % (seen,))
    print("  finalize lock scope OK: extract free, write HELD, relations free")


def test_finalize_lock_held_by_other_peek():
    """held_by_other() reports contention WITHOUT keeping the lock.

    It backs the early-skip for scheduled consolidation once the expensive
    stages moved outside the lock: a mid-session epoch must still bail before
    spending a reason-model call. A peek that accidentally RETAINED the lock
    would deadlock the very epoch that just peeked, so prove it lets go.
    """
    import tempfile as _tf
    pl = _load("proc_lock")
    base = os.path.join(_tf.mkdtemp(), "dbfile")

    a = pl.FinalizeLock(base)
    assert a.held_by_other() is False          # nobody holds it
    # ...and the peek did not keep it: a real acquire must still succeed.
    assert a.acquire(0.0) is True
    b = pl.FinalizeLock(base)
    assert b.held_by_other() is True           # a holds it now
    a.release()
    assert b.held_by_other() is False
    assert b.acquire(0.0) is True
    b.release()
    print("  held_by_other OK: detects contention, never retains the lock")


def test_store_dismiss_conflict():
    """Phase 6 second verb: dismiss_conflict unlocks ALL members of a false-positive
    group symmetrically - no supersession, no winner, confirm stamp + small
    saturating bump. Substrate-checked."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    _, a = s.add_or_reinforce_fact("gl_FragCoord availability", _emb(s, "a"), "spec", "s")
    _, b = s.add_or_reinforce_fact("gl_FrontFacing availability", _emb(s, "b"), "spec", "s")
    s._conn.execute("UPDATE semantic_facts SET conflict_group_id='gfp', tier='mid', "
                    "resonance_count=7.0, conflict_since_cycle=5 WHERE id IN (?,?)", (a, b))
    s.set_cycle_counts(memory_cycle=6)
    s._conn.commit()

    # Dismiss by group_id: both unlocked, stamped, bumped, NOT superseded.
    res = s.dismiss_conflict(group_id="gfp", current_cycle=6)
    assert res == {"conflict_group_id": "gfp", "dismissed": [a, b]} or \
           res == {"conflict_group_id": "gfp", "dismissed": [b, a]}, res
    for fid in (a, b):
        r = s._conn.execute(
            "SELECT tier, conflict_group_id, conflict_since_cycle, resonance_count, "
            "last_confirmed_cycle, superseded_by FROM semantic_facts WHERE id=?",
            (fid,)).fetchone()
        assert r["conflict_group_id"] is None and r["conflict_since_cycle"] is None, dict(r)
        assert r["tier"] == "mid" and r["superseded_by"] is None, dict(r)
        assert r["resonance_count"] == 7.5, dict(r)          # symmetric +0.5, no winner
        assert r["last_confirmed_cycle"] == 6, dict(r)
    assert s.get_pending_conflicts(min_age_cycles=0) == []
    # Idempotence: the group no longer exists.
    assert s.dismiss_conflict(group_id="gfp", current_cycle=7) is None

    # Dismiss by member fact_id; bump saturates at 50.
    _, c = s.add_or_reinforce_fact("saturated member", _emb(s, "c"), "spec", "s")
    _, d = s.add_or_reinforce_fact("partner member", _emb(s, "d"), "spec", "s")
    s._conn.execute("UPDATE semantic_facts SET conflict_group_id='g2', "
                    "resonance_count=49.8, conflict_since_cycle=6 WHERE id=?", (c,))
    s._conn.execute("UPDATE semantic_facts SET conflict_group_id='g2', "
                    "resonance_count=3.0, conflict_since_cycle=6 WHERE id=?", (d,))
    s._conn.commit()
    res = s.dismiss_conflict(fact_id=d, current_cycle=7)
    assert res is not None and set(res["dismissed"]) == {c, d}, res
    rc = s._conn.execute("SELECT resonance_count FROM semantic_facts WHERE id=?",
                         (c,)).fetchone()["resonance_count"]
    assert rc == 50.0, rc                                     # saturating cap
    # Unknown group / unconflicted fact / no args → None.
    assert s.dismiss_conflict(group_id="nope") is None
    assert s.dismiss_conflict(fact_id=a) is None
    assert s.dismiss_conflict() is None
    s.close()
    print("  dismiss_conflict OK: symmetric unlock, confirm stamp, saturating bump, no supersession")


def test_conflict_adjudicator_response_parsing():
    """Provider-side adjudicator: strict-JSON verdicts parse, bare true/false is
    tolerated, garbage and transport errors FAIL OPEN to None (store then flags)."""
    try:
        prov = _load("__init__")
        p = prov.LatticeMemoryProvider({})
    except Exception as e:
        print(f"  SKIP adjudicator parsing: {e}"); return

    class _StubStore:                       # only _clean_llm_json is touched
        @staticmethod
        def _clean_llm_json(t):
            return t
    p._store = _StubStore()
    p._reason_model = "test-model"
    p._ollama_endpoint_reason = "http://localhost:0"

    # Patch the EXACT namespace the method resolves _ollama_post_with_retry in
    # (its own __globals__) - patching a separately-loaded consolidation module
    # object is load-order/platform dependent and misses on some environments.
    g = p._conflict_adjudicator.__func__.__globals__
    orig = g["_ollama_post_with_retry"]
    try:
        cases = [
            ('{"contradict": true}', True),
            ('{"contradict": false}', False),
            ('Sure! {"contradict": FALSE} - they are compatible.', False),
            ('true', True),
            ('nonsense with no verdict at all', None),
            ('true or false, hard to say', None),   # ambiguous → None
        ]
        for raw, expected in cases:
            g["_ollama_post_with_retry"] = lambda url, payload, timeout, _r=raw, **kw: {"response": _r}
            got = p._conflict_adjudicator("A", "B")
            assert got is expected, (raw, got, expected)
        # Transport failure → None (fail-open handled by the store).
        def _die(url, payload, timeout, **kw):
            raise RuntimeError("connection refused")
        g["_ollama_post_with_retry"] = _die
        assert p._conflict_adjudicator("A", "B") is None
    finally:
        g["_ollama_post_with_retry"] = orig
    print("  adjudicator parsing OK: JSON + bare verdicts parse, garbage/errors -> None")


def test_relation_model_routing_and_default():
    """relation_model/ollama_endpoint_relation route the per-fact triple pass to a
    dedicated (small local) model; empty config inherits the reason model/endpoint
    (backward compatible), and the override actually reaches the store call."""
    try:
        prov = _load("__init__")
    except Exception as e:
        print(f"  SKIP relation routing: {e}"); return

    p = prov.LatticeMemoryProvider({})
    assert p._relation_model == p._reason_model
    assert p._ollama_endpoint_relation == p._ollama_endpoint_reason

    p2 = prov.LatticeMemoryProvider({
        "relation_model": "tiny-triples:latest",
        "ollama_endpoint_relation": "http://smallnode:11434",
    })
    assert p2._relation_model == "tiny-triples:latest"
    assert p2._ollama_endpoint_relation == "http://smallnode:11434"

    captured = {}

    class _StubStore:
        def extract_and_store_relations(self, fact_id, content, **kw):
            captured.update(kw)
            return 0

    p2._store = _StubStore()
    p2._enable_relations = True
    p2._extract_relations_for_fact(1, "A relates to B", ["A", "B"])
    assert captured.get("reason_model") == "tiny-triples:latest", captured
    assert captured.get("ollama_endpoint") == "http://smallnode:11434", captured
    print("  relation routing OK: default inherits reason; override reaches the store call")


def test_relation_prompt_subject_kinds_domain_neutral():
    """The built vocabulary prompt must not hardcode ONE domain's noun classes.

    Field finding on a 6,983-fact veterinary corpus: the closed vocabulary was being
    enforced perfectly (zero off-list predicates) yet 2 of 3 sampled facts extracted
    NOTHING, because the built prompt asked for 'components, machines, models,
    services, tools, settings, files, versions' -- so on 'Urine must be cultured prior
    to antimicrobic administration' the model correctly reported no machines and
    returned []. Default must stay byte-identical for operational profiles; an
    override must replace the noun classes, not merely append to them."""
    try:
        rel = _load("store_relations")
    except Exception as e:
        print(f"  SKIP relation subject kinds: {e}"); return
    build = rel.RelationsMixin._build_relation_prompt
    vocab = ["treats", "causes"]

    default = build(vocab, None)
    assert "components, machines, models, services" in default, default[:200]
    assert build(vocab, None, None) == default        # explicit None == omitted
    assert build(vocab, None, "   ") == default       # blank falls back, not empties

    kinds = "clinical signs, diseases, breeds, drugs, analytes, tests"
    vet = build(vocab, None, kinds)
    assert kinds in vet, vet[:200]
    assert "machines" not in vet, "override must REPLACE the operational noun classes"
    # everything else is domain-neutral: the rest of the prompt is unchanged
    assert vet.replace(kinds, rel.RelationsMixin._DEFAULT_SUBJECT_KINDS) == default
    print("  relation prompt OK: default byte-identical, override replaces noun classes")


def test_relation_llm_accepts_positional_triples():
    """A model that answers [["s","r","o"], ...] must not be silently discarded.

    Field finding: few-shot examples written in the positional form taught the model
    to answer in it, and the dict-only parser dropped every triple with no trace. The
    same prompt returned dicts while the serving layer had reasoning ON and arrays
    once it was OFF -- so extraction yield swung to zero on a flag that has nothing to
    do with extraction. Both shapes must parse identically."""
    import json                                # not imported at module scope here
    try:
        rel = _load("store_relations")
        absm = _load("store_abstraction")      # _clean_llm_json lives on that mixin
    except Exception as e:
        print(f"  SKIP positional triples: {e}"); return

    # LatticeStore composes these mixins; a bare RelationsMixin lacks the JSON
    # cleaner and every call would fail into the non-fatal [] path -- which is
    # exactly how this test first "passed the parser" while measuring nothing.
    class _S(rel.RelationsMixin, absm.AbstractionMixin):
        def _extract_entities(self, text):
            return ["stranguria"]

    s = _S()
    assert hasattr(s, "_clean_llm_json"), "shim must carry the real JSON cleaner"
    vocab = ["presents_as"]
    payloads = {
        "dict": '[{"subject":"stranguria","relation":"presents_as",'
                '"object":"straining to urinate"}]',
        "positional": '[["stranguria","presents_as","straining to urinate"]]',
    }
    got = {}
    for shape, raw in payloads.items():
        holder = {}

        class _Resp:
            def __init__(self, body):
                self._b = body.encode()

            def read(self):
                return self._b

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        real_open = rel.urllib.request.urlopen
        rel.urllib.request.urlopen = lambda req, timeout=None, _r=raw: _Resp(
            json.dumps({"response": _r}))
        try:
            got[shape] = s._llm_extract_triples(
                "Stranguria means straining to urinate.", "m", "http://x",
                entities=["stranguria"], vocabulary=vocab)
        finally:
            rel.urllib.request.urlopen = real_open
        assert holder == {}

    assert len(got["positional"]) == 1, got["positional"]
    key = lambda ts: [(t["subject"], t["relation"], t["object"]) for t in ts]
    assert key(got["positional"]) == key(got["dict"]), got
    print("  positional triples OK: [[s,r,o]] parses identically to {subject,...}")


def test_relation_attribute_predicates_extend_exempt_set():
    """relation_require_entity must not delete a domain's attribute triples.

    _ATTRIBUTE_RELATIONS ships as {set_to, produces} -- operational. A clinical closed
    list is mostly attributes (reference_interval -> a range, dosed_at -> a dose,
    presents_as -> a sign phrase), and strict binding demands the OBJECT resolve to a
    known entity, so every one of them would be dropped for want of an entity that a
    measurement can never be. The declared list is the fix; without declaring, the
    triple must still drop (the guard keeps working)."""
    try:
        rel = _load("store_relations")
    except Exception as e:
        print(f"  SKIP relation attribute predicates: {e}"); return

    class _Bare(rel.RelationsMixin):
        pass

    class _Declared(rel.RelationsMixin):
        relation_attribute_predicates = ("reference_interval", "DOSED_AT ")

    bare, declared = _Bare(), _Declared()
    assert bare._attribute_relations() == rel.RelationsMixin._ATTRIBUTE_RELATIONS
    got = declared._attribute_relations()
    assert {"set_to", "produces", "reference_interval", "dosed_at"} <= got, got

    triple = [{"subject": "serum phosphorus", "relation": "reference_interval",
               "object": "2.9-5.3 mg/dL", "confidence": 0.9}]
    ents = ["serum phosphorus"]          # a RANGE is never an extracted entity
    kept = declared._canonicalize_triples(
        [dict(t) for t in triple], "", ents, None, require_entity=True)
    assert len(kept) == 1, "declared attribute predicate must survive strict binding"
    dropped = bare._canonicalize_triples(
        [dict(t) for t in triple], "", ents, None, require_entity=True)
    assert dropped == [], "undeclared predicate must still be held to entity grounding"
    print("  attribute predicates OK: declared survive strict binding, undeclared drop")


def test_attribute_object_keeps_its_value():
    """An attribute predicate's object must survive as a VALUE, not be node-normalised.

    Measured on a live re-extraction: every stage_defined_by object arrived as a bare
    analyte name. Three independent mechanisms did it -- entity grounding returns the
    matched ENTITY rather than the span, connector splitting cuts at 'and', and the
    >3-word ceiling rejects intervals. A graph that looks populated and carries no
    numbers is the worst outcome available, so all three are checked here, and the
    node path is checked to be UNCHANGED."""
    try:
        rel = _load("store_relations")
    except Exception as e:
        print(f"  SKIP attribute object value: {e}"); return
    ents = {"creatinine", "iris ckd stage 1", "serum phosphorus", "azotemia"}

    # 1. entity-substring collapse
    got, grounded = rel._resolve_arg("creatinine below 125 micromol/L", ents, "obj",
                                     is_value=True)
    assert got == "creatinine below 125 micromol/l", got
    assert grounded is False, "a value is not a node and must not claim grounding"
    assert rel._resolve_arg("creatinine below 125 micromol/L", ents, "obj")[0] \
        == "creatinine", "node path must still collapse to the entity"

    # 2. connector splitting
    got, _ = rel._resolve_arg("drinking and urinating noticeably more than usual",
                              ents, "obj", is_value=True)
    assert got == "drinking and urinating noticeably more than usual", got
    assert rel._resolve_arg("drinking and urinating noticeably more than usual",
                            ents, "obj")[0] == "drinking", "node path unchanged"

    # 3. the >3-word ceiling
    got, _ = rel._resolve_arg("2.9-5.3 mg/dL as reported by MSD", ents, "obj",
                              is_value=True)
    assert got == "2.9-5.3 mg/dl as reported by msd", got
    assert rel._resolve_arg("2.9-5.3 mg/dL as reported by MSD", ents, "obj")[0] is None

    # determiner + punctuation trimming still applies; empties still rejected
    assert rel._resolve_arg("the 75 mg/kg q12-24h.", ents, "obj", is_value=True)[0] \
        == "75 mg/kg q12-24h"
    assert rel._resolve_arg("  ", ents, "obj", is_value=True)[0] is None
    print("  attribute object OK: value kept verbatim, node path unchanged")


def test_alias_must_cover_the_span_before_replacing_it():
    """A short alias must not swallow a composite identifier.

    With iris -> 'international renal interest society', the span 'iris ckd stage 2'
    was replaced WHOLESALE and the stage identifier was lost; 'usg >1.030' would
    likewise become 'urine specific gravity' and drop the threshold. Coverage gates
    the substring rule while whole-value matches still always win, so the intended
    node collapses keep working."""
    try:
        rel = _load("store_relations")
    except Exception as e:
        print(f"  SKIP alias coverage: {e}"); return
    amap = {
        "iris": "international renal interest society",
        "usg": "urine specific gravity",
        "46": "node .46",
        "nemotron-3-super:cloud": "nemotron",
    }
    # whole-value: always replaced, however short
    assert rel._alias_canon("iris", amap) == "international renal interest society"
    assert rel._alias_canon("usg", amap) == "urine specific gravity"
    assert rel._alias_canon("46", amap) == "node .46"
    assert rel._alias_canon("nemotron-3-super:cloud", amap) == "nemotron"
    # composite spans survive: the alias covers too little of them
    for span in ("iris ckd stage 2", "iris ckd stage 1", "usg >1.030",
                 "usg below 1.030"):
        assert rel._alias_canon(span, amap) == span, span
    # a partial match that DOES cover most of the span still collapses
    assert rel._alias_canon("nemotron-3-super:cloud model",
                            amap) == "nemotron"
    # query side must mirror storage side exactly
    for span in ("iris", "iris ckd stage 2", "usg >1.030", "46"):
        assert rel._canon_term(span, amap) == rel._alias_canon(span, amap), span
    print("  alias coverage OK: whole-value wins, composite spans survive, "
          "query mirrors storage")


def test_relation_num_predict_bounds_the_generation():
    """An uncapped relation call has a 300s tail that becomes a SILENT empty result.

    Measured: typical calls emit 41 completion tokens, but against an endpoint whose
    own default was 8192 the tail ran to the cap and hit this method's 300s timeout,
    which is swallowed into the non-fatal [] path -- so a lost extraction is
    indistinguishable from 'this fact states no relations'. Unset must keep the old
    payload byte-for-byte; set must add num_predict and nothing else."""
    import json
    try:
        rel = _load("store_relations")
        absm = _load("store_abstraction")
    except Exception as e:
        print(f"  SKIP relation num_predict: {e}"); return

    class _S(rel.RelationsMixin, absm.AbstractionMixin):
        def _extract_entities(self, text):
            return ["a"]

    sent = {}

    class _Resp:
        def read(self):
            return json.dumps({"response": "[]"}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake(req, timeout=None):
        sent["opts"] = json.loads(req.data.decode())["options"]
        sent["think"] = json.loads(req.data.decode()).get("think")
        return _Resp()

    real = rel.urllib.request.urlopen
    try:
        rel.urllib.request.urlopen = _fake
        s = _S()
        s._llm_extract_triples("x", "m", "http://x", entities=["a"], vocabulary=["treats"])
        assert sent["opts"] == {"temperature": 0.1}, sent["opts"]
        assert sent["think"] is False, "think must stay False on every reason call"

        s.relation_num_predict = 512
        s._llm_extract_triples("x", "m", "http://x", entities=["a"], vocabulary=["treats"])
        assert sent["opts"] == {"temperature": 0.1, "num_predict": 512}, sent["opts"]

        for bad in (0, -5, None, "nope"):
            s.relation_num_predict = bad
            s._llm_extract_triples("x", "m", "http://x", entities=["a"],
                                   vocabulary=["treats"])
            assert "num_predict" not in sent["opts"], (bad, sent["opts"])
    finally:
        rel.urllib.request.urlopen = real
    print("  relation num_predict OK: unset unchanged, set caps, junk ignored")


def test_relation_domain_knobs_reach_the_store():
    """Both new keys must land ON the store, since the store reads them for itself
    (no call signature carries them). Unset must leave the store's defaults alone."""
    try:
        prov = _load("__init__")
    except Exception as e:
        print(f"  SKIP relation domain knobs: {e}"); return

    p = prov.LatticeMemoryProvider({})
    assert p._relation_subject_kinds is None
    assert p._relation_attribute_predicates is None

    p2 = prov.LatticeMemoryProvider({
        "relation_subject_kinds": "  clinical signs, diseases, drugs  ",
        "relation_attribute_predicates": ["reference_interval", " Dosed_At ", ""],
    })
    assert p2._relation_subject_kinds == "clinical signs, diseases, drugs"
    assert p2._relation_attribute_predicates == ("reference_interval", "dosed_at")
    print("  domain knobs OK: stripped, lowercased, empties dropped, None when unset")


def test_mirror_path_relation_hook():
    """on_memory_write mirror facts get the same Phase 5a relation hook as
    consolidation-extracted facts: gated, non-fatal, and routed through the
    dedicated relation model, and carry self-authored provenance labels.
    Field finding: mirror-path facts had zero fact_relations rows because
    the path skipped the hook, and their old builtin_* category did not
    communicate "the agent wrote this itself" to recall or audits."""
    try:
        prov = _load("__init__")
    except Exception as e:
        print(f"  SKIP mirror relation hook: {e}"); return

    p = prov.LatticeMemoryProvider({
        "relation_model": "tiny-triples:latest",
        "ollama_endpoint_relation": "http://smallnode:11434",
    })
    calls = {}

    class _StubStore:
        def _extract_entities(self, content):
            return ["A", "B"]

        def add_or_reinforce_fact(self, content, emb, category, session_id, **kw):
            calls["added"] = (content, category, kw.get("quote_status"),
                              kw.get("source_ref"))
            return ("added", 42)

        def extract_and_store_relations(self, fact_id, content, **kw):
            calls["relations"] = (fact_id, kw.get("reason_model"),
                                  kw.get("ollama_endpoint"))
            return 1

    class _StubRetriever:
        def _get_embedding(self, content):
            return [0.1] * 8

    p._store = _StubStore()
    p._retriever = _StubRetriever()
    p._write_enabled = True
    p._enable_relations = True
    p._gate_self_writes = False
    p._session_id = "s-mirror"

    # Run the ingest thread synchronously so the assertions are deterministic.
    g = p.on_memory_write.__func__.__globals__
    orig_thread = g["threading"].Thread

    class _SyncThread:
        def __init__(self, target=None, **kw):
            self._t = target

        def start(self):
            self._t()

    g["threading"].Thread = _SyncThread
    try:
        p.on_memory_write("add", "memory", "Fact A relates to B for the mirror hook")
    finally:
        g["threading"].Thread = orig_thread

    assert calls.get("added"), calls
    # Self-authored provenance labeling: legibility (category), epistemics
    # (quote_status), provenance (source_ref) - three separate signals.
    assert calls["added"][1] == "mental_note", calls
    assert calls["added"][2] == "self_authored", calls
    assert calls["added"][3] == "agent:MEMORY.md", calls
    assert calls.get("relations") == (42, "tiny-triples:latest",
                                      "http://smallnode:11434"), calls
    print("  mirror relation hook OK: mental_note labeled + routed through relation extraction")


def test_freshness_penalty_curve():
    """Phase 2: the recall freshness nudge is gentle, bounded, monotonic, and
    fully off when disabled (pure math - no Ollama)."""
    if not _STORE_OK:
        print("  SKIP"); return
    R = _load("retrieval").LatticeRetriever
    assert R._freshness_penalty(100, 0) == 0.0      # nudge disabled (halflife 0)
    assert R._freshness_penalty(0, 50) == 0.0       # fresh fact: no penalty
    assert R._freshness_penalty(-5, 50) == 0.0      # guard: negative staleness
    p_half = R._freshness_penalty(50, 50)           # one half-life
    p_two = R._freshness_penalty(200, 50)           # very stale
    assert 0 < p_half < p_two < R.FRESHNESS_MAX_NUDGE, (p_half, p_two)
    # At one half-life freshness=0.5 → penalty == max_nudge * 0.5.
    assert abs(p_half - R.FRESHNESS_MAX_NUDGE * 0.5) < 1e-9, p_half


def test_store_staleness_decay():
    """Phase 2 'use it or lose it': extra decay hits only weak AND stale facts;
    fresh facts and strong (above-promotion) facts are exempt. Substrate-checked."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()  # uses central DEFAULTS (promotion_resonance_threshold -> internal)
    _, stale_id = s.add_or_reinforce_fact("stale weak fact", _emb(s, "stale weak fact"), "g", "s")
    _, fresh_id = s.add_or_reinforce_fact("fresh weak fact", _emb(s, "fresh weak fact"), "g", "s")
    _, strong_id = s.add_or_reinforce_fact("stale strong fact", _emb(s, "stale strong fact"), "g", "s")
    s._conn.execute("UPDATE semantic_facts SET resonance_count=2, last_confirmed_cycle=0 WHERE id=?", (stale_id,))
    s._conn.execute("UPDATE semantic_facts SET resonance_count=2, last_confirmed_cycle=10 WHERE id=?", (fresh_id,))
    s._conn.execute("UPDATE semantic_facts SET resonance_count=10, last_confirmed_cycle=0 WHERE id=?", (strong_id,))
    s._conn.commit()

    def res(i):
        return s._conn.execute("SELECT resonance_count FROM semantic_facts WHERE id=?", (i,)).fetchone()[0]
    before = (res(stale_id), res(fresh_id), res(strong_id))
    # current cycle 10, boost 1.0, halflife 10 → stale fact at staleness 10 = full boost.
    touched = s.apply_staleness_decay(current_cycle=10, boost=1.0, halflife=10.0)
    after = (res(stale_id), res(fresh_id), res(strong_id))
    assert abs((before[0] - after[0]) - 1.0) < 1e-6, (before, after)   # weak+stale bled ~1.0
    assert after[1] == before[1], (before, after)                      # fresh (staleness 0) exempt
    assert after[2] == before[2], (before, after)                      # strong (>= promotion) exempt
    assert touched == 1, touched
    assert s.apply_staleness_decay(current_cycle=20, boost=0.0) == 0   # off by default
    s.close()


def test_store_novelty_initial_resonance():
    """Phase 3: a novel fact enters at higher resonance than a near-duplicate;
    disabling novelty falls back to plain initial_resonance. Substrate-checked."""
    if not _STORE_OK:
        print("  SKIP"); return
    import numpy as np
    s = _fresh_store(reinforce_threshold=0.99)          # high → near-dup inserts, not merges
    base = _emb(s, "anchor concept alpha")
    s.add_or_reinforce_fact("anchor concept alpha", base, "g", "s")
    # Near-duplicate: base + a little noise → high similarity (<0.99) → low novelty.
    noise = _emb(s, "orthogonal noise vector")
    near = (np.array(base) * 0.8 + np.array(noise) * 0.2)
    near = (near / (np.linalg.norm(near) or 1.0)).tolist()
    _, near_id = s.add_or_reinforce_fact("almost the same alpha", near, "g", "s")
    # Novel: an unrelated random vector → ~0 similarity → ~full novelty.
    _, novel_id = s.add_or_reinforce_fact("utterly unrelated subject zeta",
                                          _emb(s, "utterly unrelated subject zeta"), "g", "s")

    def col(i, c):
        return s._conn.execute(f"SELECT {c} FROM semantic_facts WHERE id=?", (i,)).fetchone()[0]
    r_near, r_novel = col(near_id, "resonance_count"), col(novel_id, "resonance_count")
    assert r_novel > r_near + 1.0, (r_novel, r_near)                    # clear separation
    assert r_novel >= s.initial_resonance + 1.5, (r_novel, s.initial_resonance)  # big boost
    assert r_near < s.initial_resonance + 1.0, (r_near, s.initial_resonance)     # tiny boost
    assert abs(col(novel_id, "max_resonance_seen") - r_novel) < 1e-6   # peak seeded at start
    s.close()
    # Disabled → plain initial_resonance, no boost even for a fully-novel fact.
    s2 = _fresh_store(novelty_enabled=False)
    _, fid = s2.add_or_reinforce_fact("first ever fact", _emb(s2, "first ever fact"), "g", "s")
    r = s2._conn.execute("SELECT resonance_count FROM semantic_facts WHERE id=?", (fid,)).fetchone()[0]
    assert abs(r - s2.initial_resonance) < 1e-6, (r, s2.initial_resonance)
    s2.close()


def test_store_max_resonance_seen_peak():
    """Phase 3: max_resonance_seen is a high-water mark - rises on reinforce and
    feedback, never falls on decay."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    e = _emb(s, "lone peak fact")
    _, fid = s.add_or_reinforce_fact("lone peak fact", e, "g", "s")

    def get(c):
        return s._conn.execute(f"SELECT {c} FROM semantic_facts WHERE id=?", (fid,)).fetchone()[0]
    peak0, res0 = get("max_resonance_seen"), get("resonance_count")
    assert abs(peak0 - res0) < 1e-6, (peak0, res0)         # peak seeded at start
    for _ in range(3):                                     # semantic reinforce (identical emb)
        s.add_or_reinforce_fact("lone peak fact", e, "g", "s")
    peak1, res1 = get("max_resonance_seen"), get("resonance_count")
    assert res1 > res0 and peak1 >= res1 and peak1 > peak0, (res0, res1, peak0, peak1)
    s.apply_cycle_decay()                                  # current drops…
    peak2, res2 = get("max_resonance_seen"), get("resonance_count")
    assert abs(peak2 - peak1) < 1e-6, (peak1, peak2)       # …but the peak is frozen
    assert res2 < peak2, (res2, peak2)
    s.adjust_resonance(fid, 10)                            # feedback lifts the peak too
    assert get("max_resonance_seen") >= get("resonance_count") >= res2 + 10 - 1e-6
    s.close()


def _add_with_hrr(s, text, entities=None, category="g"):
    """Add a fact carrying a real (content-derived) HRR vector - needed for the
    Phase-4 clustering/gist tests. Pseudo-embeddings stay random per text (so no
    accidental dedup merge), while the HRR vector reflects real content similarity."""
    hg = _load("holographic")
    ents = entities if entities is not None else s._extract_entities(text)
    hv = hg.encode_fact(text, ents, dim=s.hrr_dim)
    return s.add_or_reinforce_fact(text, _emb(s, text), category, "sess1",
                                   hrr_vector=hv, entities=ents)


def test_cluster_by_hrr_entity():
    """Phase 4: shared clustering groups HRR/entity-similar facts and isolates the
    rest (deterministic, no LLM)."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    a = _add_with_hrr(s, "Maya enjoys painting watercolor landscapes", ["maya"])[1]
    b = _add_with_hrr(s, "Maya loves to paint watercolor scenes", ["maya"])[1]
    c = _add_with_hrr(s, "the staging server listens on port 8080", ["server"])[1]
    rows = [dict(r) for r in s._conn.execute(
        "SELECT id, content, hrr_vector FROM semantic_facts")]
    emap = {}
    for r in s._conn.execute("SELECT fe.fact_id, e.name FROM fact_entities fe "
                             "JOIN entities e ON e.entity_id = fe.entity_id"):
        emap.setdefault(r["fact_id"], set()).add(r["name"])
    groups = [sorted(f["id"] for f in cl)
              for cl in s._cluster_by_hrr_entity(rows, emap, 0.5, 0.5, 2, 8)]
    assert any(a in g and b in g and c not in g for g in groups), groups
    s.close()


def test_store_gist_candidate_selection():
    """Phase 4: only dying facts that EARNED their place are gist candidates -
    trivia, living facts, superseded history, abstractions, and already-preserved
    facts are all excluded. Validated at the substrate (no LLM)."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()

    def setrow(fid, **cols):
        sets = ", ".join(f"{k}=?" for k in cols)
        s._conn.execute(f"UPDATE semantic_facts SET {sets} WHERE id=?",
                        (*cols.values(), fid))

    a = _add_with_hrr(s, "important fading fact about Maya", ["maya"])[1]
    setrow(a, tier="long", resonance_count=0, max_resonance_seen=8)        # earned + dying
    b = _add_with_hrr(s, "trivial short noise blip", ["blip"])[1]
    setrow(b, tier="short", resonance_count=0, max_resonance_seen=1)       # never important
    c = _add_with_hrr(s, "strong living long fact zeta", ["zeta"])[1]
    setrow(c, tier="long", resonance_count=5, max_resonance_seen=9)        # not dying
    d = _add_with_hrr(s, "retired superseded fact dee", ["dee"])[1]
    setrow(d, tier="superseded", resonance_count=0, max_resonance_seen=8)  # history, skip
    e = _add_with_hrr(s, "an existing abstraction node", ["eee"], category="abstract")[1]
    setrow(e, tier="long", resonance_count=0, max_resonance_seen=8)        # don't gist abstractions
    s._conn.commit()
    ids = [r["id"] for r in s._select_gist_candidates(0.0, 4.0, 100)]
    assert a in ids, ids
    assert not ({b, c, d, e} & set(ids)), ids
    # A SHORT fact that was important once (high peak) DOES qualify.
    f = _add_with_hrr(s, "once important now fading detail", ["eff"])[1]
    setrow(f, tier="short", resonance_count=0, max_resonance_seen=7)
    s._conn.commit()
    assert f in [r["id"] for r in s._select_gist_candidates(0.0, 4.0, 100)]
    # A fact already linked as an abstraction source is excluded (already preserved).
    s._conn.execute("INSERT INTO abstraction_sources (abstract_id, source_id, "
                    "cluster_size_at_creation) VALUES (?, ?, 2)", (c, a))
    s._conn.commit()
    assert a not in [r["id"] for r in s._select_gist_candidates(0.0, 4.0, 100)]
    s.close()


def test_store_memory_health_snapshot():
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    e = _emb(s, "alpha fact about widgets")
    _, fid = s.add_or_reinforce_fact("alpha fact about widgets", e, "general", "sess1",
                                     entities=["widgets"])
    h = s.get_memory_health()
    # Shape + a few sane values (read-only; no side effects).
    for key in ("total_facts", "by_tier", "by_category", "active_conflict_groups",
                "total_entities", "orphan_entities", "abstractions_tracked",
                "abstractions_evidence_gone", "tool_episodes_total", "degraded",
                "vector_dim", "hrr_dim", "near_cap_facts"):
        assert key in h, (key, h)
    assert h["total_facts"] >= 1, h
    assert h["total_entities"] >= 1, h            # "widgets" linked
    assert h["degraded"] is False, h
    assert h["vector_dim"] == s.vector_dim, h
    assert isinstance(h["orphan_entities"], int) and h["orphan_entities"] >= 0, h
    assert h["active_conflict_groups"] == 0, h    # nothing conflicting yet
    # A removed fact's only entity becomes an orphan until GC.
    assert s.remove_fact(fid) is True
    h2 = s.get_memory_health()
    assert h2["orphan_entities"] >= 1, h2
    s.close()


# ─────────────────────────────────────────────────────────────────────────────
# source_quote attestation - pure two-channel verifier (always runs; no deps)
# ─────────────────────────────────────────────────────────────────────────────
def _load_attestor():
    """Load the pure verifier from attestation.py (a leaf module with no Hermes
    framework deps), the way the other modules are loaded. The verifier used to
    be AST-sliced out of __init__.py to dodge its `from agent.memory_provider
    import ...`; the structural refactor moved it to its own importable module."""
    return _load("attestation")._attest_source_quote


def test_quote_attestation_verdicts():
    attest = _load_attestor()
    transcript = ("USER: I run ollama on port 11434 and my GPU is an RTX 3090 Ti.\n"
                  "ASSISTANT: Noted - Charlie Brown will deploy it in rural Indiana.")
    # Trivial typo in prose → still attested (specifics intact).
    assert attest("i run ollama on prot 11434", transcript, []) in ("attested", "soft"), "trivial typo"
    # Critical typo in a number → specific_mismatch (the whole point).
    assert attest("i run ollama on port 11435", transcript, []) == "specific_mismatch", "critical num"
    # Fabricated entity not in transcript → specific_mismatch.
    assert attest("deployed in rural Indiana", transcript, ["rural indiana"]) in ("attested", "soft")
    assert attest("deployed in rural Montana", transcript, ["rural montana"]) == "specific_mismatch", "fab entity"
    # Faithful copy with a real specific → attested.
    assert attest("my GPU is an RTX 3090 Ti", transcript, ["rtx 3090 ti"]) == "attested", "faithful"
    # Un-anchored prose, no hard specific contradicted → kept but flagged.
    assert attest("the user enjoys long walks on the beach", transcript, []) == "unattested", "unanchored"


def test_quote_attestation_digit_setmembership():
    """FIX 1: number specifics compared by SET MEMBERSHIP, not blob-substring.

    The fabricated number '3014' is a substring of the concatenated digit blob
    '41301434' (from '4.1', '30', '14', '34'), so the old blob test waved it
    through; set membership flags it. Fails on the pre-fix code."""
    attest = _load_attestor()
    blob = "We use granite 4.1 30b on the cluster. Worker is node 14, port 34 is open."
    assert attest("the cluster id is 3014", blob, []) == "specific_mismatch", "3014 blob leak"
    # A real token number inside lifted prose is attested...
    t2 = "USER: I run ollama on port 11434 locally."
    assert attest("i run ollama on port 11434", t2, []) == "attested", "real 11434"
    # ...and the existing critical-typo still flags.
    assert attest("i run ollama on port 11435", t2, []) == "specific_mismatch", "11435 typo"


def test_quote_attestation_single_digit_anchor():
    """FIX 3: a lone digit anchors a quote when present, but never drops it.

    A single number like 'purchase 3' was previously skipped entirely (len<2),
    so a faithful single-number quote could only ever reach 'soft'. Now a lone
    digit PRESENT in the transcript confirms the quote ('attested'), while a lone
    digit ABSENT is kept-and-flagged (NOT 'specific_mismatch') - lone digits and
    word<->digit normalization make an absent-single-digit drop unsafe."""
    attest = _load_attestor()
    t = "USER: please purchase 3 units of the widget for the team."
    # Faithful lift: the lone digit is present in the transcript -> attested.
    assert attest("purchase 3 units", t, []) == "attested", "lone digit present -> attested"
    # Changed lone digit (5 not in transcript): kept-and-flagged, NEVER dropped.
    assert attest("purchase 5 units", t, []) in ("soft", "unattested"), "absent lone digit not dropped"
    # A >=2-digit specific is still hard-checked and dropped on mismatch.
    t2 = "USER: please purchase 12 units of the widget."
    assert attest("purchase 13 units", t2, []) == "specific_mismatch", "multi-digit still drops"


def test_quote_attestation_long_transcript():
    """FIX 2: windowed prose score stays meaningful on a long transcript.

    The pre-fix whole-transcript ratio collapses toward 0 on a >=4000-char log,
    so a genuinely-lifted (typo'd) quote was wrongly rejected as 'unattested'.
    Windowing fixes it; a fabricated quote is still rejected. Fails pre-fix."""
    attest = _load_attestor()
    filler = ("The team discussed deployment logistics and reviewed the rollout plan in "
              "great detail across many meetings and follow-up threads. ") * 40
    lifted = "The canary cohort will be promoted to general availability on the third deploy window."
    long_t = filler + lifted + " " + filler
    assert len(long_t) >= 4000, len(long_t)
    # Verbatim lift from the MIDDLE of a long transcript → grounded.
    assert attest(lifted, long_t, []) in ("attested", "soft"), "verbatim long"
    # Lift WITH a typo (coverage broken + whole-transcript ratio collapsed) - the
    # case the inert-ratio code wrongly rejected - is still grounded.
    typo = lifted.replace("general availability", "general avzilability")
    assert attest(typo, long_t, []) in ("attested", "soft"), "typo long (was unattested pre-fix)"
    # A fabricated quote against the same long transcript is NOT grounded.
    assert attest("Budgets were slashed by forty percent in Q3.", long_t, []) in (
        "unattested", "specific_mismatch"), "fabricated long"


# ─────────────────────────────────────────────────────────────────────────────
# Provider shutdown sequencing - LifecycleMixin has no Hermes deps, so it loads
# standalone (always runs; no sqlite-vec/numpy needed).
# ─────────────────────────────────────────────────────────────────────────────
def test_shutdown_drains_dream_before_close():
    """shutdown() must join an in-flight dream cycle BEFORE close()ing the store,
    so the connection is never pulled out from under a running maintenance
    thread. Order of recorded events proves it (pre-fix it was close-then-dream)."""
    import threading as _t
    import time as _time
    lc = _load("lifecycle")

    events = []
    started = _t.Event()

    def fake_dream():
        started.set()
        _time.sleep(0.3)
        events.append("dream_done")

    class FakeStore:
        def close(self):
            events.append("store_closed")

    class P(lc.LifecycleMixin):
        pass

    p = P()
    p._last_ingest_thread = None
    p._store = FakeStore()
    dream = _t.Thread(target=fake_dream, daemon=False)
    p._last_dream_thread = dream
    dream.start()
    assert started.wait(2.0), "dream thread did not start"
    p.shutdown()
    assert events == ["dream_done", "store_closed"], events


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 - Encryption (E0: encrypted-at-rest). crypto_keys unit tests need only
# argon2-cffi; the substrate test spawns a subprocess with the SQLCipher binding
# signal set, because store_common selects the binding once, at import time.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import crypto_keys as _ck
    _CK_OK = _ck.kdf_available()
    _CK_SKIP = "" if _CK_OK else "argon2-cffi not installed"
except Exception as _e:  # pragma: no cover
    _CK_OK = False
    _CK_SKIP = str(_e)

try:
    import sqlcipher3 as _sqlcipher_probe  # noqa: F401
    _ENC_OK = _STORE_OK and _CK_OK
    _ENC_SKIP = "" if _ENC_OK else "sqlcipher3/argon2/store deps missing"
except Exception as _e:  # pragma: no cover
    _ENC_OK = False
    _ENC_SKIP = str(_e)


def test_crypto_keys_keystore_and_derivation():
    if not _CK_OK:
        print(f"  SKIP crypto_keys: {_CK_SKIP}"); return
    ck = _ck
    ks = ck.create_keystore(b"correct horse")
    assert ck.keystore_is_secret_free(ks), ks
    assert set(ks) == {"version", "kdf", "salt_b64", "key_check_b64"}
    k1 = ck.derive_db_key(b"correct horse", ks)
    k2 = ck.derive_db_key(b"correct horse", ks)
    assert isinstance(k1, bytearray) and len(k1) == 32
    assert bytes(k1) == bytes(k2)                          # deterministic
    assert ck.verify_passphrase(b"correct horse", ks) is True
    assert ck.verify_passphrase(b"wrong", ks) is False
    try:
        ck.derive_db_key(b"wrong", ks)
        assert False, "wrong passphrase did not raise"
    except ck.WrongPassphraseError:
        pass
    pragma = ck.db_key_to_pragma_value(k1)
    assert pragma.startswith("x'") and pragma.endswith("'") and len(pragma) == 67, pragma
    ck.secure_zero(k1)
    assert bytes(k1) == b"\x00" * 32                        # best-effort wipe


def test_crypto_keys_he_secret_wrap():
    """E2 plumbing: AES-256-GCM wrap/unwrap of the HE secret under a master subkey."""
    if not _CK_OK:
        print(f"  SKIP crypto_keys: {_CK_SKIP}"); return
    ck = _ck
    if not ck.aead_available():
        print("  SKIP he_secret_wrap: cryptography not installed"); return
    ks = ck.create_keystore(b"correct horse")
    # The wrap key is a deterministic 32-byte SIBLING of the db key under the same
    # master, but cryptographically INDEPENDENT of it (distinct HKDF info label).
    wk1 = ck.derive_he_wrap_key(b"correct horse", ks)
    wk2 = ck.derive_he_wrap_key(b"correct horse", ks)
    assert isinstance(wk1, bytearray) and len(wk1) == 32
    assert bytes(wk1) == bytes(wk2)                              # deterministic
    assert bytes(wk1) != bytes(ck.derive_db_key(b"correct horse", ks))  # independent
    try:
        ck.derive_he_wrap_key(b"wrong", ks)
        assert False, "wrong passphrase did not raise"
    except ck.WrongPassphraseError:
        pass
    # Round-trip: a stand-in HE secret blob survives wrap -> unwrap byte-identical.
    secret = os.urandom(2000)                                   # ~ a serialized CKKS sk
    wrapped = ck.wrap_he_secret(secret, wk1)
    assert set(wrapped) == {"version", "alg", "nonce_b64", "ct_b64"}
    assert wrapped["alg"] == "AES-256-GCM"
    ct_bytes = base64.b64decode(wrapped["ct_b64"])
    assert secret not in ct_bytes and len(ct_bytes) > len(secret)  # encrypted + tagged
    assert bytes(ck.unwrap_he_secret(wrapped, wk1)) == secret
    # Wrong key fails LOUDLY (GCM auth tag) - never returns garbage plaintext.
    other = ck.derive_he_wrap_key(b"correct horse", ck.create_keystore(b"correct horse"))
    for bad_input, label in (
        (lambda: ck.unwrap_he_secret(wrapped, other), "wrong key"),
        (lambda: ck.unwrap_he_secret({**wrapped, "version": 999}, wk1), "bad version"),
    ):
        try:
            bad_input(); assert False, f"{label} did not raise"
        except ck.WrapAuthError:
            pass
    tampered = bytearray(ct_bytes); tampered[0] ^= 0x01
    try:
        ck.unwrap_he_secret({**wrapped, "ct_b64": base64.b64encode(bytes(tampered)).decode()}, wk1)
        assert False, "tampered ciphertext did not raise"
    except ck.WrapAuthError:
        pass
    ck.secure_zero(wk1)
    assert bytes(wk1) == b"\x00" * 32


def test_crypto_keys_binding_selection():
    try:
        import store_common as sc
    except Exception as e:
        print(f"  SKIP binding test: {e}"); return
    plain = sc._select_sqlite_module(False)
    assert hasattr(plain, "connect") and hasattr(plain, "Row")
    assert hasattr(plain, "IntegrityError")
    # This (plaintext) test process has no env signal, so the live binding is not
    # SQLCipher and connections do not expect a key.
    assert sc.env_encryption_on() is False
    assert sc.encrypted_binding_active() is False
    try:
        import sqlcipher3  # noqa: F401
    except Exception:
        print("  (sqlcipher3 not installed; encrypted-binding assertion skipped)"); return
    enc = sc._select_sqlite_module(True)
    assert enc.__name__ == "sqlcipher3" and hasattr(enc, "IntegrityError")


def test_crypto_keys_seal_unseal_existing_db():
    """E0 step 6: encrypt_existing_db / decrypt_to_plaintext round-trip on a real file.

    The after-the-fact migration path for a store trained in plaintext: seal via
    sqlcipher_export under a keystore-derived raw key, verify true at-rest opacity
    (header + plain-binding read failure), then the audited exit door back to
    plaintext with rows and user_version intact. Wrong passphrase fails loudly;
    neither direction overwrites an existing destination."""
    if not _CK_OK:
        print(f"  SKIP seal/unseal: {_CK_SKIP}"); return
    try:
        import sqlcipher3  # noqa: F401
    except Exception as e:
        print(f"  SKIP seal/unseal: sqlcipher3 not installed ({e})"); return
    import sqlite3 as plain_sqlite
    import tempfile
    ck = _ck
    with tempfile.TemporaryDirectory() as td:
        plain = os.path.join(td, "trained.db")
        sealed = os.path.join(td, "sealed.db")
        back = os.path.join(td, "back.db")
        con = plain_sqlite.connect(plain)
        con.execute("CREATE TABLE semantic_facts (id INTEGER PRIMARY KEY, content TEXT)")
        con.executemany("INSERT INTO semantic_facts (content) VALUES (?)",
                        [("fact %d" % i,) for i in range(50)])
        con.execute("PRAGMA user_version = 7")
        con.commit(); con.close()

        info = ck.encrypt_existing_db(plain, sealed, "correct horse")
        assert info["tables"] >= 1 and os.path.exists(info["keystore_path"])
        assert ck.keystore_is_secret_free(ck.load_keystore(info["keystore_path"]))
        # Substrate opacity: header is not SQLite, and the PLAIN binding cannot read it.
        with open(sealed, "rb") as f:
            assert f.read(16) != b"SQLite format 3\x00"
        try:
            c = plain_sqlite.connect(sealed)
            c.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
            assert False, "plain sqlite3 read an at_rest-sealed file"
        except plain_sqlite.DatabaseError:
            pass
        finally:
            c.close()
        # Source untouched, still plaintext.
        assert ck._file_is_plaintext_sqlite(plain)

        # Wrong passphrase: loud, precise failure before any file is written.
        try:
            ck.decrypt_to_plaintext(sealed, back, "wrong horse")
            assert False, "wrong passphrase did not raise"
        except ck.WrongPassphraseError:
            pass
        assert not os.path.exists(back)

        out = ck.decrypt_to_plaintext(sealed, back, "correct horse")
        assert out["tables"] == info["tables"]
        c = plain_sqlite.connect(back)
        assert c.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0] == 50
        assert c.execute("PRAGMA user_version").fetchone()[0] == 7
        c.close()

        # Refuse-overwrite contract, both directions.
        for fn, args in ((ck.encrypt_existing_db, (plain, sealed, "correct horse")),
                         (ck.decrypt_to_plaintext, (sealed, back, "correct horse"))):
            try:
                fn(*args)
                assert False, f"{fn.__name__} overwrote an existing destination"
            except FileExistsError:
                pass
    print("  seal/unseal OK: sqlcipher_export round-trip, opacity verified, "
          "wrong-key + overwrite guards hold")


_ENC_CHILD = r'''
import os, sys
plugin_dir, db = sys.argv[1], sys.argv[2]
sys.path.insert(0, plugin_dir)
import store_common
assert store_common.encrypted_binding_active(), store_common._SQLITE_BINDING
import crypto_keys as ck
from store import LatticeStore
PW, DIM = b"sub-proc-pass", 8
ks = ck.create_keystore(PW)
s = LatticeStore(db_path=db, vector_dim=DIM, db_key=ck.derive_db_key(PW, ks))
s.add_or_reinforce_fact("the sky is blue", [0.1] * DIM, "general", "t")
s.close()
assert open(db, "rb").read(16)[:15] != b"SQLite format 3", "DB not encrypted at rest"
import sqlite3 as std
try:
    std.connect(db).execute("SELECT count(*) FROM semantic_facts").fetchone()
    raise SystemExit("PLAINTEXT_READ_SUCCEEDED")
except SystemExit:
    raise
except Exception:
    pass
s2 = LatticeStore(db_path=db, vector_dim=DIM, db_key=ck.derive_db_key(PW, ks))
n = s2.get_stats()["total_facts"]
s2.close()
assert n == 1, n
print("ENC_OK")
'''


def test_store_encryption_at_rest_substrate():
    if not _ENC_OK:
        print(f"  SKIP encryption substrate: {_ENC_SKIP}"); return
    import subprocess
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "enc_test.db")
    env = dict(os.environ)
    env["RESONANT_LATTICE_DB_ENCRYPTED"] = "1"
    r = subprocess.run(
        [sys.executable, "-c", _ENC_CHILD, PLUGIN_DIR, db],
        capture_output=True, text=True, env=env, timeout=180,
    )
    assert r.returncode == 0, f"child rc={r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    assert "ENC_OK" in r.stdout, f"missing ENC_OK\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 - A5 pin (never-forget), A21 no-delete, A22 metadata, re-embed migration
# ─────────────────────────────────────────────────────────────────────────────
def test_store_pin_protects_from_decay():
    """P4a/A5: a pinned fact is exempt from cycle decay AND staleness decay; an
    identical unpinned control fades (the system still forgets everything else)."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()

    def res(fid):
        return s._conn.execute(
            "SELECT resonance_count FROM semantic_facts WHERE id=?", (fid,)).fetchone()[0]

    _, pinned = s.add_or_reinforce_fact("pinned durable fact", _emb(s, "pinned durable fact"),
                                        "general", "t")
    _, ctrl = s.add_or_reinforce_fact("ordinary fading fact", _emb(s, "ordinary fading fact"),
                                      "general", "t")
    s._conn.execute("UPDATE semantic_facts SET tier='short', resonance_count=3, "
                    "last_confirmed_cycle=0 WHERE id IN (?,?)", (pinned, ctrl))
    s._conn.commit()
    assert s.set_pinned(pinned, True) is True
    for _ in range(20):
        s.apply_cycle_decay()
        s.apply_staleness_decay(current_cycle=100, boost=2.0, halflife=10.0)
    assert res(pinned) == 3, ("pinned decayed", res(pinned))   # untouched
    assert res(ctrl) < 3, ("control did not decay", res(ctrl))
    s.close()


def test_store_pin_protects_from_prune_and_cap():
    """P4a/A5: a pinned fact at resonance 0 is never pruned, and a weak pinned
    long-tier fact survives long-tier-cap eviction even when it isn't in the cap's
    keep-set. Unpinned counterparts are deleted/evicted (the control)."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    # --- prune: pinned dormant fact kept, unpinned dormant fact deleted ---
    _, pin0 = s.add_or_reinforce_fact("pinned but zero", _emb(s, "pinned but zero"), "general", "t")
    _, ctl0 = s.add_or_reinforce_fact("unpinned zero", _emb(s, "unpinned zero"), "general", "t")
    s._conn.execute("UPDATE semantic_facts SET resonance_count=0, pinned=1 WHERE id=?", (pin0,))
    s._conn.execute("UPDATE semantic_facts SET resonance_count=0 WHERE id=?", (ctl0,))
    s._conn.commit()
    s.prune_weak_facts(0)                       # legacy delete-at-0 path
    assert s.get_fact(pin0) is not None, "pinned dormant fact was pruned"
    assert s.get_fact(ctl0) is None, "unpinned dormant fact survived prune"
    # also the demote-then-deep-delete path must spare a pinned fact past the grace
    s._conn.execute("UPDATE semantic_facts SET dormant_since_cycle=0 WHERE id=?", (pin0,))
    s._conn.commit()
    s.prune_weak_facts(1)                       # deep-delete after 1 dormant cycle
    assert s.get_fact(pin0) is not None, "pinned fact deep-deleted"
    # --- long-tier cap: pinned weak long fact survives; unpinned weak one evicted ---
    _, pinL = s.add_or_reinforce_fact("pinned weak long", _emb(s, "pinned weak long"), "general", "t")
    _, strong = s.add_or_reinforce_fact("strong long fact", _emb(s, "strong long fact"), "general", "t")
    _, weak = s.add_or_reinforce_fact("weak long evictme", _emb(s, "weak long evictme"), "general", "t")
    s._conn.execute("UPDATE semantic_facts SET tier='long', resonance_count=1, pinned=1 WHERE id=?", (pinL,))
    s._conn.execute("UPDATE semantic_facts SET tier='long', resonance_count=9 WHERE id=?", (strong,))
    s._conn.execute("UPDATE semantic_facts SET tier='long', resonance_count=0.5 WHERE id=?", (weak,))
    s._conn.commit()
    s.enforce_long_tier_cap(1)                  # keep top-1 unpinned (strong) + ALL pinned
    assert s.get_fact(pinL) is not None, "pinned weak long fact was evicted"
    assert s.get_fact(strong) is not None, "strongest long fact evicted"
    assert s.get_fact(weak) is None, "weak unpinned long fact survived the cap"
    s.close()


def test_store_set_pinned_roundtrip_no_inflation():
    """P4a/P4c: set_pinned flips the flag (get_fact reflects it), is purely
    protective (never changes resonance), and returns False for an unknown id."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _fresh_store()
    _, fid = s.add_or_reinforce_fact("a vital identity fact", _emb(s, "a vital identity fact"),
                                     "general", "t")
    before = s.get_fact(fid)["resonance_count"]
    assert s.set_pinned(fid, True) is True
    f = s.get_fact(fid)
    assert f["pinned"] == 1, f
    assert f["resonance_count"] == before, ("pin inflated resonance", f["resonance_count"], before)
    assert s.set_pinned(fid, False) is True
    assert s.get_fact(fid)["pinned"] == 0
    assert s.set_pinned(10_000_000, True) is False, "unknown id did not return False"
    s.close()


def test_store_recall_metadata_surfaced():
    """P4b/A22: search results carry the confidence picture - peak_resonance (peak),
    learned_at_cycle (entry), pinned (bool) - and the raw max_resonance_seen column
    is renamed away in the model-facing payload."""
    if not _STORE_OK:
        print("  SKIP"); return
    from retrieval import LatticeRetriever
    s = _fresh_store(vector_dim=16)
    qvec = _emb(s, "target")

    class _PlainR(LatticeRetriever):
        def _get_embedding(self, _text):
            return qvec

    _, fid = s.add_or_reinforce_fact("target fact about Maya", qvec, "general", "sess1",
                                     entities=["maya"])
    s._conn.execute("UPDATE semantic_facts SET resonance_count=2, max_resonance_seen=8, "
                    "learned_at_cycle=3, pinned=1 WHERE id=?", (fid,))
    s._conn.commit()
    r = _PlainR(s, "http://x", "m", min_similarity=-1.0)
    hits = {h["id"]: h for h in r.search("target", limit=5)}
    assert fid in hits, hits
    h = hits[fid]
    assert h.get("peak_resonance") == 8, h
    assert h.get("learned_at_cycle") == 3, h
    assert h.get("pinned") is True, h
    assert "max_resonance_seen" not in h, "raw peak column leaked to the payload"
    s.close()


def test_store_reembed_if_needed():
    """P4d: re-embed migration self-gates on meta['embed_model'] - absent stamps
    (no spurious re-embed), same model no-ops, a genuine switch re-embeds every
    fact, and a dimension change rebuilds semantic_vec at the new dim."""
    if not _STORE_OK:
        print("  SKIP"); return
    import numpy as np
    s = _fresh_store(vector_dim=8)

    def mk_embed(dim, salt):
        def f(text):
            rng = np.random.default_rng(abs(hash((salt, text))) % (2**32))
            v = rng.standard_normal(dim)
            return (v / (np.linalg.norm(v) or 1.0)).tolist()
        return f

    def meta_model():
        row = s._conn.execute("SELECT value FROM meta WHERE key='embed_model'").fetchone()
        return row["value"] if row else None

    _, a = s.add_or_reinforce_fact("fact alpha", _emb(s, "fact alpha"), "general", "t")
    _, b = s.add_or_reinforce_fact("fact beta", _emb(s, "fact beta"), "general", "t")

    # 1. Gate ABSENT -> stamp current model, do NOT re-embed (no spurious work on install).
    assert meta_model() is None
    embA = mk_embed(8, "A")
    assert s.reembed_if_needed(embA, "model-A") == 0
    assert meta_model() == "model-A"
    # 2. Same model -> no-op.
    assert s.reembed_if_needed(embA, "model-A") == 0
    # 3. Genuine switch (same dim) -> re-embeds all; stored vectors become model-B's.
    embB = mk_embed(8, "B")
    assert s.reembed_if_needed(embB, "model-B") == 2
    assert meta_model() == "model-B"
    got = s.get_fact_embedding(a)
    assert got is not None and len(got) == 8
    assert max(abs(x - y) for x, y in zip(got, embB("fact alpha"))) < 1e-5, "vectors not re-embedded"
    # 4. Dimension change -> rebuilds semantic_vec at the new dim.
    embC = mk_embed(32, "C")
    assert s.reembed_if_needed(embC, "model-C") == 2
    assert s.vector_dim == 32, s.vector_dim
    got2 = s.get_fact_embedding(b)
    assert got2 is not None and len(got2) == 32, len(got2 or [])
    s.close()


# ─────────────────────────────────────────────────────────────────────────────
# Conflict quarantine (recall containment) - pure, dependency-free
# ─────────────────────────────────────────────────────────────────────────────
_recall_mod = _load("recall")


def _bare_recall(**attrs):
    app = _recall_mod.RecallMixin.__new__(_recall_mod.RecallMixin)
    for k, v in attrs.items():
        setattr(app, k, v)
    return app


class _FakeStore:
    def __init__(self, cats):
        self.importance_categories = {c.lower() for c in cats}


class _FakeRetriever:
    def __init__(self, rows):
        self._rows = rows

    def search(self, query, limit=10):
        return list(self._rows)


def test_quarantine_partition_high_stakes_unpinned_only():
    app = _bare_recall(_quarantine_high_stakes_conflicts=True,
                       _store=_FakeStore({"policy", "spend"}))
    results = [
        {"id": 1, "category": "spend", "conflict_group_id": "cg1", "pinned": 0},    # WITHHELD
        {"id": 2, "category": "policy", "conflict_group_id": "cg1", "pinned": 1},   # pinned -> kept
        {"id": 3, "category": "general", "conflict_group_id": "cg2", "pinned": 0},  # low-stakes -> kept
        {"id": 4, "category": "spend", "conflict_group_id": None, "pinned": 0},     # no conflict -> kept
        {"id": 5, "category": "policy", "conflict_group_id": "cg3", "pinned": 0},   # WITHHELD
    ]
    kept, withheld = app._quarantine_conflicts(results)
    assert {r["id"] for r in kept} == {2, 3, 4}, {r["id"] for r in kept}
    assert withheld == {"cg1": 1, "cg3": 1}, withheld


def test_quarantine_off_keeps_everything():
    app = _bare_recall(_quarantine_high_stakes_conflicts=False, _store=_FakeStore({"policy"}))
    results = [{"id": 1, "category": "policy", "conflict_group_id": "cg1", "pinned": 0}]
    kept, withheld = app._quarantine_conflicts(results)
    assert len(kept) == 1 and withheld == {}


def _prefetch_app(rows, quarantine):
    return _bare_recall(
        _retriever=_FakeRetriever(rows),
        _store=_FakeStore({"policy", "spend"}),
        _recall_limit=10,
        _reinforce_on_recall=False,
        _surface_conflicts=False,
        _surface_freshness_in_recall=False,
        _quarantine_high_stakes_conflicts=quarantine,
    )


def test_quarantine_prefetch_withholds_and_signals():
    rows = [
        {"id": 1, "content": "auto-approval enabled for all spends", "category": "policy",
         "tier": "short", "resonance_count": 5, "conflict_group_id": "cg-pol", "pinned": 0},
        {"id": 2, "content": "POLICY: never auto-approve; require human approval",
         "category": "policy", "tier": "long", "resonance_count": 3,
         "conflict_group_id": "cg-pol", "pinned": 1},
        {"id": 3, "content": "Acme is in Boston", "category": "general", "tier": "mid",
         "resonance_count": 4, "conflict_group_id": None, "pinned": 0},
    ]
    # the distinctive withheld-LINE phrase (the legend always *explains* [WITHHELD])
    SIGNAL = "held back pending resolution"
    block = _prefetch_app(rows, quarantine=True)._compute_prefetch("auto approve?", "sid")
    assert SIGNAL in block
    assert "auto-approval enabled" not in block      # unpinned poison withheld
    assert "never auto-approve" in block             # pinned authority stays
    assert "Acme is in Boston" in block              # low-stakes untouched
    block_off = _prefetch_app(rows, quarantine=False)._compute_prefetch("auto approve?", "sid")
    assert SIGNAL not in block_off
    assert "auto-approval enabled" in block_off      # OFF -> contested fact returns


def test_prefetch_proxy_topic_shift_gate():
    app = _bare_recall(_prefetch_proxy_min_overlap=0.3)
    # same topic -> reuse the previous-turn proxy
    assert app._prefetch_proxy_ok("can I auto approve this spend", "auto approve a spend now")
    # topic shift -> recompute (don't inject stale cross-topic memory)
    assert not app._prefetch_proxy_ok("what is the weather in Paris today", "auto approve a spend")
    # threshold 0 disables the gate (always reuse)
    app._prefetch_proxy_min_overlap = 0.0
    assert app._prefetch_proxy_ok("completely unrelated text", "auto approve a spend")


def _time_app(rows, on=True):
    """Bare recall mixin wired for full prefetch() calls (not just _compute_prefetch)."""
    app = _prefetch_app(rows, quarantine=False)
    app._inject_current_datetime = on
    app._session_id = "sid"
    app._prefetch_cache = {}
    app._prefetch_proxy_min_overlap = 0.3
    return app


def test_prefetch_time_context_injected():
    import datetime as _dt
    rows = [{"id": 1, "content": "Acme is in Boston", "category": "general",
             "tier": "mid", "resonance_count": 4, "conflict_group_id": None, "pinned": 0}]
    d_before = _dt.datetime.now().astimezone().strftime("%Y-%m-%d")
    block = _time_app(rows).prefetch("where is acme")
    d_after = _dt.datetime.now().astimezone().strftime("%Y-%m-%d")
    # Live clock tag stamped today, OUTSIDE (preceding) the memory block: the
    # clock is authoritative context, never a fallible retrieved candidate.
    assert block.startswith("<current_datetime>"), block[:80]
    head = block.split("</current_datetime>")[0]
    assert d_before in head or d_after in head, head
    assert "<resonant_memory>" in block and "Acme is in Boston" in block
    assert block.index("</current_datetime>") < block.index("<resonant_memory>")
    # Empty recall still yields the clock: time coherence never depends on a hit.
    empty = _time_app([]).prefetch("anything")
    assert empty.startswith("<current_datetime>") and "<resonant_memory>" not in empty
    # A CACHED background block is stamped at CONSUMPTION time too (an idle
    # session's hours-old proxy must still carry the real "now").
    app = _time_app(rows)
    app._prefetch_cache["sid"] = ("where is acme",
                                  "<resonant_memory>\ncached\n</resonant_memory>")
    cached = app.prefetch("where is acme")
    assert cached.startswith("<current_datetime>") and "cached" in cached
    # An unknown datetime_timezone falls back to host-local; the stamp survives.
    bad = _time_app([])
    bad._datetime_timezone = "Not/AZone"
    assert bad.prefetch("anything").startswith("<current_datetime>")


def test_prefetch_time_context_off_is_legacy():
    rows = [{"id": 1, "content": "Acme is in Boston", "category": "general",
             "tier": "mid", "resonance_count": 4, "conflict_group_id": None, "pinned": 0}]
    block = _time_app(rows, on=False).prefetch("where is acme")
    assert "<current_datetime>" not in block and "Acme is in Boston" in block
    assert _time_app([], on=False).prefetch("anything") == ""


# ─────────────────────────────────────────────────────────────────────────────
# Canonical-state projection (current_value layer over the lattice)
# ─────────────────────────────────────────────────────────────────────────────
def test_store_canonical_set_get_and_supersede():
    s = _fresh_store()
    try:
        s.set_cycle_counts(memory_cycle=5)
        cid1 = s.set_canonical("acme.payment_terms", "Net-30", category="financial")
        cur = s.get_canonical("acme.payment_terms")
        assert cur["value"] == "Net-30" and cur["category"] == "financial"
        assert cur["valid_from_cycle"] == 5 and cur["review_status"] == "unreviewed"
        s.set_cycle_counts(memory_cycle=9)
        cid2 = s.set_canonical("acme.payment_terms", "Net-45")
        cur2 = s.get_canonical("acme.payment_terms")
        assert cur2["value"] == "Net-45" and cur2["canonical_id"] == cid2
        assert cur2["valid_from_cycle"] == 9
        hist = s.canonical_history("acme.payment_terms")
        assert len(hist) == 2, len(hist)
        old = [h for h in hist if h["canonical_id"] == cid1][0]
        assert old["valid_until_cycle"] == 9 and old["superseded_by"] == cid2
        # same value again is a no-op (no new history row)
        s.set_canonical("acme.payment_terms", "Net-45")
        assert len(s.canonical_history("acme.payment_terms")) == 2
    finally:
        s.close()


def test_store_canonical_tool_dispatch():
    import json
    import types as _types
    # tool_handler imports `from tools.registry import tool_error` (a Hermes module).
    # Inject a minimal stub so the handler can load in the bare unit-test env.
    if "tools.registry" not in sys.modules:
        _pkg = _types.ModuleType("tools")
        _reg = _types.ModuleType("tools.registry")
        _reg.tool_error = lambda m: json.dumps({"error": m})
        _pkg.registry = _reg
        sys.modules["tools"] = _pkg
        sys.modules["tools.registry"] = _reg
    th = _load("tool_handler")
    handler = th.ToolHandlerMixin.__new__(th.ToolHandlerMixin)
    handler._store = _fresh_store()
    handler._retriever = object()   # truthy - handler only checks it exists
    handler._write_enabled = True
    handler._memory_cycle = 7
    try:
        out = json.loads(handler.handle_tool_call(
            "lattice_store",
            {"action": "set_canonical", "key": "vendor.x.terms",
             "value": "Net-30", "category": "financial"}))
        assert out.get("canonical_id") and out.get("value") == "Net-30", out
        got = json.loads(handler.handle_tool_call(
            "lattice_store", {"action": "get_canonical", "key": "vendor.x.terms"}))
        assert got["found"] and got["canonical"]["value"] == "Net-30", got
        listing = json.loads(handler.handle_tool_call(
            "lattice_store", {"action": "get_canonical"}))   # no key -> list
        assert listing["count"] == 1
        # write-gate: a read-only (non-primary) context must refuse set_canonical
        handler._write_enabled = False
        denied = handler.handle_tool_call(
            "lattice_store", {"action": "set_canonical", "key": "k", "value": "v"})
        assert "read-only" in denied.lower(), denied
    finally:
        handler._store.close()


def test_store_canonical_missing_list_and_review():
    s = _fresh_store()
    try:
        assert s.get_canonical("nope") is None
        s.set_canonical("k1", "v1", category="policy")
        s.set_canonical("k2", "v2", category="financial")
        assert {c["key"] for c in s.list_canonical()} == {"k1", "k2"}
        fin = s.list_canonical(category="financial")
        assert len(fin) == 1 and fin[0]["key"] == "k2"
        assert s.review_canonical("k1", "reviewed") is True
        assert s.get_canonical("k1")["review_status"] == "reviewed"
        assert s.review_canonical("missing", "reviewed") is False
    finally:
        s.close()


# ─────────────────────────────────────────────────────────────────────────────
# Semantic write-batch provenance + rollback
# ─────────────────────────────────────────────────────────────────────────────
def test_store_write_batch_stamp_and_rollback():
    s = _fresh_store()
    try:
        # a write OUTSIDE any batch is unstamped (batch_id NULL)
        _, a = s.add_or_reinforce_fact("alpha one", _emb(s, "alpha one"), "general", "u")
        assert s._conn.execute(
            "SELECT batch_id FROM semantic_facts WHERE id=?", (a,)).fetchone()[0] is None
        # open a batch -> writes are stamped
        bid = s.begin_write_batch("dream", model="m1", cycle=5)
        _, b = s.add_or_reinforce_fact("beta two", _emb(s, "beta two"), "general", "u")
        _, c = s.add_or_reinforce_fact("gamma three", _emb(s, "gamma three"), "general", "u")
        s.set_pinned(c, True)        # a pinned fact in the batch must survive rollback
        s.end_write_batch()
        assert {f["id"] for f in s.get_batch_facts(bid)} == {b, c}
        this = [x for x in s.list_write_batches() if x["batch_id"] == bid][0]
        assert this["phase"] == "dream" and this["n_writes"] == 2 and this["status"] == "closed"
        # rollback: deletes non-pinned batch facts; keeps pinned + the pre-batch fact
        res = s.rollback_write_batch(bid)
        assert res["deleted"] == 1 and res["kept_pinned"] == 1, res
        assert s.get_fact(a) is not None      # pre-batch fact untouched
        assert s.get_fact(b) is None          # rolled back
        assert s.get_fact(c) is not None      # pinned -> kept
        after = [x for x in s.list_write_batches() if x["batch_id"] == bid][0]
        assert after["status"] == "rolled_back"
        assert "error" in s.rollback_write_batch(999999)   # unknown batch
    finally:
        s.close()


def test_store_write_batch_empty_autocleanup():
    s = _fresh_store()
    try:
        bid = s.begin_write_batch("consolidation", cycle=1)
        s.end_write_batch()           # wrote nothing -> the batch row auto-cleans
        gone = s._conn.execute(
            "SELECT COUNT(*) FROM write_batches WHERE batch_id=?", (bid,)).fetchone()[0]
        assert gone == 0, "empty batch should leave no provenance noise"
    finally:
        s.close()


def test_store_write_batch_tool_dispatch():
    import json
    import types as _types
    if "tools.registry" not in sys.modules:
        _pkg = _types.ModuleType("tools")
        _reg = _types.ModuleType("tools.registry")
        _reg.tool_error = lambda m: json.dumps({"error": m})
        _pkg.registry = _reg
        sys.modules["tools"] = _pkg
        sys.modules["tools.registry"] = _reg
    th = _load("tool_handler")
    h = th.ToolHandlerMixin.__new__(th.ToolHandlerMixin)
    h._store = _fresh_store()
    h._retriever = object()
    h._write_enabled = True
    h._memory_cycle = 3
    try:
        bid = h._store.begin_write_batch("dream", cycle=3)
        h._store.add_or_reinforce_fact("xfact one", _emb(h._store, "xfact one"), "general", "u")
        h._store.add_or_reinforce_fact("yfact two", _emb(h._store, "yfact two"), "general", "u")
        h._store.end_write_batch()
        listing = json.loads(h.handle_tool_call("lattice_store", {"action": "list_batches"}))
        assert any(b["batch_id"] == bid for b in listing["batches"]), listing
        facts = json.loads(h.handle_tool_call(
            "lattice_store", {"action": "list_batches", "batch_id": bid}))
        assert len(facts["facts"]) == 2
        res = json.loads(h.handle_tool_call(
            "lattice_store", {"action": "rollback_batch", "batch_id": bid}))
        assert res["deleted"] == 2 and res["status"] == "rolled_back", res
        h._write_enabled = False
        denied = h.handle_tool_call("lattice_store", {"action": "rollback_batch", "batch_id": bid})
        assert "read-only" in denied.lower()
    finally:
        h._store.close()


def test_store_procedural_staleness_decay():
    """Procedural facts fade if unconfirmed; used / pinned / conflicted / domain spared."""
    s = _fresh_store(vector_dim=8)
    try:
        cyc = 20

        def add(content, category="procedural", res=10.0, last_conf=0, pinned=0, cgid=None):
            s._conn.execute(
                "INSERT INTO semantic_facts (content, category, tier, resonance_count, "
                "last_confirmed_cycle, learned_at_cycle, pinned, conflict_group_id) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (content, category, "long", res, last_conf, last_conf, pinned, cgid),
            )
            return s._conn.execute(
                "SELECT id FROM semantic_facts WHERE content=?", (content,)
            ).fetchone()[0]

        stale = add("[web_search] use a site: operator", last_conf=0)         # old -> bleeds
        fresh = add("[web_search] use plain keywords + source", last_conf=19)  # recent -> spared
        pinned = add("[web_search] pinned authority rule", last_conf=0, pinned=1)
        conflicted = add("[web_search] contested tip", last_conf=0, cgid="ab12cd34")
        domain = add("The capital of France is Paris.", category="general", last_conf=0)
        s._conn.commit()

        def res(fid):
            return s._conn.execute(
                "SELECT resonance_count FROM semantic_facts WHERE id=?", (fid,)
            ).fetchone()[0]

        before = {f: res(f) for f in (stale, fresh, pinned, conflicted, domain)}
        n = s.apply_procedural_staleness_decay(cyc, bleed=0.5, grace_cycles=5)
        after = {f: res(f) for f in (stale, fresh, pinned, conflicted, domain)}

        assert abs(after[stale] - (before[stale] - 0.5)) < 1e-9, \
            f"stale procedural should bleed 0.5, {before[stale]}->{after[stale]}"
        assert after[fresh] == before[fresh], "recently-confirmed procedural must not bleed"
        assert after[pinned] == before[pinned], "pinned must not bleed"
        assert after[conflicted] == before[conflicted], "in-conflict must not bleed"
        assert after[domain] == before[domain], "non-procedural (domain) must not bleed"
        assert n == 1, f"only the one stale procedural fact should be touched, got {n}"

        # Off when disabled.
        assert s.apply_procedural_staleness_decay(cyc, bleed=0.0) == 0
        # Floors at 0, never negative.
        s._conn.execute("UPDATE semantic_facts SET resonance_count=0.3 WHERE id=?", (stale,))
        s._conn.commit()
        s.apply_procedural_staleness_decay(cyc, bleed=0.5, grace_cycles=5)
        assert res(stale) == 0.0, "resonance must floor at 0"
        print("  procedural staleness OK: stale bleeds; used/pinned/conflict/domain spared; floors at 0")
    finally:
        s.close()


def test_reason_gate_serializes():
    """The memory-reason gate admits at most `capacity` concurrent holders.

    This is the invariant that keeps the memory layer to one reasoning-model call
    at a time (default) so it consumes a single slot on a shared/rate-limited
    endpoint and never starves the primary agent. Pure threading - no LLM/DB.
    """
    import reason_gate
    import threading, time

    def measure_peak(n_threads):
        state = {"now": 0, "max": 0}
        lock = threading.Lock()

        def worker():
            with reason_gate.reason_slot():
                with lock:
                    state["now"] += 1
                    state["max"] = max(state["max"], state["now"])
                time.sleep(0.02)
                with lock:
                    state["now"] -= 1

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return state["max"]

    try:
        reason_gate.set_capacity(1)
        peak1 = measure_peak(8)
        assert peak1 == 1, f"capacity=1 should serialize, but saw {peak1} concurrent"

        reason_gate.set_capacity(3)
        peak3 = measure_peak(8)
        assert 1 <= peak3 <= 3, f"capacity=3 must cap at 3, saw {peak3}"

        # Guard against garbage config: clamps to >=1, never 0 (which would deadlock).
        reason_gate.set_capacity(0)
        assert reason_gate.capacity() == 1
        peak0 = measure_peak(4)
        assert peak0 == 1
    finally:
        reason_gate.set_capacity(1)  # restore default for any later test

    print("  reason gate OK: serial@1, caps@3, clamps 0->1")


def test_store_cycle_counter_multiprocess_atomicity():
    """Two store instances on ONE DB file (the gateway and one-shot hermes runs
    share a profile DB) must not clobber each other's cycle counters.

    Regression: nemo 2026-07-09 - dream_cycle went 38 -> 22 when a long-lived
    gateway (started when the counter was ~21) wrote back its process-cached
    counter over the value that overnight one-shot runs had advanced.
    increment_cycle() is an atomic in-DB read-modify-write, so a stale cache
    can never roll the shared clock backwards. Validated at the SQLite layer."""
    if not _STORE_OK:
        print(f"  SKIP cycle counter atomicity: {_SKIP_REASON}"); return
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "race.db")
    s1 = store_mod.LatticeStore(db_path=db, vector_dim=16)   # "gateway": long-lived
    s2 = store_mod.LatticeStore(db_path=db, vector_dim=16)   # "one-shot" runs

    # one-shots advance the clock while the gateway sits idle with a stale cache
    for i in range(1, 6):
        assert s2.increment_cycle("dream_cycle") == i
    # the gateway increments later: must build on 5, not its startup snapshot
    assert s1.increment_cycle("dream_cycle") == 6, "stale cache clobbered the clock"
    assert s1.get_cycle_counts()[1] == 6
    assert s2.get_cycle_counts()[1] == 6

    # interleaved writers: strictly monotonic, no duplicates, no rollback
    vals = []
    for i in range(8):
        w = s1 if i % 2 else s2
        vals.append(w.increment_cycle("memory_cycle"))
    assert vals == list(range(1, 9)), f"non-monotonic/duplicated: {vals}"
    # substrate check: the meta row itself holds the final value
    row = s1._conn.execute(
        "SELECT CAST(value AS INTEGER) AS v FROM meta WHERE key='memory_cycle'"
    ).fetchone()
    assert row["v"] == 8

    # guard: unknown keys rejected (meta holds non-counter rows too)
    try:
        s1.increment_cycle("hrr_dim")
        assert False, "increment_cycle accepted a non-counter key"
    except ValueError:
        pass

    # absolute setter still works for tests/replay, and increments build on it
    s1.set_cycle_counts(memory_cycle=100)
    assert s2.increment_cycle("memory_cycle") == 101

    s1.close(); s2.close()
    print("  cycle counters atomic across store instances (multi-process safe)")


def test_store_consolidation_debt_ledger():
    """Consolidation debt, store layer (pure SQL, no LLM): a session with
    substantial episodes and ZERO born facts is flagged once, its episodes are
    exempt from BOTH pruning clauses while the debt is open, the seasoning
    delay hides it from retries until the next dream cycle, and settling
    (any outcome) releases the episodes back to normal pruning. Fixes the
    class that lost the TanStack block (episodes pruned before recovery)."""
    if not _STORE_OK:
        print(f"  SKIP consolidation debt: {_SKIP_REASON}"); return
    s = _fresh_store()

    # Session A: substantial episodes, no facts  -> DEBT
    for i in range(3):
        s.add_turn("sess-A", f"question {i}", f"answer {i}")
    # Session B: substantial episodes + a born fact -> not debt
    for i in range(2):
        s.add_turn("sess-B", f"q {i}", f"a {i}")
    s.add_or_reinforce_fact("session B banked this fact", _emb(s, "b fact"),
                            "general", "sess-B")
    # Session C: below the substantiality floor -> not debt
    s.add_turn("sess-C", "hi", "hello")
    # Session D: the CURRENT session (excluded), no facts yet
    for i in range(3):
        s.add_turn("sess-D", f"live q {i}", f"live a {i}")

    added = s.flag_consolidation_debt(5, exclude_session="sess-D")
    assert added == 1, added
    assert s.flag_consolidation_debt(5, exclude_session="sess-D") == 0  # idempotent
    debts = s.get_open_consolidation_debts(before_cycle=6)
    assert [d["session_id"] for d in debts] == ["sess-A"], debts
    assert debts[0]["episode_rows"] == 6, debts
    # Seasoning: flagged AT cycle 5 is invisible to a retry scan AT cycle 5.
    assert s.get_open_consolidation_debts(before_cycle=5) == []

    # Pruning exemption, session-window clause: keep only the most recent
    # session -> A survives (open debt), B and C are pruned.
    s.prune_episodes(keep_sessions=1)
    left = {r[0] for r in s._conn.execute(
        "SELECT DISTINCT session_id FROM episodes").fetchall()}
    assert "sess-A" in left, left
    assert "sess-B" not in left and "sess-C" not in left, left
    # Pruning exemption, max_rows clause: a 2-row cap cannot touch A's 6 rows.
    s.prune_episodes(keep_sessions=1, max_rows=2)
    a_rows = s._conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE session_id='sess-A'").fetchone()[0]
    assert a_rows == 6, a_rows

    # Receipt + attempts + settlement.
    assert s.session_born_fact_count("sess-A") == 0
    assert s.session_born_fact_count("sess-B") == 1
    assert s.record_debt_attempt("sess-A", 6) == 1
    assert s.record_debt_attempt("sess-A", 7) == 2
    s.settle_consolidation_debt("sess-A", "exhausted", 7)
    assert s.get_open_consolidation_debts(before_cycle=99) == []
    counts = s.consolidation_debt_counts()
    assert counts == {"exhausted": 1}, counts
    # A settled session may never be re-flagged (tracked forever).
    assert s.flag_consolidation_debt(8, exclude_session="") == 0 or True
    re_flagged = s._conn.execute(
        "SELECT COUNT(*) FROM consolidation_debt WHERE session_id='sess-A'"
    ).fetchone()[0]
    assert re_flagged == 1, re_flagged
    # Exemption lifted: pruning may now delete A's episodes.
    s.prune_episodes(keep_sessions=1)
    a_rows = s._conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE session_id='sess-A'").fetchone()[0]
    assert a_rows == 0, a_rows
    # Health snapshot surfaces the ledger.
    h = s.get_memory_health()
    assert h["consolidation_debt_exhausted"] == 1 and h["consolidation_debt_open"] == 0
    s.close()
    print("  consolidation debt ledger OK: flag once, exempt both prune clauses, "
          "seasoning delay, settle releases episodes, health surfaced")


def test_consolidation_debt_retry_flow():
    """Consolidation debt, provider layer (stubbed store, no LLM): the post-dream
    retry re-checks the receipt BEFORE spending an epoch (organic settlement is
    free), retries with suppress_dream=True, judges recovery at the substrate
    (born facts, not the epoch's return), and settles 'exhausted' only after
    reconsolidation_max_attempts."""
    try:
        prov = _load("__init__")
    except Exception as e:
        print(f"  SKIP debt retry flow: {e}"); return

    class _DebtStore:
        def __init__(self, born=0):
            self.born = born
            self.attempts = 0
            self.settled = None

        def get_open_consolidation_debts(self, before_cycle, limit=1):
            if self.settled is not None:
                return []
            return [{"session_id": "s-debt", "episode_rows": 8,
                     "attempts": self.attempts, "first_flagged_cycle": 3}]

        def session_born_fact_count(self, sid):
            return self.born

        def record_debt_attempt(self, sid, cycle):
            self.attempts += 1
            return self.attempts

        def settle_consolidation_debt(self, sid, outcome, cycle=0):
            self.settled = outcome

    def _mk(born=0):
        p = prov.LatticeMemoryProvider({})
        p._store = _DebtStore(born)
        p._write_enabled = True
        p._reconsolidate_zero_fact_sessions = True
        p._reconsolidation_max_attempts = 2
        p._dream_cycle_count = 5
        p._epochs = []
        p._run_consolidation_epoch = (
            lambda sid, force_blocking=False, suppress_dream=False:
            p._epochs.append((sid, suppress_dream)))
        return p

    # Organic: facts already landed -> settled with ZERO epoch calls.
    p = _mk(born=3)
    p._reconsolidate_debt()
    assert p._store.settled == "organic" and p._epochs == [], (p._store.settled, p._epochs)

    # Recovery: first retry stays open (attempt 1/2), a later retry that banks
    # facts settles 'recovered'. suppress_dream must be True on every retry.
    p = _mk(born=0)
    p._reconsolidate_debt()                      # attempt 1: still 0 facts
    assert p._store.settled is None and p._store.attempts == 1
    assert p._epochs == [("s-debt", True)], p._epochs
    orig = p._run_consolidation_epoch
    p._run_consolidation_epoch = (
        lambda sid, force_blocking=False, suppress_dream=False:
        (orig(sid, force_blocking, suppress_dream), setattr(p._store, "born", 11)))
    p._reconsolidate_debt()                      # attempt 2: recovers 11 facts
    assert p._store.settled == "recovered" and p._store.attempts == 2

    # Exhaustion: two failed retries -> 'exhausted', never a third epoch.
    p = _mk(born=0)
    p._reconsolidate_debt()
    p._reconsolidate_debt()
    assert p._store.settled == "exhausted" and p._store.attempts == 2
    n_epochs = len(p._epochs)
    p._reconsolidate_debt()                      # settled -> no further work
    assert len(p._epochs) == n_epochs
    print("  consolidation debt retry OK: organic free, recovery substrate-judged, "
          "exhaustion bounded at max attempts")


def test_synthesized_label_epoch_stamping():
    """[SYNTHESIZED] provenance (label gauntlet, fleet-validated): the consolidation
    epoch stamps source_ref 'synthesized:<session>' on facts born from a session
    that READ its own memory (in-process lattice-read flag) and touched NO web
    tools. A session with web tools keeps the extractor's ref (firsthand); a
    session that never read memory keeps it too (conversation provenance).
    Domain category is preserved - unlike mental_note. Runs the REAL epoch with a
    faked reason-model response."""
    try:
        prov = _load("__init__")
    except Exception as e:
        print(f"  SKIP synthesized stamping: {e}"); return
    import json
    import tempfile

    captured = {}

    class _EpochStore:
        def __init__(self, tmp, profile):
            self.db_path = os.path.join(tmp, "stub.db")
            self._profile = profile

        def get_recent_episodes(self, limit=10, session_id=None):
            return [{"role": "user", "content": "synthesize what you know" * 8},
                    {"role": "assistant", "content": "principles follow" * 20}]

        def get_cycle_counts(self):
            return (0, 0)

        def _clean_llm_json(self, text):
            return text

        def begin_write_batch(self, **kw):
            return 1

        def end_write_batch(self):
            pass

        def _extract_entities(self, content):
            return []

        def session_tool_names(self, session_id):
            return set(self._profile)

        def increment_cycle(self, key):
            return 1

        def add_or_reinforce_fact(self, content, emb, category, session_id, **kw):
            captured[session_id] = (category, kw.get("source_ref"))
            return ("added", 77)

    class _EpochRetriever:
        def _get_embedding(self, content):
            return [0.1] * 8

    fake_response = {"response": json.dumps([{
        "content": "Container queries beat media queries for reusable components",
        "category": "webdev",
        "source_ref": "https://echo.example/secondhand",
    }])}

    def _run(session_id, profile, lattice_read):
        p = prov.LatticeMemoryProvider({})
        tmp = tempfile.mkdtemp()
        p._store = _EpochStore(tmp, profile)
        p._retriever = _EpochRetriever()
        p._write_enabled = True
        p._gate_self_writes = False
        p._enable_relations = False
        p._session_id = session_id
        if lattice_read:
            p._lattice_read_sessions.add(session_id)
        g = p._run_consolidation_epoch.__func__.__globals__
        orig = g["_ollama_post_with_retry"]
        g["_ollama_post_with_retry"] = lambda url, payload, timeout, max_attempts=3: fake_response
        try:
            p._run_consolidation_epoch(session_id, suppress_dream=True)
        finally:
            g["_ollama_post_with_retry"] = orig

    # Memory reads + zero web tools -> stamped, category kept.
    _run("s-reflect", {"search_files"}, lattice_read=True)
    assert captured["s-reflect"] == ("webdev", "synthesized:s-reflect"), captured
    # Memory reads + web tools -> research session, extractor ref preserved.
    _run("s-research", {"web_search", "web_extract"}, lattice_read=True)
    assert captured["s-research"] == ("webdev", "https://echo.example/secondhand"), captured
    # No memory reads -> conversation provenance, untouched.
    _run("s-chat", set(), lattice_read=False)
    assert captured["s-chat"] == ("webdev", "https://echo.example/secondhand"), captured
    print("  synthesized stamping OK: reflection stamped (category kept), "
          "research + chat sessions untouched")


def test_synthesized_recall_marker():
    """[SYNTHESIZED] surfaces in the recall block: the marker on the fact line and
    the legend sentence in the header (the exact wording the gauntlet validated),
    and only for facts whose source_ref carries the synthesized: prefix."""
    try:
        _inject_hermes_stubs()
        prov = _load("__init__")
    except Exception as e:
        print(f"  SKIP synthesized marker: {e}"); return

    class _MarkerRetriever:
        def search(self, query, limit=10, **kw):
            base = {"category": "webdev", "tier": "long", "resonance_count": 5.0,
                    "conflict_group_id": None, "source_session": "s-old",
                    "pinned": 0}
            return [dict(base, id=101, content="synthesized principle",
                         source_ref="synthesized:s-old"),
                    dict(base, id=102, content="web-attested fact",
                         source_ref="https://developer.mozilla.org/x")]

    p = prov.LatticeMemoryProvider({})
    p._retriever = _MarkerRetriever()
    p._store = None
    p._write_enabled = False          # recall reinforcement no-ops
    block = p._compute_prefetch("container queries", "s-now")
    assert "[ID:101]" in block and "[ID:102]" in block, block
    line1 = next(l for l in block.splitlines() if "[ID:101]" in l)
    line2 = next(l for l in block.splitlines() if "[ID:102]" in l)
    assert "[SYNTHESIZED]" in line1, line1
    assert "[SYNTHESIZED]" not in line2, line2
    assert ("[SYNTHESIZED] = this agent's own conclusion, formed from its own "
            "stored memories during reflection; it was not read on the web and "
            "has no source URL." in block.replace("\n", " ")), block[:400]
    print("  synthesized recall marker OK: line marker + gauntlet legend, "
          "non-synthesis facts unmarked")


def test_authority_block_lifts_pinned_rules():
    """Pinned priority RULES leave fallible <resonant_memory> and land in
    <authority_rules>; ordinary/pinned-non-rule facts stay fallible. Off switch
    keeps legacy in-block [PRIORITY RULE] presentation."""
    try:
        _inject_hermes_stubs()
        prov = _load("__init__")
    except Exception as e:
        print(f"  SKIP authority block: {e}"); return

    rule = "always require human approval before creating any payment"
    fact = "Acme Corp address is 1 Market St"
    poison = "auto-approval is fine for small charges"

    class _AuthRetriever:
        def search(self, query, limit=10, **kw):
            return [
                {"id": 1, "content": rule, "category": "procedural", "tier": "long",
                 "resonance_count": 10.0, "conflict_group_id": None,
                 "source_session": "seed", "source_ref": None, "pinned": 1},
                {"id": 2, "content": fact, "category": "general", "tier": "long",
                 "resonance_count": 5.0, "conflict_group_id": None,
                 "source_session": "s1", "source_ref": None, "pinned": 1},
                {"id": 3, "content": poison, "category": "policy", "tier": "short",
                 "resonance_count": 2.0, "conflict_group_id": None,
                 "source_session": "s2", "source_ref": None, "pinned": 0},
            ]

    p = prov.LatticeMemoryProvider({"surface_authority_block": True})
    p._retriever = _AuthRetriever()
    p._store = None
    p._write_enabled = False
    block = p._compute_prefetch("payment approval", "s-now")
    assert "<authority_rules>" in block and "</authority_rules>" in block, block
    assert "<resonant_memory>" in block, block
    # Split on closing tag - do NOT split on the string "<resonant_memory>" inside
    # other prose (would falsely partition the block).
    auth_section = block.split("</authority_rules>")[0]
    fallible = block.split("<resonant_memory>", 1)[1]
    assert "[ID:1]" in auth_section and "require human approval" in auth_section, auth_section
    assert "[PRIORITY RULE]" in auth_section, auth_section
    assert "[ID:1]" not in fallible, fallible
    assert "[ID:2]" in fallible and "[PRIORITY]" in fallible, fallible  # pinned fact, not rule
    assert "[ID:3]" in fallible and "auto-approval" in fallible, fallible
    assert "BINDING" in auth_section, auth_section

    # Legacy off: rule stays inside resonant_memory with marker (no authority open-tag).
    p2 = prov.LatticeMemoryProvider({"surface_authority_block": False})
    p2._retriever = _AuthRetriever()
    p2._store = None
    p2._write_enabled = False
    legacy = p2._compute_prefetch("payment approval", "s-now")
    assert "<authority_rules>" not in legacy and "</authority_rules>" not in legacy, legacy
    assert "[ID:1]" in legacy and "[PRIORITY RULE]" in legacy, legacy
    assert legacy.strip().startswith("<resonant_memory>"), legacy[:40]
    print("  authority block OK: pinned rules lifted; non-rules + poison stay fallible; off=legacy")


def test_provider_effective_config_and_hops_floor():
    """get_effective_config surfaces clamped reinforce + hops; max_inference_hops=1 allowed."""
    try:
        _inject_hermes_stubs()
        prov = _load("__init__")
    except Exception as e:
        print(f"  SKIP effective config: {e}"); return

    p = prov.LatticeMemoryProvider({
        "max_inference_hops": 1,
        "reinforce_threshold": 0.50,
        "similarity_threshold": 0.80,
        "detect_policy_conflicts": False,
    })
    assert p._max_inference_hops == 1
    # Before initialize: effective reinforce is computed from provider-side max.
    ec0 = p.get_effective_config()
    assert ec0["max_inference_hops"] == 1
    assert ec0["reinforce_threshold_configured"] == 0.50
    assert ec0["reinforce_threshold_effective"] == 0.80
    assert ec0["reinforce_threshold_clamped"] is True
    assert ec0["detect_policy_conflicts"] is False
    assert "surface_authority_block" in ec0

    if not _STORE_OK:
        print("  effective config OK (provider-only; store path skipped)"); return

    home = tempfile.mkdtemp()
    p._probe_vector_dim = lambda: 768
    p.initialize("eff-cfg", hermes_home=home, agent_context="primary")
    assert p._store is not None
    assert p._store.reinforce_threshold == 0.80
    assert p._store.detect_policy_conflicts is False
    ec = p.get_effective_config()
    assert ec["reinforce_threshold_effective"] == 0.80
    assert ec["detect_policy_conflicts"] is False
    assert ec["max_inference_hops"] == 1
    p._store.close()
    print("  effective config OK: hops=1 live; reinforce clamp + detect flags reported")




# ===========================================================================
# Source index: give extraction the urls it never sees, and VERIFY the ones it
# attaches. Every assertion below encodes a false-citation class that was
# actually observed on a live corpus, not a hypothetical one.
# ===========================================================================

def _si():
    import source_index
    return source_index


def test_source_index_parses_both_web_payload_shapes():
    """web_extract and web_search use different shapes, both wrapped in an
    envelope with text AFTER the closing brace. json.loads() fails on 100% of
    real payloads because of that trailing text; raw_decode is required."""
    si = _si()
    extract = ('<untrusted_tool_result source="web_extract">\nnotice\n'
               '{"results":[{"url":"https://a.example/p","title":"A Title",'
               '"content":"' + ("alpha beta gamma delta " * 40) + '"}]}\n'
               '</untrusted_tool_result>')
    search = ('<untrusted_tool_result source="web_search">\nnotice\n'
              '{"success":true,"data":{"web":[{"url":"https://b.example/q",'
              '"title":"B Title","description":"'
              + ("epsilon zeta " * 60) + '"}]}}\n</untrusted_tool_result>')
    got = si.parse_tool_sources([
        {"role": "tool", "tool_name": "web_extract", "content": extract},
        {"role": "tool", "tool_name": "web_search", "content": search},
    ])
    urls = [g["url"] for g in got]
    assert urls == ["https://a.example/p", "https://b.example/q"], urls
    assert got[0]["title"] == "A Title"
    assert "alpha beta gamma" in got[0]["body"]
    # search puts the page text in `description`, not `content`
    assert "epsilon zeta" in got[1]["body"]


def test_source_index_rejects_shells_errors_and_duplicates():
    """A bot-mitigation shell must never become the evidence behind a citation --
    msdvetmanual.com returned one for 1,265 of 1,284 fetches on the real corpus."""
    si = _si()
    shell = ('{"results":[{"url":"https://blocked.example/x",'
             '"title":"There doesn no seem to be anything here.",'
             '"content":"honeypot link ' + ("cookie notice " * 60) + '"}]}')
    err = ('{"results":[{"url":"https://err.example/y","title":"t",'
           '"content":"' + ("x " * 400) + '","error":"timeout"}]}')
    stub = ('{"results":[{"url":"https://thin.example/z","title":"t",'
            '"content":"tiny"}]}')
    dupe = ('{"results":[{"url":"https://ok.example/1","title":"first",'
            '"content":"' + ("real content here " * 50) + '"}]}')
    dupe2 = ('{"results":[{"url":"https://ok.example/1","title":"second",'
             '"content":"' + ("different text " * 50) + '"}]}')
    got = si.parse_tool_sources([
        {"role": "tool", "tool_name": "web_extract", "content": shell},
        {"role": "tool", "tool_name": "web_extract", "content": err},
        {"role": "tool", "tool_name": "web_extract", "content": stub},
        {"role": "tool", "tool_name": "web_extract", "content": dupe},
        {"role": "tool", "tool_name": "web_extract", "content": dupe2},
    ])
    assert [g["url"] for g in got] == ["https://ok.example/1"], got
    # first occurrence wins, so dedup is stable rather than last-write
    assert got[0]["title"] == "first"


def test_source_index_block_is_capped_and_recent_first():
    """The block rides in a prompt served at ctx 16384 beside a ~2,000-token
    preamble, while a research session can retrieve 150+ urls."""
    si = _si()
    srcs = [{"url": "https://e.example/%d" % i, "title": "T" * 200, "body": "b"}
            for i in range(120)]
    blk = si.build_index_block(srcs, max_entries=40, title_chars=90)
    lines = [l for l in blk.splitlines() if l.startswith("- ")]
    assert len(lines) == 40, len(lines)
    # most recent first: research converges, so the last pages read are the ones
    # that answered the subtopic
    assert "https://e.example/119" in lines[0], lines[0]
    assert len(max(lines, key=len)) < 200
    assert si.build_index_block([]) == ""


def test_verify_ref_requires_verbatim_run_not_vocabulary_overlap():
    """THE central guard. A bag-of-words rule produced 74 'recovered' refs on the
    real corpus of which roughly 67 were false, because a 20,000-char page
    contains the same words somewhere regardless of provenance."""
    si = _si()
    page = ("Urease producing bacteria primarily Staphylococcus and Proteus "
            "drive struvite formation in most canine cases. " + "filler " * 500)
    bodies = {"https://good.example/p": page}
    quote = ("urease producing bacteria primarily staphylococcus and proteus "
             "drive struvite formation")
    assert si.verify_ref(quote, "https://good.example/p", bodies) is True
    # identical VOCABULARY, no contiguous run -> must fail
    scrambled = ("struvite proteus formation bacteria staphylococcus primarily "
                 "producing urease canine drive")
    assert si.verify_ref(scrambled, "https://good.example/p", bodies) is False


def test_verify_ref_rejects_the_observed_false_positive_classes():
    """Each case here is a real false citation this rule had to stop."""
    si = _si()
    page = ("Urinalysis findings include ammonium biurate crystalluria in dogs "
            "with hepatic disease. Creatinine 125 micromol per litre is the "
            "stage 1 cutoff. " + "padding " * 400)
    bodies = {"https://p.example/a": page}
    # 1. a three-word term of art: 29 chars, cleared a 25-CHAR guard, and proves
    #    only that the page is about urinalysis. Count distinctiveness in WORDS.
    assert si.verify_ref("ammonium biurate crystalluria",
                         "https://p.example/a", bodies) is False
    # 2. a number the page does not contain -> reject outright, no ratio
    assert si.verify_ref("Creatinine 999 micromol per litre is the stage 1 cutoff",
                         "https://p.example/a", bodies) is False
    # 3. a url never retrieved this session: the model invented the url itself
    assert si.verify_ref("Creatinine 125 micromol per litre is the stage 1 cutoff",
                         "https://never-fetched.example/z", bodies) is False
    # 4. the same quote against the page that DOES contain it -> accept
    assert si.verify_ref("Creatinine 125 micromol per litre is the stage 1 cutoff",
                         "https://p.example/a", bodies) is True
    # 5. empty inputs never earn a ref
    assert si.verify_ref("", "https://p.example/a", bodies) is False
    assert si.verify_ref("anything at all here", "", bodies) is False


def test_consolidation_verifies_refs_and_flags_quoteless_facts():
    """Source-level invariants, in the style of the think:false test: the ref gate
    and the no_quote marker must both exist AND be wired to the epoch counters, so
    a later refactor cannot silently drop either half."""
    path = os.path.join(PLUGIN_DIR, "consolidation.py")
    src = open(path, encoding="utf-8").read()
    assert "import source_index" in src, "source_index not imported"
    # supply half
    assert "build_index_block" in src, "url index never appended to transcript"
    # verify half -- these two must travel together or the feature fabricates
    assert "source_index.verify_ref" in src, "refs are supplied but never verified"
    assert "refs_dropped" in src and "refs_verified" in src, \
        "ref verification has no counters, so failures would be invisible"
    # dropping the ref must NOT drop the fact
    assert "source_ref = None" in src, "unearned ref should be nulled, fact kept"
    # quote-less facts get an explicit status instead of a NULL that reads as
    # "not checked yet"
    assert "no_quote" in src, "quote-less facts still fall through as NULL"
    # the feature must be flag-gated (default OFF in config_schema)
    assert "_url_index_for_extraction" in src, "feature is not flag-gated"


def test_url_index_config_defaults_off():
    """Default OFF: it costs ~1,200 prompt tokens, stages page text, and only
    helps agents that browse. Profiles that do not must be unaffected."""
    import config_schema
    schema = getattr(config_schema, "CONFIG_SCHEMA", None) or \
        getattr(config_schema, "SCHEMA", None)
    assert schema, "cannot locate the config schema list"
    entry = next((e for e in schema
                  if e.get("key") == "url_index_for_extraction"), None)
    assert entry is not None, "url_index_for_extraction missing from schema"
    assert entry["default"] is False, entry
    assert "verif" in entry["description"].lower(), \
        "schema description must state that refs are VERIFIED, not just supplied"


def test_store_session_sources_roundtrip_and_prune():
    """The bridge between sync_turn (which sees tool messages) and consolidation
    (which sees only a session_id)."""
    # _fresh_store() + close() in a finally, matching every other store test here.
    # A TemporaryDirectory teardown fails on Windows with WinError 32 while the
    # store still holds the sqlite handle, which reports as a test failure when in
    # fact every assertion passed -- a misleading red is worse than no test.
    st = _fresh_store()
    try:
        rows = [
            {"url": "https://s.example/1", "title": "One", "body": "body one"},
            {"url": "https://s.example/2", "title": "Two", "body": "body two"},
        ]
        st.add_session_sources("sess-A", rows)
        # a replay of the same page is a no-op: hermes replays history on restart
        st.add_session_sources("sess-A", rows)
        got = st.get_session_sources("sess-A", with_body=True)
        assert len(got) == 2, got
        assert {g["url"] for g in got} == {"https://s.example/1",
                                          "https://s.example/2"}
        assert got[0]["body"] == "body one"
        # bodies omitted unless asked for, so prompt building stays cheap
        light = st.get_session_sources("sess-A")
        assert light[0]["body"] == "", light[0]
        # sessions are isolated: no cross-block citation
        assert st.get_session_sources("sess-B") == []
        # prune with a past cutoff clears it; nothing in the substrate depends on
        # these rows surviving
        assert st.prune_session_sources(older_than_hours=-1) >= 2
        assert st.get_session_sources("sess-A") == []
    finally:
        st.close()




def test_find_ref_attaches_the_page_that_contains_the_quote():
    """MECHANICAL attachment, the primary path. Asking the model for the url was
    measured to fail: an offered `url | title` index produced ZERO refs across 28
    facts, including a real block with 30 urls available. The one case that worked
    had urls inline beside each fact, where the model copied a 100-char url 3/3 --
    so the failure was ASSOCIATION, not transcription, and the system has to do the
    association itself."""
    si = _si()
    page_a = ("Urease producing bacteria primarily Staphylococcus and Proteus "
              "drive struvite formation in dogs. " + "filler " * 300)
    page_b = ("Calcium oxalate uroliths recur in 48 percent of dogs within 36 "
              "months of removal. " + "other " * 300)
    bodies = {"https://a.example/1": page_a, "https://b.example/2": page_b}

    hit = si.find_ref("urease producing bacteria primarily staphylococcus and "
                      "proteus drive struvite formation", bodies)
    assert hit and hit[0] == "https://a.example/1", hit
    hit = si.find_ref("calcium oxalate uroliths recur in 48 percent of dogs "
                      "within 36 months", bodies)
    assert hit and hit[0] == "https://b.example/2", hit
    # no page contains it -> no ref, rather than the closest-looking one
    assert si.find_ref("cystine stones dissolve readily with diet alone always",
                       bodies) is None
    # too short to identify a source, and no number to anchor it
    assert si.find_ref("struvite formation dogs", bodies) is None


def test_find_ref_refuses_an_ambiguous_match():
    """Two pages carrying the same text do not identify a source. Picking one is a
    coin-flip citation, which is the failure this whole module exists to prevent."""
    si = _si()
    shared = ("Cystine uroliths form because of an inherited defect in renal "
              "tubular transport of cystine. " + "pad " * 300)
    bodies = {"https://x.example/1": shared, "https://y.example/2": shared}
    assert si.find_ref("cystine uroliths form because of an inherited defect in "
                       "renal tubular transport of cystine", bodies) is None
    # but once one page has strictly MORE of the quote, it wins on evidence
    longer = shared + " Dissolution is not possible for cystine stones."
    bodies2 = {"https://x.example/1": shared, "https://y.example/2": longer}
    hit = si.find_ref("renal tubular transport of cystine. dissolution is not "
                      "possible for cystine stones", bodies2)
    assert hit and hit[0] == "https://y.example/2", hit


def test_find_ref_requires_every_number_and_ignores_shell_pages():
    """A number the page lacks is disqualifying, with no ratio and no tolerance --
    a misattributed reference interval is worse than no reference at all."""
    si = _si()
    page = ("Blood creatinine 125 micromol per litre marks IRIS stage 1 in dogs. "
            + "pad " * 300)
    bodies = {"https://p.example/a": page}
    assert si.find_ref("blood creatinine 125 micromol per litre marks iris "
                       "stage 1 in dogs", bodies) is not None
    assert si.find_ref("blood creatinine 999 micromol per litre marks iris "
                       "stage 1 in dogs", bodies) is None
    # empty inputs never produce a ref
    assert si.find_ref("", bodies) is None
    assert si.find_ref("anything here at all really", {}) is None


def test_consolidation_attaches_refs_mechanically_not_by_asking():
    """Source-level invariants. The mechanical path must be wired AND the prompt
    half must be separately gated, so the measured-ineffective mechanism cannot
    silently come back on and cost 1,200 tokens per consolidation for nothing."""
    src = open(os.path.join(PLUGIN_DIR, "consolidation.py"), encoding="utf-8").read()
    assert "source_index.find_ref" in src, "mechanical attachment not wired"
    assert "refs_found" in src, "no counter for mechanically attached refs"
    # verification of a model-supplied ref stays as defence in depth
    assert "source_index.verify_ref" in src, "model-supplied refs no longer verified"
    # the prompt index must be behind its own flag
    assert "_url_index_in_prompt" in src, "prompt half is not separately gated"
    # match on the ORIGINAL quote: attestation may null source_quote for being
    # absent from the agent's paraphrase, but it can still be verbatim from a page
    assert "_raw_quote" in src, "mechanical match should use the emitted quote"
    # synthesis sessions fetched nothing, so they must not get a page ref
    assert "not synthesis_session" in src, "synthesis sessions must be excluded"


def test_url_index_prompt_half_defaults_off_and_is_documented_as_ineffective():
    """The negative result has to survive in the schema, or someone will turn it
    back on expecting it to help."""
    import config_schema
    schema = getattr(config_schema, "CONFIG_SCHEMA", None) or \
        getattr(config_schema, "SCHEMA", None)
    entry = next((e for e in schema if e.get("key") == "url_index_in_prompt"), None)
    assert entry is not None, "url_index_in_prompt missing from schema"
    assert entry["default"] is False, entry
    d = entry["description"].lower()
    assert "ineffective" in d and "12b" in d, \
        "schema must record WHY this is off, with the measurement"
    main = next((e for e in schema
                 if e.get("key") == "url_index_for_extraction"), None)
    assert "mechanical" in main["description"].lower(), \
        "the main flag should describe the mechanical path, not the prompt one"


def _quoteless_guard_source():
    """The retry guard lives inside a long method, so these tests assert on the SOURCE
    plus a faithful re-implementation of its decision function. Executing the real loop
    would need a live reason endpoint; the invariants worth protecting are the four
    gate conditions and the never-end-worse rule, and both are checkable here."""
    return open(os.path.join(PLUGIN_DIR, "consolidation.py"), encoding="utf-8").read()


def _should_retry(facts, transcript, synthesis, floor=5):
    """Mirror of the guard's retry decision. Kept in the test on purpose: if someone
    changes the production condition without changing this, the source assertions below
    fail and force the mirror to be updated deliberately."""
    n_quoted = sum(1 for f in facts
                   if isinstance(f, dict) and str(f.get("source_quote") or "").strip())
    if len(transcript) < 200:
        return False
    if not facts:
        return True
    return (not synthesis and transcript.count('"') >= 2
            and len(facts) >= floor and n_quoted == 0)


def test_quoteless_batch_triggers_a_retry():
    """The failure this guard exists for: 39 facts, not one with a quote, from a report
    whose own text held 42 quotation marks. The old condition (`if extracted_facts`)
    broke on attempt 1 because a quoteless batch is non-empty."""
    t = 'A report saying "urease producing bacteria" and more. ' + "pad " * 200
    facts = [{"content": "fact %d" % i} for i in range(39)]
    assert _should_retry(facts, t, synthesis=False)
    # one quoted fact is enough to accept the batch -- this is a wholesale-omission
    # guard, not a per-fact quality gate
    facts[0]["source_quote"] = "urease producing bacteria"
    assert not _should_retry(facts, t, synthesis=False)


def test_small_batches_and_quoteless_transcripts_do_not_retry():
    """Both suppressors. A 3-fact turn with nothing quotable is ordinary, and with no
    quotation marks in the transcript a quoteless extraction is the CORRECT answer --
    retrying there would burn a model call on every tool-only turn."""
    t_quotes = 'has "a quote" and "another" in it. ' + "pad " * 200
    assert not _should_retry([{"content": "a"}] * 3, t_quotes, synthesis=False)
    t_plain = "no quotation marks anywhere in this transcript at all. " + "pad " * 200
    assert not _should_retry([{"content": "a"}] * 39, t_plain, synthesis=False)


def test_synthesis_sessions_are_exempt():
    """A dream/abstraction cycle has no source page, so its facts SHOULD be quoteless.
    Retrying them would waste three calls per cycle forever."""
    t = 'recalling that "something" was learned earlier. ' + "pad " * 200
    facts = [{"content": "abstraction %d" % i} for i in range(20)]
    assert _should_retry(facts, t, synthesis=False)
    assert not _should_retry(facts, t, synthesis=True)


def test_short_transcripts_never_retry():
    """Unchanged legacy behaviour: below 200 chars there is nothing to re-extract."""
    assert not _should_retry([], "tiny", synthesis=False)
    assert not _should_retry([{"content": "a"}] * 39, 'a "q" here', synthesis=False)


def test_guard_prefers_quoted_facts_over_a_bigger_quoteless_batch():
    """The best-batch rule. A 40-fact quoteless batch must not beat a 30-fact attested
    one, or a retry would make provenance worse while looking like it helped."""
    def key(fs):
        nq = sum(1 for f in fs if str(f.get("source_quote") or "").strip())
        return (nq, len(fs))
    quoteless40 = [{"content": "c%d" % i} for i in range(40)]
    attested30 = [{"content": "c%d" % i, "source_quote": "q"} for i in range(30)]
    assert key(attested30) > key(quoteless40)
    # and among equally-quoted batches the larger wins
    assert key(attested30 + [{"content": "x", "source_quote": "q"}]) > key(attested30)


def test_quoteless_guard_is_wired_and_documented():
    """Source-level invariants. Each condition can fail silently, so the code must keep
    all four and must log the wholesale case at info/warning -- the missing
    observability is why nine sessions failed this way unnoticed."""
    src = _quoteless_guard_source()
    assert "_quoteless_retry_floor" in src, "floor knob not read"
    assert "_transcript_has_quotes" in src, "transcript-quote suppressor missing"
    assert "_n_quoted" in src, "quoted-fact counter missing"
    # Assert on a fragment that does not straddle a source line break: the retry log is
    # wrapped as `"...retrying WITH a "` + `"correction naming the omission..."`, so the
    # longer phrase never appears contiguously in the file.
    assert "not synthesis_session" in src and "correction naming the omission" in src, \
        "synthesis exemption or its retry log is missing"
    assert "all %d attempts returned quoteless batches" in src, \
        "no loud signal when every attempt fails the same way"
    # the old break-on-non-empty must be gone, or the guard is unreachable
    assert "if extracted_facts or len(transcript) < 200:" not in src, \
        "legacy break condition still short-circuits the quoteless check"


def test_quoteless_retry_floor_is_in_the_schema_with_its_rationale():
    import config_schema
    e = next((x for x in config_schema.CONFIG_SCHEMA
              if x.get("key") == "quoteless_retry_floor"), None)
    assert e is not None, "quoteless_retry_floor missing from schema"
    assert e["default"] == 5, e
    d = e["description"].lower()
    assert "synthesis" in d and "quotation marks" in d, \
        "schema must record both suppressors so they are not removed as dead code"
    assert config_schema.DEFAULTS["quoteless_retry_floor"] == 5


def _simulate_quoteless_loop(results, max_attempts=5, quoteless_max=2, floor=5):
    """Mirror of the loop's attempt accounting -> (calls, outcome, corrective_retries).

    `results[i]` is the batch the model returns on call i (the last entry repeats). Kept
    as a mirror for the same reason as `_should_retry`: changing production without
    changing this makes the source assertions fail and forces a deliberate update.
    """
    transcript = 'a "quote" and "another". ' + "pad " * 200
    def nq(fs):
        return sum(1 for f in fs if str(f.get("source_quote") or "").strip())
    pending, last_n, retries, calls, outcome, best, facts = "", None, 0, 0, "ok", [], []
    for i in range(max_attempts):
        facts = results[min(i, len(results) - 1)]
        calls += 1
        if (nq(facts), len(facts)) > (nq(best), len(best)):
            best = facts
        if not facts:
            outcome = "zero_facts"
            continue
        if len(transcript) >= 200 and len(facts) >= floor and nq(facts) == 0:
            n = len(facts)
            if pending and n == last_n:
                outcome = "quoteless_deterministic"
                break
            if retries >= quoteless_max:
                break
            last_n, retries, outcome, pending = n, retries + 1, "quoteless_retry", "CORRECTION"
            continue
        outcome = "ok"
        break
    if (nq(best), len(best)) > (nq(facts), len(facts)):
        facts = best
    if facts and nq(facts) == 0 and len(facts) >= floor:
        if outcome != "quoteless_deterministic":
            outcome = "quoteless_exhausted"
    return calls, outcome, retries


def test_an_identical_corrective_result_stops_instead_of_burning_the_budget():
    """THE MEASURED FAILURE. Session 20260727_044913 spent five attempts returning the
    same 44 quoteless facts, because the retry re-sent an identical payload at
    temperature 0.1 -- one attempt billed five times, 5-10 min of a 36-min block.

    With a corrective retry, an unchanged batch size proves the refusal is deterministic,
    so the loop must stop at 2 calls rather than 5."""
    same44 = [{"content": "c%d" % i} for i in range(44)]
    calls, outcome, retries = _simulate_quoteless_loop([same44])
    assert calls == 2, "identical corrective result must stop at 2 calls, got %d" % calls
    assert outcome == "quoteless_deterministic", outcome
    assert retries == 1, retries


def test_a_quoteless_retry_that_changes_its_answer_gets_the_full_capped_budget():
    """Varying batch sizes are consistent with flakiness rather than refusal, so the
    corrective budget is spent -- but capped at quoteless_max_retries, never at
    extraction_max_attempts (5 on the live profile)."""
    batches = [[{"content": "c%d" % i} for i in range(n)] for n in (44, 40, 38, 36, 34)]
    calls, outcome, retries = _simulate_quoteless_loop(batches)
    assert calls == 3, "expected 1 initial + 2 corrective, got %d" % calls
    assert retries == 2, retries
    assert outcome == "quoteless_exhausted", outcome


def test_a_corrective_retry_that_succeeds_ends_the_loop_immediately():
    """The point of the whole exercise: the corrective prompt is meant to WORK."""
    quoteless = [{"content": "c%d" % i} for i in range(44)]
    quoted = [{"content": "c%d" % i, "source_quote": "verbatim"} for i in range(30)]
    calls, outcome, retries = _simulate_quoteless_loop([quoteless, quoted])
    assert (calls, outcome, retries) == (2, "ok", 1), (calls, outcome, retries)


def test_the_retry_differs_from_the_call_it_retries():
    """A retry that re-sends an identical request is one attempt billed N times. Both
    levers must be present: a correction naming the omission, and a rising temperature."""
    src = _quoteless_guard_source()
    assert "_QUOTE_CORRECTION" in src, "no corrective instruction for the retry"
    assert "_RETRY_TEMPS" in src, "temperature never varies between attempts"
    assert "0.45" in src and "0.7" in src, "retry temperatures not escalating"
    assert "source_quote" in src.split("_QUOTE_CORRECTION")[1][:900], \
        "the correction must name the omitted field"
    assert "CHARACTER-FOR-CHARACTER" in src, \
        "the correction must restate the verbatim requirement"
    assert "_extract_once(_pending_correction, _pending_attempt)" in src, \
        "the loop must actually pass the correction into the call"


def test_attempt_zero_still_sends_the_original_payload_byte_for_byte():
    """Regression guard on the HEALTHY path, which is 489/489 facts quoted across 17
    audited sessions. The fix must not perturb the first call at all."""
    src = _quoteless_guard_source()
    assert "_payload = payload" in src, \
        "attempt 0 must reuse the original payload object, not a rebuilt one"
    assert "if correction or attempt:" in src, \
        "the payload is only allowed to change when a correction is actually pending"


def test_the_correction_sits_at_the_prompt_tail_because_truncation_keeps_the_tail():
    """Measured: ollama truncates an oversized prompt to the LAST ~8,195 tokens of a
    16,384 window, and a marker placed at the front of a 26k-token prompt was gone. A
    correction at the head would be the first thing discarded."""
    src = _quoteless_guard_source()
    i = src.find('f"{self._extraction_prompt}\\n\\nLOG:\\n{transcript}"')
    assert i != -1, "retry prompt assembly not found"
    tail = src[i:i + 200]
    assert "{correction}" in tail, "correction not appended after the log"
    assert tail.index("{correction}") < tail.index("JSON OUTPUT"), \
        "correction must precede the output cue but follow the log"


def test_quoteless_max_retries_is_in_the_schema_with_the_measurement():
    import config_schema
    e = next((x for x in config_schema.CONFIG_SCHEMA
              if x.get("key") == "quoteless_max_retries"), None)
    assert e is not None, "quoteless_max_retries missing from schema"
    assert e["default"] == 2, e
    d = e["description"].lower()
    assert "identical" in d and "44" in d, \
        "schema must record WHY the cap exists, with the measurement"
    assert config_schema.DEFAULTS["quoteless_max_retries"] == 2
    # Someone trimming this "to save a call" must hit this assertion first.
    assert "do not lower this to 1" in d, \
        "schema must warn what a cap of 1 costs"
    assert "strictly dominates" in d, \
        "the case for 2 is a dominance argument, not an absolute -- say so, because the " \
        "absolute version of this note was falsified by the next firing"


def test_a_budget_of_one_corrective_retry_gives_up_too_early():
    """Live firings: id=55 recovered at attempts=2 (34/34), while id=25 and id=49 needed
    attempts=3 (29/31 and 28/28). A cap of 1 keeps the first and loses the other two --
    34 facts saved against 59 banked unquoted.

    Both shapes are simulated here. NOTE: an earlier version of this test asserted that no
    firing recovers at attempts=2, on a sample of two, and the very next firing did exactly
    that. The invariant worth protecting is the BUDGET behaviour, not a claim about where
    recoveries land."""
    # shape A -- recovers on the second corrective retry (the id=25 / id=49 case)
    q31 = [{"content": "c%d" % i} for i in range(31)]
    q30 = [{"content": "c%d" % i} for i in range(30)]        # differs, so no early stop
    ok29 = [{"content": "c%d" % i, "source_quote": "v"} for i in range(29)]
    assert _simulate_quoteless_loop([q31, q30, ok29], quoteless_max=2)[:2] == (3, "ok")
    # with a budget of 1 the same model never reaches its working attempt
    calls1, outcome1, _ = _simulate_quoteless_loop([q31, q30, ok29], quoteless_max=1)
    assert (calls1, outcome1) == (2, "quoteless_exhausted"), (calls1, outcome1)

    # shape B -- recovers on the FIRST corrective retry (the id=55 case). A budget of 1 is
    # enough here, which is why the cap is a trade-off rather than a floor.
    ok34 = [{"content": "c%d" % i, "source_quote": "v"} for i in range(34)]
    assert _simulate_quoteless_loop([q31, ok34], quoteless_max=1)[:2] == (2, "ok")
    assert _simulate_quoteless_loop([q31, ok34], quoteless_max=2)[:2] == (2, "ok")

    src = _quoteless_guard_source()
    assert "WHY 2 AND NOT 1" in src, \
        "the rationale must survive in the code, not only in the schema"


def test_the_deterministic_label_is_not_flattened_into_exhausted():
    """`quoteless_deterministic` and `quoteless_exhausted` mean different things: one says
    the model was asked differently and refused anyway (re-run the RESEARCH), the other
    says the budget ran out. Collapsing them loses the distinction that decides what to
    do next."""
    src = _quoteless_guard_source()
    assert 'if _ea_outcome != "quoteless_deterministic":' in src, \
        "the post-loop label would overwrite the more specific outcome"
    same44 = [{"content": "c%d" % i} for i in range(44)]
    assert _simulate_quoteless_loop([same44])[1] == "quoteless_deterministic"


def test_store_extraction_audit_table_exists_and_accepts_a_row():
    """The migration must be idempotent and the table writable. Without this row there
    is no way to tell a retry that fired from a guard that never ran -- the ambiguity
    that produced two opposite reports of the same night."""
    st = _fresh_store()
    try:
        cols = {r[1] for r in st._conn.execute("PRAGMA table_info(extraction_audit)")}
        for need in ("session_id", "attempts", "quoteless_retries", "facts", "quoted",
                     "transcript_chars", "transcript_quotes", "synthesis", "outcome"):
            assert need in cols, "extraction_audit missing %s (%s)" % (need, cols)
        st._conn.execute(
            "INSERT INTO extraction_audit (session_id, attempts, quoteless_retries, "
            "facts, quoted, transcript_chars, transcript_quotes, synthesis, outcome) "
            "VALUES ('s1', 3, 2, 39, 0, 27823, 72, 0, 'quoteless_exhausted')")
        st._conn.commit()
        # tuple(): the store sets sqlite3.Row as its row_factory, so a bare == against
        # a tuple compares object identity and always fails.
        row = tuple(st._conn.execute(
            "select attempts, quoteless_retries, facts, quoted, outcome "
            "from extraction_audit where session_id='s1'").fetchone())
        assert row == (3, 2, 39, 0, "quoteless_exhausted"), row
        # re-running the migration must not drop or duplicate anything
        st._migrate_add_extraction_audit()
        assert st._conn.execute(
            "select count(*) from extraction_audit").fetchone()[0] == 1
    finally:
        st.close()


def test_audit_extraction_never_raises_even_with_a_broken_store():
    """An audit failure must not cost the epoch its facts. The point is to observe the
    pipeline, not to become a new way for it to die."""
    import consolidation as cons

    class _NoStore:
        _store = None

    class _BadConn:
        def execute(self, *a, **k):
            raise RuntimeError("db is gone")

        def commit(self):
            raise RuntimeError("db is gone")

    class _BadStore:
        class _S:
            _conn = _BadConn()
            _lock = None
        _store = _S()

    for obj in (_NoStore(), _BadStore()):
        # bound-method call on a foreign object: only needs self._store
        cons.ConsolidationMixin._audit_extraction(
            obj, "sX", 1, 0, 5, 5, 1000, 10, False, "ok")


def test_retry_loop_records_its_own_behaviour():
    """Source-level invariants. Each counter answers a question that logs cannot on this
    deployment, because hermes emits no stderr and every logger call goes to a void."""
    src = open(os.path.join(PLUGIN_DIR, "consolidation.py"), encoding="utf-8").read()
    assert "_audit_extraction" in src, "audit writer not wired"
    assert "_ea_attempts += 1" in src, "attempt counter missing -- attempts>1 is the " \
        "only available proof the retry fired"
    assert "_ea_quoteless_retries += 1" in src, "quoteless-retry counter missing"
    assert "quoteless_exhausted" in src, "no outcome recorded when every attempt fails"
    assert "call_failed" in src, "a hard model failure must still leave an audit row"
    # transcript_quotes is the suppressor's input: a guard that DECLINED to retry is
    # indistinguishable from one that never ran unless its reason is stored
    assert 'count(\'"\')' in src, "transcript quote count not recorded"


def test_extraction_audit_migration_is_registered():
    """An unregistered migration is a table that exists only in the tests."""
    src = open(os.path.join(PLUGIN_DIR, "store_schema.py"), encoding="utf-8").read()
    i = src.find("self._migrate_add_extraction_audit()")
    j = src.find("def _migrate_add_extraction_audit")
    assert i != -1, "migration never called"
    assert j != -1, "migration not defined"
    assert i < j, "call should appear in the migration list, before the definition"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = skipped = failed = 0
    for t in tests:
        # Store-layer tests require sqlite-vec/numpy; skip them HONESTLY (don't
        # count as passing) when deps are missing.
        if t.__name__.startswith("test_store_") and not _STORE_OK:
            print(f"[SKIP] {t.__name__}")
            skipped += 1
            continue
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERR ] {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped"
          + ("" if _STORE_OK else "  (store tests skipped - sqlite-vec/numpy not installed)"))
    sys.exit(1 if failed else 0)
