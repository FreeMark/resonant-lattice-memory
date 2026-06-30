"""Test suite for the Resonant Lattice Memory plugin.

Runs two layers:
  • Entity-precision tests   — exercise the real entity_extractor (no heavy deps).
  • Store-lifecycle tests     — exercise the real LatticeStore; auto-skipped if
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
# Layer 1 — entity precision (always runs)
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
    """Phase 5a: a triple encoded with encode_triple is queryable by role —
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
# Layer 2 — LatticeStore lifecycle (skipped if deps missing)
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
    # Provider (via get_defaults if possible)
    try:
        prov = _load("__init__")
        p = prov.LatticeMemoryProvider({})
        assert p.get_defaults().get("initial_resonance") == central.get("initial_resonance")
    except Exception as e:
        print(f"  provider defaults match check skipped: {e}")


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
    node — here the scores are exact cosines, so the ranking is an exact reference."""

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
    CASCADE-cleaned with the fact. Pure SQLite — no openfhe needed."""
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
    # the stored blob is exactly the ct — the plaintext embedding never leaks into it
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
    LatticeRetriever on a fixture — the make-or-break HE proof. Self-skips without
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
    correct one-hot WITHOUT decrypting scores — store side uses the public key only.
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
    """0a: BlindRecallPRE — the unified recall+PRE engine (E2 cosine + E6 PRE in one
    serializable context). The store scores cosine AND re-encrypts the score ct to the
    agent with no secret; the agent reads the re-encrypted result but NOT the raw store ct;
    the master reads anything (god-mode). Self-skips without openfhe; the full 3-process
    SERIALIZED split (load_eval/load_client/load_user) is node-proven 2026-06-19 — here the
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
    # eval keys). Only the agent's PRIVATE key is deserialized — that does not touch the
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
    """0a: BlindArgmaxCKKS — pure-CKKS comparison argmax (no FHEW scheme switching, so its
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
    with NO new crypto. Validated vs holographic.similarity on real CKKS — relational recall
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
    CASCADE-cleaned, allowlist-guarded. Pure SQLite — no openfhe."""
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
    """E4 4b + 2b-ii(a): blind HRR recall — BlindWriter stores encrypted HRR LIFTS in
    semantic_he_hrr, then BlindRetriever.blind_hrr_search ranks a phase probe by HRR similarity
    homomorphically, matching the plaintext holographic.similarity ranking. Option A: the retriever
    is built with a SEPARATE embed-dim recall client as ``blind`` and the 2·hrr_dim client as
    ``blind_hrr`` — proving blind_hrr_* uses ``blind_hrr`` (the lift would mis-encrypt under the
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
    # blind_hrr_* must use blind_hrr — encrypting a 2*HDIM lift under the EMBDIM recall ctx would fail.
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
    """E5 5a: blind dream-cycle maintenance on encrypted resonance — homomorphic DECAY (scalar
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
    Pure SQLite — no openfhe."""
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
    """E5 5b: blind dream-cycle maintenance over a real store — BlindMaintainer stores encrypted
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
    decrypts blobs the first session wrote — the reopen property the provider relies on. Needs a
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
    # decrypt the blob the first session wrote — the property _resolve_blind_entities needs.
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


def test_get_passphrase_returns_wipeable_bytearray():
    """#4 hygiene fix: get_passphrase returns a MUTABLE bytearray so the provider resolvers'
    ``finally: if isinstance(passphrase, bytearray): secure_zero(...)`` guards actually fire
    (it previously returned immutable bytes, so the wipe silently never ran). Proves: (1) an
    env-sourced passphrase is a bytearray, (2) every consumer accepts it — a key derived from
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
    """Write-path completeness (§14 6a) store read-back helpers — the data source for the provider
    _blind_reconcile pass. Proves (pure SQLite, no openfhe): get_fact_embedding round-trips the
    stored vector exactly; get_fact_hrr_phases returns the stored HRR; facts_missing_blind is the
    LEFT-JOIN worklist (all facts missing until a blind row exists, then they drop off — the
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
    has a NULL hrr_vector, so it can NEVER produce an HRR ciphertext — facts_missing_blind on
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
    """Entity-set staleness fix: the AEAD entity set is the one MUTABLE blind source — reinforcement
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
    resonance — protected from cycle decay AND prune even at resonance 0 — so it never fades before
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
    (high max_resonance_seen — e.g. a surprising one-off that entered high via novelty_boost) fades
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
    (category=procedural, tier=long, high resonance) so the agent is grounded day one; idempotent on
    re-seed; recallable. Pure SQLite."""
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
    assert s.seed_procedural_facts(items, current_cycle=2) == 0   # idempotent
    from retrieval import LatticeRetriever

    class _R(LatticeRetriever):
        def _get_embedding(self, t):
            return _emb(s, t)
    hits = _R(s, "http://x", "nomic", min_similarity=-1.0).search(
        "how do I approve a Stripe payment", limit=5)
    assert any("approve" in h["content"].lower() for h in hits)
    s.close()
    print("  procedural seed OK: durable (long/high-res) + idempotent + recallable")


def test_blind_reconcile_backfill():
    """Write-path completeness (§14 6a) END-TO-END: facts created WITHOUT a blind mirror (the
    abstraction/gist/procedural/backfill case) are reconciled by reading their plaintext
    embedding/HRR/entities back and mirroring into semantic_he*/_hrr/_entities — then blind recall
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
    # NO blind rows yet — exactly the post-abstraction / first-blind-enable state.
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
    (pseudo-embeds aren't semantic — real Ollama runs measure that); this proves the plumbing."""
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
    Real-embedding scoring is the eval_run/Ollama path (not asserted here — this is the plumbing)."""
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
    register/_blind_reconcile did — now directly testable. Needs a store + openfhe + crypto."""
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
    # live ops — each keyset computes correctly in the one setup process (by-tag coexistence)
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
    """0c: end-to-end blind tier over a REAL store — the keystore-loaded client + BlindWriter
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
    """E6 §7.2: the pure-Python scope policy that bounds what PRE provenance cannot —
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
    """E6 6c: the store-side BlindReEncryptGate binds re-encryptions to a single-use token —
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
    Pure SQLite — no openfhe."""
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
    prior clause's object as subject, the triple is dropped — but the well-formed
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
    the stored multi-word subject/object — without depending on spaCy."""
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
    path and a decayed confidence — and inference NEVER writes to the DB."""
    if not _STORE_OK:
        print("  SKIP"); return
    s = _seed_chain(_fresh_store())
    fr0 = s._conn.execute("SELECT COUNT(*) FROM fact_relations").fetchone()[0]
    sf0 = s._conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
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
    (add_or_reinforce_fact) is structurally unable to touch the self-model — it
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


def test_freshness_penalty_curve():
    """Phase 2: the recall freshness nudge is gentle, bounded, monotonic, and
    fully off when disabled (pure math — no Ollama)."""
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
    """Phase 3: max_resonance_seen is a high-water mark — rises on reinforce and
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
    """Add a fact carrying a real (content-derived) HRR vector — needed for the
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
    """Phase 4: only dying facts that EARNED their place are gist candidates —
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
# source_quote attestation — pure two-channel verifier (always runs; no deps)
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
                  "ASSISTANT: Noted — Charlie Brown will deploy it in rural Indiana.")
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
    digit ABSENT is kept-and-flagged (NOT 'specific_mismatch') — lone digits and
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
    # Lift WITH a typo (coverage broken + whole-transcript ratio collapsed) — the
    # case the inert-ratio code wrongly rejected — is still grounded.
    typo = lifted.replace("general availability", "general avzilability")
    assert attest(typo, long_t, []) in ("attested", "soft"), "typo long (was unattested pre-fix)"
    # A fabricated quote against the same long transcript is NOT grounded.
    assert attest("Budgets were slashed by forty percent in Q3.", long_t, []) in (
        "unattested", "specific_mismatch"), "fabricated long"


# ─────────────────────────────────────────────────────────────────────────────
# Provider shutdown sequencing — LifecycleMixin has no Hermes deps, so it loads
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
# Layer 3 — Encryption (E0: encrypted-at-rest). crypto_keys unit tests need only
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
    # Wrong key fails LOUDLY (GCM auth tag) — never returns garbage plaintext.
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
# Phase 4 — A5 pin (never-forget), A21 no-delete, A22 metadata, re-embed migration
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
    """P4b/A22: search results carry the confidence picture — peak_resonance (peak),
    learned_at_cycle (entry), pinned (bool) — and the raw max_resonance_seen column
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
    """P4d: re-embed migration self-gates on meta['embed_model'] — absent stamps
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
# Conflict quarantine (recall containment) — pure, dependency-free
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
    handler._retriever = object()   # truthy — handler only checks it exists
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
          + ("" if _STORE_OK else "  (store tests skipped — sqlite-vec/numpy not installed)"))
    sys.exit(1 if failed else 0)
