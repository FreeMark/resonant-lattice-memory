"""Simulate the Hermes plugin loader without the real framework.

Verifies load-path (b): `import resonant_lattice; resonant_lattice.register(ctx)`
returns a registered LatticeMemoryProvider, and a fresh LatticeMemoryProvider()
constructs. The real agent/tools/hermes modules aren't installed in this dev box,
so we inject minimal stubs (mirrors how the store tests gate on sqlite-vec).
Run as a subprocess so it never pollutes the parent interpreter.
"""
import sys
import types
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)            # repo root — holds the package dir
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# --- minimal Hermes framework stubs ---
# Faithful to agent.memory_provider.MemoryProvider: an ABC whose abstractmethods
# are exactly name / is_available / initialize / get_tool_schemas. This makes the
# construct check actually enforce that all four stay satisfied across the MRO
# after get_tool_schemas is moved into a mixin.
from abc import ABC, abstractmethod


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


_agent = _stub("agent")
_stub("agent.memory_provider", MemoryProvider=MemoryProvider)
_agent.memory_provider = sys.modules["agent.memory_provider"]

_tools = _stub("tools")
_stub("tools.registry", tool_error=lambda msg: msg)
_tools.registry = sys.modules["tools.registry"]

_stub("hermes_constants", get_hermes_home=lambda: __import__("pathlib").Path(HERE))
_hcli = _stub("hermes_cli")
_stub("hermes_cli.config", cfg_get=lambda cfg, *keys, default=None: default)
_hcli.config = sys.modules["hermes_cli.config"]


class _Collector:
    def __init__(self):
        self.registered = []

    def register_memory_provider(self, provider):
        self.registered.append(provider)


def main():
    import resonant_lattice
    assert hasattr(resonant_lattice, "register"), "register(ctx) missing"
    assert hasattr(resonant_lattice, "LatticeMemoryProvider"), "LatticeMemoryProvider missing"

    ctx = _Collector()
    resonant_lattice.register(ctx)
    assert len(ctx.registered) == 1, f"register did not register exactly one provider: {ctx.registered}"
    prov = ctx.registered[0]
    assert prov.__class__.__name__ == "LatticeMemoryProvider"
    assert prov.name == "resonant_lattice", f"name property changed: {prov.name!r}"

    fresh = resonant_lattice.LatticeMemoryProvider()
    assert fresh.name == "resonant_lattice"
    # is_available() must work and (no sqlite-vec here) report False without raising.
    avail = fresh.is_available()
    # Mirror MemoryManager.add_provider: it calls get_tool_schemas() + name at
    # registration time (before initialize()), so these must work store-less.
    schemas = prov.get_tool_schemas()
    assert isinstance(schemas, list) and schemas and schemas[0].get("name") == "lattice_store", \
        f"get_tool_schemas() regressed: {schemas}"
    enum = schemas[0]["parameters"]["properties"]["action"]["enum"]
    assert "memory_audit" in enum, f"memory_audit action missing from schema: {enum}"
    # P4a tool-surface: pin/unpin/request_abstraction added; agent-facing 'remove'
    # retired from the enum (no-delete by design — fade via unhelpful feedback instead).
    for a in ("pin", "unpin", "request_abstraction"):
        assert a in enum, f"P4a action {a!r} missing from schema enum: {enum}"
    assert "remove" not in enum, f"'remove' must be retired from the agent enum (A21 no-delete): {enum}"
    print(f"[OK] import+register+construct succeeded; name={prov.name!r}; "
          f"is_available={avail}; tools={[s.get('name') for s in schemas]}")

    # End-to-end memory_audit when sqlite-vec is present: inject a real store
    # (no initialize()/Ollama) and exercise the read-only tool action.
    try:
        import sqlite_vec  # noqa: F401
        _have_vec = True
    except Exception:
        _have_vec = False
    if _have_vec:
        import tempfile, os as _os, json as _json
        import numpy as _np
        _plug = _os.path.join(PARENT, "resonant_lattice")
        if _plug not in sys.path:
            sys.path.insert(0, _plug)
        import store as _store_mod
        _db = _os.path.join(tempfile.mkdtemp(), "audit.db")
        s = _store_mod.LatticeStore(db_path=_db)
        v = _np.random.default_rng(1).standard_normal(s.vector_dim)
        v = (v / (_np.linalg.norm(v) or 1.0)).tolist()
        s.add_or_reinforce_fact("health probe fact", v, "general", "sess",
                                entities=["probe"])
        p = resonant_lattice.LatticeMemoryProvider()
        p._store = s
        p._retriever = object()   # non-None; memory_audit never touches it
        out = p.handle_tool_call("lattice_store", {"action": "memory_audit"})
        data = _json.loads(out)
        for k in ("total_facts", "by_tier", "active_conflict_groups", "memory_cycle"):
            assert k in data, f"memory_audit payload missing {k}: {data}"
        assert data["total_facts"] >= 1, data
        print(f"[OK] memory_audit action returned a snapshot ({len(data)} fields, "
              f"total_facts={data['total_facts']})")

        # --- P4a: no-delete gating + pin tool action (provider-level, real store) ---
        fid = s._conn.execute("SELECT id FROM semantic_facts LIMIT 1").fetchone()[0]
        # remove is refused for the agent (agent_can_delete defaults False) and steers to feedback.
        rm = p.handle_tool_call("lattice_store", {"action": "remove", "fact_id": fid})
        assert "feedback" in rm.lower() and "admin" in rm.lower(), f"no-delete gating regressed: {rm}"
        assert s.get_fact(fid) is not None, "fact was deleted despite no-delete gating"
        # pin flips the flag through the store; unpin clears it.
        pj = _json.loads(p.handle_tool_call("lattice_store", {"action": "pin", "fact_id": fid}))
        assert pj.get("pinned") is True and s.get_fact(fid)["pinned"] == 1, pj
        uj = _json.loads(p.handle_tool_call("lattice_store", {"action": "unpin", "fact_id": fid}))
        assert uj.get("pinned") is False and s.get_fact(fid)["pinned"] == 0, uj
        print("[OK] P4a no-delete gating + pin/unpin tool actions verified")

        # --- P4b: A22 confidence metadata surfaces in the <resonant_memory> block ---
        import retrieval as _retr
        _qv = _np.random.default_rng(7).standard_normal(s.vector_dim)
        _qv = (_qv / (_np.linalg.norm(_qv) or 1.0)).tolist()

        class _FixedR(_retr.LatticeRetriever):
            def _get_embedding(self, _text):
                return _qv

        s.add_or_reinforce_fact("a memorable pinned fact about widgets", _qv, "general", "sess",
                                entities=["widgets"])
        wid = s._conn.execute(
            "SELECT id FROM semantic_facts WHERE content LIKE 'a memorable pinned%'").fetchone()[0]
        s._conn.execute("UPDATE semantic_facts SET resonance_count=2, max_resonance_seen=9, "
                        "learned_at_cycle=4, pinned=1, tier='long' WHERE id=?", (wid,))
        s._conn.commit()
        p._retriever = _FixedR(s, "http://x", "m", min_similarity=-1.0)
        p._session_id = "sess"
        block = p._compute_prefetch("widgets", "sess")
        assert "[PRIORITY" in block, f"PRIORITY (pinned) tag missing from recall block:\n{block}"
        assert "peak:9" in block, f"peak metadata missing from recall block:\n{block}"
        assert "learned@c4" in block, f"entry-cycle metadata missing from recall block:\n{block}"
        print("[OK] P4b A22 metadata (peak/learned/PRIORITY) surfaced in recall block")
        s.close()
    else:
        print("[SKIP] memory_audit end-to-end (sqlite-vec not installed)")
    print("STUB-LOADER: PASS")


if __name__ == "__main__":
    main()
