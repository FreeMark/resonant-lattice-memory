"""Standalone construction of the Resonant Lattice provider - no hermes gateway.

The RLM package is imported as a library; the only hermes tie is the 4-method MemoryProvider ABC,
which is importable wherever the RLM package + its deps are installed. We pass an explicit
hermes_home so the store's db lands inside this instance dir. Set RLM_PACKAGE_DIR if the package
lives somewhere other than ~/he-rlm.
"""
import os
import sys

HE_RLM = os.environ.get("RLM_PACKAGE_DIR", os.path.expanduser("~/he-rlm"))  # dir HOLDING resonant_lattice/
if HE_RLM not in sys.path:
    sys.path.insert(0, HE_RLM)

INSTANCE = os.path.dirname(os.path.abspath(__file__))  # this instance dir (holds config.yaml + lattice.db)


def load_config(path=None):
    import yaml
    path = path or os.path.join(INSTANCE, "config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_provider(session_id="grok-ingest", config=None, hermes_home=None):
    """Construct + initialize a write-enabled LatticeMemoryProvider against this instance's db."""
    import resonant_lattice
    cfg = config or load_config()
    home = hermes_home or INSTANCE
    prov = resonant_lattice.LatticeMemoryProvider(config=cfg)
    prov.initialize(session_id, hermes_home=home, agent_context="primary")  # agent_context flips write-enabled
    prov._write_enabled = True
    if getattr(prov, "_store", None) is None:
        raise SystemExit("RLM store failed to initialize (see logs) -- aborting.")
    return prov


def fact_count(store):
    try:
        return store._conn.execute("SELECT count(*) FROM semantic_facts").fetchone()[0]
    except Exception:
        return -1
