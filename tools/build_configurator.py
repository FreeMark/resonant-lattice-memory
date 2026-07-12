"""build_configurator.py - generate docs/index.html (the RLM Configurator).

Single source of truth: resonant_lattice/config_schema.py (every key,
description, default) plus the six overridable prompts in
resonant_lattice/prompts.py. Re-run this after adding config keys and the
new knobs appear on the page automatically.

    python tools/build_configurator.py

Presets live in tools/configurator_presets.json (overrides-on-top-of-
defaults); the build FAILS if a preset references an unknown key, so
presets can never silently rot.

The output is a fully static, dependency-free page suitable for GitHub
Pages (Settings -> Pages -> deploy from branch -> /docs on main).
"""
import io
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "resonant_lattice")
sys.path.insert(0, PKG)

from config_schema import CONFIG_SCHEMA  # noqa: E402
import prompts as prompt_defaults        # noqa: E402

# ---------------------------------------------------------------- grouping
GROUPS = [
    ("models", "Models & Endpoints"),
    ("cadence", "Cadence & Cycles"),
    ("hebbian", "Hebbian Dynamics"),
    ("recall", "Recall & Injection"),
    ("conflicts", "Conflict Detection & Containment"),
    ("abstraction", "Abstraction & Gist"),
    ("relations", "Relations & Graph"),
    ("toolmem", "Tool & Procedural Memory"),
    ("identity", "Identity, Narrative & Governance"),
    ("attestation", "Attestation & Grounding"),
    ("housekeeping", "Housekeeping & Health"),
    ("encryption", "Encryption"),
    ("prompts", "Prompt Overrides"),
    ("other", "Other"),
]

KEY_GROUP = {
    "ollama_endpoint_embed": "models", "ollama_endpoint_reason": "models",
    "embed_model": "models", "embed_timeout": "models", "embed_keep_alive": "models",
    "reason_model": "models", "reason_timeout": "models",
    "memory_reason_max_concurrency": "models",
    "relation_model": "models", "ollama_endpoint_relation": "models",
    "reflection_frequency": "cadence", "dream_every_n_consolidations": "cadence",
    "short_tier_cycles": "cadence", "mid_tier_cycles": "cadence",
    "extraction_max_attempts": "cadence",
    "initial_resonance": "hebbian", "decay_per_cycle": "hebbian",
    "promotion_resonance_threshold": "hebbian", "similarity_threshold": "hebbian",
    "reinforce_on_recall": "hebbian", "recall_bump": "hebbian",
    "reinforce_threshold": "hebbian", "novelty_enabled": "hebbian",
    "novelty_boost": "hebbian", "surprise_decay_discount": "hebbian",
    "importance_decay_discount": "hebbian", "importance_categories": "hebbian",
    "stale_decay_boost": "hebbian", "freshness_halflife_cycles": "hebbian",
    "max_long_facts": "hebbian", "forget_after_dormant_cycles": "hebbian",
    "procedural_staleness_bleed": "hebbian",
    "procedural_staleness_grace_cycles": "hebbian",
    "recall_floor": "recall", "recall_limit": "recall",
    "recall_relevance_margin": "recall", "surface_freshness_in_recall": "recall",
    "prefetch_proxy_min_overlap": "recall", "inject_current_datetime": "recall",
    "datetime_timezone": "recall",
    "conflict_decay_floor": "conflicts", "conflict_limbo": "conflicts",
    "keep_superseded": "conflicts", "max_superseded_history": "conflicts",
    "surface_conflicts": "conflicts",
    "conflict_surface_min_group_age_cycles": "conflicts",
    "conflict_sim_low": "conflicts", "conflict_sim_high": "conflicts",
    "detect_policy_conflicts": "conflicts", "detect_procedural_conflicts": "conflicts",
    "conflict_subject_veto": "conflicts", "conflict_llm_adjudication": "conflicts",
    "quarantine_high_stakes_conflicts": "conflicts",
    "abstraction_frequency": "abstraction", "abstraction_max_facts": "abstraction",
    "abstraction_max_clusters": "abstraction",
    "abstraction_min_cluster_size": "abstraction",
    "abstraction_max_cluster_size": "abstraction",
    "cluster_hrr_similarity": "abstraction", "cluster_entity_overlap": "abstraction",
    "abstraction_dedup_threshold": "abstraction",
    "gist_before_prune": "abstraction", "gist_floor": "abstraction",
    "gist_min_peak_resonance": "abstraction", "gist_frequency": "abstraction",
    "gist_min_cluster_size": "abstraction", "gist_max_clusters": "abstraction",
    "enable_relations": "relations", "relation_min_confidence": "relations",
    "relation_extract_llm": "relations", "relation_recall_hrr_floor": "relations",
    "max_inference_hops": "relations",
    "enable_tool_memory": "toolmem", "tool_distill_frequency": "toolmem",
    "tool_distill_min_episodes": "toolmem", "tool_distill_max_tools": "toolmem",
    "tool_distill_sample_size": "toolmem", "tool_episode_keep": "toolmem",
    "procedural_seed": "toolmem",
    "enable_self_model": "identity", "self_model_seed": "identity",
    "enable_narrative": "identity", "narrative_keep": "identity",
    "narrative_surface": "identity", "narrative_min_episodes": "identity",
    "gate_self_writes": "identity", "agent_can_delete": "identity",
    "verify_source_quote": "attestation", "quote_match_threshold": "attestation",
    "prune_keep_sessions": "housekeeping", "episode_max_rows": "housekeeping",
    "health_check_every_n_dream_cycles": "housekeeping",
    "health_near_cap": "housekeeping", "hrr_dim": "housekeeping",
    "encryption_mode": "encryption", "encryption_keystore_path": "encryption",
    "blind_he_keystore_path": "encryption", "blind_reconcile_batch": "encryption",
    "blind_scan_batch": "encryption", "blind_scan_concurrency": "encryption",
    "blind_gpu_recall": "encryption", "blind_gpu_socket": "encryption",
    "blind_gpu_binary": "encryption",
}

SELECT_OPTIONS = {
    "encryption_mode": ["none", "at_rest", "blind"],
}

# Keys whose real-world range differs from the default*5 heuristic (resonance
# units, archive-scale caps, long cloud timeouts). Without these the slider
# clamps below values the shipped presets actually use (e.g. Goldfish
# decay_per_cycle 5.0 against an inferred max of 1.0).
RANGE_OVERRIDES = {
    "decay_per_cycle": {"min": 0.0, "max": 10.0, "step": 0.05},
    "recall_bump": {"min": 0.0, "max": 5.0, "step": 0.01},
    "stale_decay_boost": {"min": 0.0, "max": 5.0, "step": 0.1},
    "novelty_boost": {"min": 0.0, "max": 10.0, "step": 0.1},
    "initial_resonance": {"min": 0, "max": 50, "step": 1},
    "promotion_resonance_threshold": {"min": 0, "max": 50, "step": 1},
    "health_near_cap": {"min": 0.0, "max": 50.0, "step": 0.5},
    "max_long_facts": {"min": 0, "max": 50000, "step": 1},
    "episode_max_rows": {"min": 0, "max": 100000, "step": 1},
    "tool_episode_keep": {"min": 0, "max": 10000, "step": 1},
    "max_superseded_history": {"min": 0, "max": 20000, "step": 1},
    "narrative_keep": {"min": 0, "max": 500, "step": 1},
    "freshness_halflife_cycles": {"min": 0, "max": 500, "step": 1},
    "forget_after_dormant_cycles": {"min": -1, "max": 1000, "step": 1},
    "reason_timeout": {"min": 0.0, "max": 1800.0, "step": 10},
    "embed_timeout": {"min": 0.0, "max": 300.0, "step": 5},
    "recall_limit": {"min": 0, "max": 2000, "step": 1},
    "hrr_dim": {"min": 64, "max": 4096, "step": 64},
}

PROMPT_KEYS = [
    ("extraction_prompt", "Rules the reason model follows when mining facts from a transcript.",
     prompt_defaults.DEFAULT_EXTRACTION_PROMPT),
    ("consolidation_prompt", "Rules for the abstraction pass that merges related long-tier facts.",
     prompt_defaults.DEFAULT_CONSOLIDATION_PROMPT),
    ("procedural_prompt", "Rules for distilling raw tool episodes into reusable procedural facts.",
     prompt_defaults.DEFAULT_PROCEDURAL_PROMPT),
    ("narrative_prompt", "Rules for the one-paragraph end-of-session autobiographical summary.",
     prompt_defaults.DEFAULT_NARRATIVE_PROMPT),
    ("relation_prompt", "Rules for extracting (subject, relation, object) triples from one fact.",
     prompt_defaults.DEFAULT_RELATION_PROMPT),
    ("gist_prompt", "Rules for compressing dying facts into a gist before pruning.",
     prompt_defaults.DEFAULT_GIST_PROMPT),
]


def infer_control(key, default):
    """Map a schema default to a UI control spec."""
    if key in SELECT_OPTIONS:
        return {"type": "select", "options": SELECT_OPTIONS[key]}
    if isinstance(default, bool):
        return {"type": "bool"}
    if key in RANGE_OVERRIDES:
        spec = dict(RANGE_OVERRIDES[key])
        spec["type"] = "int" if isinstance(default, int) else "float"
        return spec
    if isinstance(default, int):
        lo = -1 if key == "forget_after_dormant_cycles" else 0
        hi = max(default * 5, 10)
        return {"type": "int", "min": lo, "max": hi, "step": 1}
    if isinstance(default, float):
        if 0.0 <= default <= 1.0:
            return {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01}
        return {"type": "float", "min": 0.0, "max": max(default * 5, 10.0), "step": 0.1}
    if isinstance(default, (list, dict)):
        return {"type": "json"}
    return {"type": "text"}


def build_schema_blob():
    items = []
    for entry in CONFIG_SCHEMA:
        key, desc, default = entry["key"], entry["description"], entry["default"]
        spec = infer_control(key, default)
        spec.update({
            "key": key,
            "group": KEY_GROUP.get(key, "other"),
            "help": desc,
            "default": (json.dumps(default) if spec["type"] == "json" else default),
        })
        items.append(spec)
    for key, desc, default in PROMPT_KEYS:
        items.append({"type": "prompt", "key": key, "group": "prompts",
                      "help": desc, "default": default})
    return items


def load_presets(schema_keys):
    path = os.path.join(ROOT, "tools", "configurator_presets.json")
    presets = json.loads(io.open(path, encoding="utf-8").read())
    for p in presets:
        unknown = [k for k in p["values"] if k not in schema_keys]
        if unknown:
            raise SystemExit("preset '%s' references unknown keys: %s" % (p["id"], unknown))
    return presets


def git_short_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def main():
    schema = build_schema_blob()
    keys = {s["key"] for s in schema}
    presets = load_presets(keys)
    template = io.open(os.path.join(ROOT, "tools", "configurator_template.html"),
                       encoding="utf-8").read()
    html = (template
            .replace("__SCHEMA_JSON__", json.dumps(schema))
            .replace("__PRESETS_JSON__", json.dumps(presets))
            .replace("__GROUPS_JSON__", json.dumps(GROUPS))
            .replace("__KEY_COUNT__", str(len(schema)))
            .replace("__BUILD_DATE__", time.strftime("%Y-%m-%d"))
            .replace("__BUILD_SHA__", git_short_sha()))
    docs = os.path.join(ROOT, "docs")
    os.makedirs(docs, exist_ok=True)
    out = os.path.join(docs, "index.html")
    io.open(out, "w", encoding="utf-8", newline="\n").write(html)
    io.open(os.path.join(docs, ".nojekyll"), "w").write("")
    print("wrote %s (%d keys, %d presets, %d bytes)" % (
        out, len(schema), len(presets), os.path.getsize(out)))


if __name__ == "__main__":
    main()
