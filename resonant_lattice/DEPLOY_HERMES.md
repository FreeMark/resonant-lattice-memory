# Deploying Resonant Lattice Memory on a fresh hermes-agent

The exact, battle-tested procedure (incl. the gotchas found in the field). Everything runs on the
machine where hermes-agent is installed. Paths assume the default `HERMES_HOME=~/.hermes`.

**Shorthands used below:**
- `VENV=~/.hermes/hermes-agent/venv/bin` — hermes-agent's Python venv (its `pip`/`python`).
- `PLUGINS=~/.hermes/plugins` — the user plugins dir (created in Step 2).

---

## 0. Prerequisites
- hermes-agent installed (`curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash`) and
  `hermes status` works.
- Ollama reachable (local or another LAN host) for the embedding + reasoning models.
- Confirm the venv + that the framework imports resolve:
  ```bash
  ~/.hermes/hermes-agent/venv/bin/python -c "import sys; print(sys.version)"
  ```

## 1. Install the hard dependencies into the hermes venv
`sqlite-vec` is **required** (the provider declines activation without it); `numpy` powers HRR.

> ⚠️ **Do NOT `source .../activate` and use bare `pip`.** hermes uses a **uv** venv that may not ship a
> standalone `pip`, so after `activate` your `pip` silently falls through to the **system** Python — which
> on Debian/Ubuntu refuses with `error: externally-managed-environment`. That is the wrong interpreter.
> Always target the venv's Python by absolute path with `python -m pip` (below). Do NOT use
> `--break-system-packages` (that installs into the system Python, which hermes-agent never uses).
> If `python -m pip` reports `No module named pip`, bootstrap it once: `$VENV/python -m ensurepip --upgrade`.

```bash
~/.hermes/hermes-agent/venv/bin/python -m pip install numpy sqlite-vec
# sanity: sys.prefix must end in .hermes/hermes-agent/venv (NOT /usr)
~/.hermes/hermes-agent/venv/bin/python -c "import sqlite_vec, numpy, sys; print('OK', sys.prefix)"
# (the rest of this guide's `$VENV/pip ...` shorthand == `~/.hermes/hermes-agent/venv/bin/python -m pip ...`)
$VENV/pip install numpy sqlite-vec   # equivalent IF the venv has its own pip
# verify the vec extension actually loads (it does on stock Linux python):
$VENV/python - <<'PY'
import sqlite3, sqlite_vec
c=sqlite3.connect(":memory:"); c.enable_load_extension(True); sqlite_vec.load(c)
print("sqlite_vec OK", c.execute("select vec_version()").fetchone()[0])
PY
```
Optional (encryption tiers — skip for a basic deploy):
```bash
$VENV/pip install argon2-cffi sqlcipher3-wheels   # Tier 0 at-rest (SQLCipher)
# Tier 1 blind store needs a real OpenFHE build (the `openfhe` pip wheel is a stub) — node-pending.
```

## 2. Install the plugin into the user plugins dir

**Note:** the bundled `.deployignore` lists the exact exclude patterns to use.
User-installed memory providers live in `$HERMES_HOME/plugins/<name>/` (imported under a synthetic
`_hermes_user_memory.<name>` package). Bundled providers in `hermes-agent/plugins/memory/` win on a
name clash, so keep the name `resonant_lattice`.

> ⚠️ **Copy RUNTIME modules only — never dev/test/eval scripts.** hermes discovery `exec_module()`s
> **every `.py`** in the plugin dir (to wire relative imports). It swallows import *exceptions* but NOT
> a *hang* — so any script with top-level side effects (especially a network call, like the `contest_*`/
> benchmark scripts) will make `hermes memory status` (and startup) **hang forever**. Keep dev scripts
> OUT of `$PLUGINS` (run them from elsewhere via `sys.path.insert`).
>
> Recommended: use a `.deployignore` or explicit excludes. Example full clean copy:
```bash
mkdir -p $PLUGINS
rsync -a --exclude __pycache__ --exclude '*.pyc' \
      --exclude 'test_*.py' --exclude 'eval_*.py' --exclude 'eval_*.json' \
      --exclude 'contest_*.py' --exclude 'CONTEST_PLAN.md' \
      --exclude 'original-holographic.py' --exclude 'REMEDIATION_PLAN.md' \
      --exclude 'RESEARCH_SCHOLAR.md' --exclude 'WEBDEV_CURRICULUM.md' \
      --exclude 'REFACTOR_NOTES.md' \
      /path/to/resonant_lattice/ $PLUGINS/resonant_lattice/
ls $PLUGINS/resonant_lattice/plugin.yaml  # must exist
# Optional one-liner helper (save as deploy-plugin.sh):
# rsync -a --exclude-from=.deployignore /path/to/resonant_lattice/ $PLUGINS/resonant_lattice/
```
# .deployignore example (add to source tree):
# test_*.py
# eval_*.py
# eval_*.json
# contest_*.py
# *_PLAN.md
# REMEDIATION_PLAN.md
# original-holographic.py
# RESEARCH_SCHOLAR.md
# WEBDEV_CURRICULUM.md
# REFACTOR_NOTES.md
# __pycache__/
# *.pyc
```

## Clean-deploy helper (Phase 9)
For convenience, a minimal Python helper (run from repo root):
```python
import shutil, os
src = "resonant_lattice"
dst = os.path.expanduser("~/.hermes/plugins/resonant_lattice")
if os.path.exists(dst): shutil.rmtree(dst)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
    "__pycache__", "*.pyc", "test_*.py", "eval_*.py", "eval_*.json",
    "contest_*.py", "*_PLAN.md", "REMEDIATION_PLAN.md", "original-holographic.py",
    "RESEARCH_SCHOLAR.md", "WEBDEV_CURRICULUM.md", "REFACTOR_NOTES.md"
))
print("Clean deploy done to", dst)
```
(Or use the .deployignore with rsync as shown above.)
```
`plugin.yaml` (ships with the plugin; declares the optional hooks it implements):
```yaml
name: resonant_lattice
version: "1.2.1"  # matches __version__ in __init__.py
description: "Resonant Lattice Memory — neuroplastic Hebbian long-term memory…"
hooks: [on_session_end, on_session_switch, on_pre_compress, on_delegation, on_memory_write]
```

## 3. Pull the models (Ollama)
```bash
ollama pull embeddinggemma:300m     # embeddings (768-d; best precision/byte in our tests)
ollama pull deepseek-v4-flash:cloud # memory/dream reasoner (top pick: wicked fast + excellent quality, generous cloud limits)
# Alternative: gemma4:26b or nemotron-3-ultra:cloud via Ollama account for heavier reasoning.
# The PRIMARY agent model is configured in hermes itself (hermes model / model.default), not here.
```
> ⚠️ **The config tag must EXACTLY match what `ollama list` shows on that endpoint.** Ollama tags differ
> by host: e.g. one box has `embeddinggemma:latest`, another `embeddinggemma:300m`, another
> `embeddinggemma:300m-bf16` — these are NOT interchangeable strings. A mismatch gives `model "…" not
> found` and the provider logs `Ollama probe failed` / `Failed to generate embedding`. Verify per
> endpoint: `curl http://<host>:11434/api/tags | grep <model>`, then set `embed_model` / `reason_model`
> to the exact tag (or `ollama pull` the one you want).

The three slots are independent and may live on different hosts — see the topology table at the bottom.

## 4. Configure (config.yaml, via dotted keys)
```bash
cp ~/.hermes/config.yaml ~/.hermes/config.yaml.bak     # always back up first
hermes config set memory.provider resonant_lattice
hermes config set plugins.resonant_lattice.embed_model embeddinggemma:300m
hermes config set plugins.resonant_lattice.ollama_endpoint_embed  http://localhost:11434
hermes config set plugins.resonant_lattice.reason_model gemma4:26b
hermes config set plugins.resonant_lattice.ollama_endpoint_reason http://localhost:11434
# Optional but recommended for a flagship/cloud reasoner that "thinks" a long time:
hermes config set plugins.resonant_lattice.reason_timeout 420
```
> NOTE: the provider uses Ollama's **native** API (`/api/embeddings`, `/api/generate`) for its embed/
> reason calls, so those endpoints are bare `http://host:11434` (NO `/v1`). The hermes **primary**
> model uses the OpenAI-compatible `/v1` path — that's a separate setting (`model.base_url`).

## 5. Activate + verify
```bash
# (a) The loader sees it and it's available (deps present):
HERMES_HOME=~/.hermes $VENV/python - <<'PY'
import os,sys; sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent"))
from plugins.memory import load_memory_provider
p=load_memory_provider("resonant_lattice")
print("loaded:", type(p).__name__, "| is_available:", p.is_available())
print("embed:", p._embed_model, "@", p._ollama_endpoint_embed)
print("reason:", p._reason_model, "@", p._ollama_endpoint_reason)
PY

# (b) hermes agrees it's active:
hermes memory status            # -> Provider: resonant_lattice ... available ✓ ← active

# (c) substrate smoke (add a fact, recall it) — proves embed + store + recall end-to-end:
HERMES_HOME=~/.hermes $VENV/python - <<'PY'
import os,sys,json; sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent"))
from plugins.memory import load_memory_provider
p=load_memory_provider("resonant_lattice")
p.initialize("smoke", hermes_home=os.path.expanduser("~/.hermes"), agent_context="primary")
print(p.handle_tool_call("lattice_store", {"action":"add","content":"My main GPU is a 3090ti.","category":"hardware"}))
print(json.loads(p.handle_tool_call("lattice_store", {"action":"search","query":"what GPU do I have"}))["results"][0]["content"])
p.shutdown()
PY

# (d) live agent turn (memory works automatically — no tool call needed):
hermes -z "In one sentence, what persistent memory system do you have active?"
```
A clean run: `is_available: True`, `hermes memory status` shows it active, the smoke prints the recalled
fact, and the agent reports "Resonant Lattice Memory." Delete `~/.hermes/resonant_lattice_memory.db*`
afterward for a pristine start.

## 6. Make it live + tuning
- **Gateway restart**: the running gateway loaded the old config at boot. CLI (`hermes`/`hermes chat`)
  reads config fresh, but **messaging/gateway sessions need a restart** to pick up the provider.
- **Keep the embedder warm**: defaults send Ollama `keep_alive=10m` and use a 30s `embed_timeout`, so a
  small/idle GPU that cold-loads (~6s) doesn't drop facts. Tune via `plugins.resonant_lattice.embed_timeout`
  / `embed_keep_alive`.
- **Behaviour flags** default sensibly (P1–P3 cognition on; relations/self-model/narrative/gist off).
  See `README.md` "Sample configuration" for all ~80 keys.

## Recommended multi-node topology (ideal: everything local)

The ideal setup runs the entire stack locally. When local inference capacity is limited, keep the primary agent and embeddings local while using a fast cloud model for the memory/dream layer (off the hot path).

| Slot | Model | Endpoint | Why |
|---|---|---|---|
| Primary agent (hermes `model.*`) | `gemma4:12b` | `http://<gpu1>:11434/v1` | local, fast, long-context, robust tool-calling |
| Memory/dream (`reason_model`) | `deepseek-v4-flash:cloud` (fast cloud winner) or `gemma4:26b` / `nemotron-3-ultra` (NVIDIA) | `http://<gpu2-or-cloud>:11434` | off hot path; fast + high quality extraction (cloud excellent when local limited; Nemotron thematic for NVIDIA) |
| Embedding (`embed_model`) | `embeddinggemma:300m` | `http://<small-gpu>:11434` | best precision/byte, 768-d, keep-alive pinned |

## Fast cloud profile (limited local inference — recommended for memory layer)

Ideal is fully local. When local inference capacity is constrained, keep the primary agent and embeddings local (or very light) and use a fast cloud model for the memory/dream reasoner (consolidation/abstraction runs off the hot path).

Observed cost on the lowest $20 Ollama tier during full heavy E2E + tool-grounding tests:
- ~0.1% of 5-hour usage limit
- ~0.02% of weekly limit

| Slot | Model | Endpoint | Why |
|---|---|---|---|
| Primary agent (hermes `model.*`) | `gemma4:12b` | `http://<local or small-gpu>:11434/v1` | local, fast, long-context, robust tool-calling |
| Memory/dream (`reason_model`) | `deepseek-v4-flash:cloud` | `http://localhost:11434` (Ollama Cloud) | wicked fast + strong quality; extremely low usage; ideal when local resources are limited |
| Embedding (`embed_model`) | `nomic-embed-text` or `embeddinggemma:300m` | `http://<local>:11434` | simple, pinned keep-alive, negligible cost |

## Troubleshooting (all seen in the field)
| Symptom | Cause → fix |
|---|---|
| `hermes memory status` **hangs** (Ctrl-C trace in `urlopen`/`exec_module`) | a non-runtime `.py` in the plugin dir runs a top-level network call at discovery → remove `contest_*`/`eval_*`/`test_*` from `$PLUGINS/resonant_lattice` (Step 2) |
| `is_available: False` / provider not active | `sqlite-vec` missing → `pip install sqlite-vec` into the **hermes venv** |
| `Ollama probe failed` / `Failed to generate embedding` | `embed_model` tag not present on the embed endpoint (tags differ per host!) → `curl …/api/tags \| grep embeddinggemma`, set the exact tag (e.g. `embeddinggemma:latest`) or `ollama pull` it |
| Consolidation stores **0 facts** | cold embedder timed out (old 5s) → fixed by `embed_timeout`/`keep_alive`; or **warm it first** with one embed call |
| Consolidation **times out** with a slow reasoner | raise `reason_timeout` (default 300) |
| Recall returns garbage after changing `embed_model` | the next dream cycle re-embeds all facts to the new model (rebuilds the vector index at the new dim) — turnkey; `force_dream_cycle` to do it now |
| `DIMENSION MISMATCH … DEGRADED (FTS-only)` | the DB's stored vec dim ≠ the embedder's; either keep the old model or let the re-embed migration run |
| `hermes -z` gives no answer on a recall | one-shot mode doesn't auto-consolidate, and the model may try to *call* the memory tool (needs approval) — use an interactive `hermes chat` session, or instruct it to answer from context |
