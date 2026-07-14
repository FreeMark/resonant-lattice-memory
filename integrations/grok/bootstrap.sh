#!/usr/bin/env bash
# One-time per-workspace bootstrap for the RLM<->grok memory integration.
#
# With grok's own memory writers disabled (RLM is the sole writer), grok will NOT auto-create the
# workspace memory dir, and its folder hash is not reproducible externally. This forces creation
# once via grok's remember tool, after which the SessionStart hook can find the dir (glob
# <reponame>-* under ~/.grok/memory) and write MEMORY.md from the lattice.
#
# Run ONCE from inside the git repo you want RLM memory for. Requires grok on PATH (or set GROK_BIN)
# and [memory] enabled in ~/.grok/config.toml (see grok-memory-config.toml).
set -u
GROK="${GROK_BIN:-grok}"

echo "RLM<->grok bootstrap for: $(pwd)"
command -v "$GROK" >/dev/null 2>&1 || { echo "ERROR: '$GROK' not found on PATH (set GROK_BIN)."; exit 1; }

# force grok to create the workspace memory dir + index via a single remember
"$GROK" --experimental-memory -p \
  "Use memory to remember this one bootstrap marker: RLM memory initialized for this workspace." \
  --max-turns 8 >/dev/null 2>&1 || true

# locate the created dir by repo name from the git origin
REPO=$(git remote get-url origin 2>/dev/null | sed -e 's#.*/##' -e 's#\.git$##')
DIR=""
if [ -n "${REPO:-}" ]; then
  DIR=$(ls -d "$HOME/.grok/memory/${REPO}-"*/ 2>/dev/null | head -1)
fi

if [ -n "$DIR" ]; then
  echo "OK: grok memory dir ready -> $DIR"
  echo "The SessionStart hook will populate MEMORY.md from the lattice on your next grok session."
else
  echo "WARN: memory dir not found. Check that:"
  echo "  - [memory] enabled=true in ~/.grok/config.toml"
  echo "  - grok saved the marker (its remember tool ran)"
  echo "  - this directory has a git 'origin' remote"
fi
