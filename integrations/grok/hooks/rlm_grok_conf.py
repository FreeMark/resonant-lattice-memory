#!/usr/bin/env python3
"""Shared connection config + transport for the RLM<->grok hooks.

Reads ~/.grok/rlm-grok.conf (simple KEY=VALUE lines); environment variables of the same name
override the file. Two transports, chosen automatically:

  LOCAL (default): grok runs on the SAME machine as the RLM instance. Node scripts run directly
    as local subprocesses -- no ssh. Only RLM_DIR (the instance dir) and RLM_PY (the python that
    imports the RLM package) are needed.
  SSH (opt-in): set SSH_HOST (and SSH_KEY) and the node scripts run over ssh/scp on that host.

REMOTE_DIR / REMOTE_PY are accepted as aliases for RLM_DIR / RLM_PY (backward compatible with
older confs). The hooks + MCP server call run() / push() and never build ssh themselves, so
switching transport is a one-line config change. Local mode targets a POSIX host (uses `sh -c`),
which is the normal case when grok runs on the RLM node.
"""
import os
import shutil
import subprocess

CONF_PATH = os.path.expanduser("~/.grok/rlm-grok.conf")
_KEYS = ("SSH_KEY", "SSH_HOST", "REMOTE_DIR", "REMOTE_PY", "RLM_DIR", "RLM_PY")


def _load():
    vals = {}
    try:
        with open(CONF_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return {k: os.environ.get(k, vals.get(k, "")) for k in _KEYS}


_C = _load()
SSH_KEY = os.path.expanduser(_C["SSH_KEY"]) if _C["SSH_KEY"] else ""
SSH_HOST = _C["SSH_HOST"]
# RLM_DIR / RLM_PY are the transport-neutral names; REMOTE_DIR / REMOTE_PY are back-compat aliases.
REMOTE_DIR = (_C["RLM_DIR"] or _C["REMOTE_DIR"]).rstrip("/")
REMOTE_PY = _C["RLM_PY"] or _C["REMOTE_PY"]
SSH_OPTS = (["-i", SSH_KEY] if SSH_KEY else []) + [
    "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]

# Transport: LOCAL unless an SSH_HOST is configured. Local is the default (grok on the RLM node).
LOCAL = not SSH_HOST


def configured():
    """True if the minimum settings for the active transport are present."""
    if not (REMOTE_DIR and REMOTE_PY):
        return False
    return True if LOCAL else bool(SSH_HOST and SSH_KEY)


def _argv(cmd_str):
    """The argv to run a node shell command under the active transport."""
    if LOCAL:
        return ["sh", "-c", cmd_str]          # same machine -> POSIX shell, no ssh
    return ["ssh", *SSH_OPTS, SSH_HOST, cmd_str]


def run(cmd_str, input=None, timeout=90, capture_output=True,
        text=False, encoding=None, errors=None):
    """Run a node shell command under the active transport (local `sh -c`, or ssh).

    Drop-in for the old subprocess.run(["ssh", ...]) call: same CompletedProcess result, but
    transport-agnostic. `input` may be bytes (default) or str (with text=True)."""
    return subprocess.run(_argv(cmd_str), input=input, capture_output=capture_output,
                          timeout=timeout, text=text, encoding=encoding, errors=errors)


def push(local_path, remote_relpath, timeout=120):
    """Place a local file at REMOTE_DIR/remote_relpath under the active transport.

    Local: a filesystem copy (creates the parent dir). SSH: scp. Returns (ok, error_str)."""
    dest = f"{REMOTE_DIR}/{remote_relpath}"
    if LOCAL:
        try:
            parent = os.path.dirname(dest)
            if parent:
                os.makedirs(parent, exist_ok=True)
            shutil.copy2(local_path, dest)
            return True, ""
        except Exception as e:
            return False, str(e)[:200]
    r = subprocess.run(["scp", *SSH_OPTS, local_path, f"{SSH_HOST}:{dest}"],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0, (r.stderr or "")[:200]
