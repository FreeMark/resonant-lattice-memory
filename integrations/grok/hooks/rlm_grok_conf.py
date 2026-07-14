#!/usr/bin/env python3
"""Shared connection config for the RLM<->grok hooks.

Reads ~/.grok/rlm-grok.conf (simple KEY=VALUE lines); environment variables of the same name
override the file. Exposes SSH_KEY, SSH_HOST, REMOTE_DIR, REMOTE_PY and a ready SSH_OPTS list.
Keeping node host / key / paths here (not hardcoded in each hook) is what makes this portable.
"""
import os

CONF_PATH = os.path.expanduser("~/.grok/rlm-grok.conf")
_KEYS = ("SSH_KEY", "SSH_HOST", "REMOTE_DIR", "REMOTE_PY")


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
    out = {}
    for k in _KEYS:
        out[k] = os.environ.get(k, vals.get(k, ""))
    if out["SSH_KEY"]:
        out["SSH_KEY"] = os.path.expanduser(out["SSH_KEY"])
    return out


_C = _load()
SSH_KEY = _C["SSH_KEY"]
SSH_HOST = _C["SSH_HOST"]
REMOTE_DIR = _C["REMOTE_DIR"].rstrip("/")
REMOTE_PY = _C["REMOTE_PY"]
SSH_OPTS = ["-i", SSH_KEY, "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]


def configured():
    """True if the minimum connection settings are present."""
    return bool(SSH_KEY and SSH_HOST and REMOTE_DIR and REMOTE_PY)
