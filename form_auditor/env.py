"""Minimal .env loader (no extra dependency).

Reads KEY=VALUE lines into os.environ. Variables that are already set win, so a real
environment variable (or a Databricks cluster env var) always overrides the file.

    AZURE_DI_ENDPOINT=https://my-resource.cognitiveservices.azure.com
    AZURE_DI_KEY="abc123"            # quotes optional; `export KEY=...` also accepted
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_env_file(explicit: Optional[str | Path] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f".env file not found: {p}")
        return p
    for p in (Path.cwd() / ".env", PROJECT_ROOT / ".env"):
        if p.is_file():
            return p
    return None


def load_env(path: Optional[str | Path] = None, override: bool = False) -> Optional[Path]:
    """Load `path` (or ./.env, or <project>/.env). Returns the file used, if any."""
    p = find_env_file(path)
    if p is None:
        return None
    for raw in p.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        elif " #" in val:                       # trailing comment on an unquoted value
            val = val.split(" #", 1)[0].rstrip()
        if key and (override or key not in os.environ):
            os.environ[key] = val
    return p
