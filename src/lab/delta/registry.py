"""Repository-to-experiment reconciliation plugin registry."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

VLLM_REPO = "https://github.com/vllm-project/vllm.git"


class UnsupportedRepositoryError(ValueError):
    pass


def normalize_repo(repo: str) -> str:
    value = repo.strip().rstrip("/")
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    parts = urlsplit(value)
    if parts.scheme and parts.netloc:
        path = parts.path.rstrip("/")
        if not path.endswith(".git"):
            path += ".git"
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))
    return value


def plugin_for_repo(repo: str, *, project_root: Path | None = None) -> Any:
    normalized = normalize_repo(repo)
    if normalized != VLLM_REPO:
        raise UnsupportedRepositoryError(
            f"unsupported DELTA repository: {repo!r}; supported: {VLLM_REPO}"
        )

    root = project_root or Path(__file__).resolve().parents[3]
    plugin_path = root / "experiments" / "e1_vllm" / "reconcile_plugin.py"
    spec = importlib.util.spec_from_file_location(
        "lab_delta_e1_vllm_reconcile_plugin", plugin_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reconcile plugin: {plugin_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.E1VllmReconcilePlugin(project_root=root)
