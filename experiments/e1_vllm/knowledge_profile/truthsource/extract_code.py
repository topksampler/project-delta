"""Code-backed claim seeds: C_distill and C_full from a pinned vLLM snapshot."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

FLAG_RE = re.compile(r"add_argument\(\s*['\"](--[a-zA-Z0-9][a-zA-Z0-9-]*)['\"]")
ENV_ASSIGN_RE = re.compile(
    r"""^([A-Z][A-Z0-9_]*)\s*=\s*(?:os\.environ|envs\.|environment)""",
    re.M,
)
# vllm/envs.py often uses: env var names as string keys in a dict or annotation block
ENV_KEY_RE = re.compile(r"""['\"]([A-Z][A-Z0-9_]{2,})['\"]""")
ENV_PREFIX = "VLLM_"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_argparse_flags(snapshot: Path) -> list[dict]:
    """Distill rule (1): flags in engine/arg_utils.py."""
    path = snapshot / "vllm" / "engine" / "arg_utils.py"
    if not path.is_file():
        return []
    text = _read(path)
    out: list[dict] = []
    seen: set[str] = set()
    for m in FLAG_RE.finditer(text):
        name = m.group(1)
        if name in seen:
            continue
        seen.add(name)
        out.append(
            {
                "kind": "flag",
                "name": name,
                "source_path": "vllm/engine/arg_utils.py",
                "authority": "code_distill",
                "rule": "arg_utils_add_argument",
            }
        )
    return out


def extract_public_exports(snapshot: Path) -> list[dict]:
    """Distill rule (2): MODULE_ATTRS in vllm/__init__.py."""
    path = snapshot / "vllm" / "__init__.py"
    if not path.is_file():
        return []
    text = _read(path)
    out: list[dict] = []
    # "Name": ".module:obj"
    for m in re.finditer(r"""["']([A-Za-z_][A-Za-z0-9_]*)["']\s*:\s*["']([^"']+)["']""", text):
        name, target = m.group(1), m.group(2)
        if name in {"__version__", "__version_tuple__"}:
            continue
        out.append(
            {
                "kind": "export",
                "name": name,
                "source_path": "vllm/__init__.py",
                "target": target,
                "authority": "code_distill",
                "rule": "module_attrs_export",
            }
        )
    return out


def extract_env_vars(snapshot: Path) -> list[dict]:
    """Distill rule (3): VLLM_* keys referenced in vllm/envs.py."""
    path = snapshot / "vllm" / "envs.py"
    if not path.is_file():
        return []
    text = _read(path)
    keys = sorted({k for k in ENV_KEY_RE.findall(text) if k.startswith(ENV_PREFIX)})
    return [
        {
            "kind": "env",
            "name": k,
            "source_path": "vllm/envs.py",
            "authority": "code_distill",
            "rule": "envs_py_key",
        }
        for k in keys
    ]


def extract_distill(snapshot: Path) -> dict:
    flags = extract_argparse_flags(snapshot)
    exports = extract_public_exports(snapshot)
    envs = extract_env_vars(snapshot)
    return {
        "arm": "C_distill",
        "inclusion_rule": "arg_utils_flags | module_attrs | envs_VLLM_keys",
        "flags": flags,
        "exports": exports,
        "envs": envs,
        "stats": {
            "n_flags": len(flags),
            "n_exports": len(exports),
            "n_envs": len(envs),
        },
    }


def _module_depth(rel: str) -> int:
    return rel.count("/")


def extract_full_symbols(snapshot: Path, *, max_symbols: int = 600) -> list[dict]:
    """C_full: classes/functions under vllm/."""
    root = snapshot / "vllm"
    symbols: list[dict] = []
    if not root.is_dir():
        return symbols
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(snapshot).as_posix()
        if "/tests/" in rel or rel.endswith("_test.py"):
            continue
        if _module_depth(rel) > 4:
            continue
        try:
            tree = ast.parse(_read(path))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                symbols.append(
                    {
                        "kind": kind,
                        "name": node.name,
                        "qualname": f"{rel}:{node.name}",
                        "source_path": rel,
                        "authority": "code_full",
                        "rule": "ast_top_level",
                    }
                )
                if len(symbols) >= max_symbols:
                    return symbols
    return symbols


def extract_full_flags(snapshot: Path) -> list[dict]:
    """All add_argument flags under vllm/."""
    root = snapshot / "vllm"
    seen: set[str] = set()
    out: list[dict] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(snapshot).as_posix()
        text = _read(path)
        for m in FLAG_RE.finditer(text):
            name = m.group(1)
            if name in seen:
                continue
            seen.add(name)
            out.append(
                {
                    "kind": "flag",
                    "name": name,
                    "source_path": rel,
                    "authority": "code_full",
                    "rule": "any_vllm_add_argument",
                }
            )
    return out


def extract_full(snapshot: Path, *, max_symbols: int = 600) -> dict:
    distill = extract_distill(snapshot)
    symbols = extract_full_symbols(snapshot, max_symbols=max_symbols)
    flags = extract_full_flags(snapshot)
    return {
        "arm": "C_full",
        "inclusion_rule": "distill_union_ast_symbols_union_all_flags",
        "distill": distill,
        "symbols": symbols,
        "flags": flags,
        "stats": {
            **{f"distill_{k}": v for k, v in distill["stats"].items()},
            "n_symbols": len(symbols),
            "n_flags_all": len(flags),
        },
    }


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
