"""Static extractors for environment, public-export, and config contracts."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError):
        return {"dynamic": ast.unparse(node)}
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
        return value
    return repr(value)


def _evidence(path: Path, snapshot: Path, node: ast.AST) -> dict:
    return {
        "source_path": path.relative_to(snapshot).as_posix(),
        "line_start": node.lineno,
        "line_end": getattr(node, "end_lineno", node.lineno),
        "source_sha256": _sha(path),
    }


def _parse(path: Path) -> ast.Module:
    return ast.parse(
        path.read_text(encoding="utf-8", errors="replace"),
        filename=path.as_posix(),
    )


def _assigned_dict(tree: ast.Module, name: str) -> ast.Dict | None:
    for node in tree.body:
        value: ast.AST | None = None
        target_name: str | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            target_name = target.id if isinstance(target, ast.Name) else None
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target_name = node.target.id if isinstance(node.target, ast.Name) else None
            value = node.value
        if target_name == name and isinstance(value, ast.Dict):
            return value
    return None


def _getenv_default(node: ast.AST) -> Any:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or len(child.args) < 2:
            continue
        func = child.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
            and func.attr in {"getenv", "environ.get"}
        ):
            return _literal(child.args[1])
    return None


def extract_env_contracts(snapshot: Path) -> tuple[list[dict], list[dict]]:
    path = snapshot / "vllm" / "envs.py"
    if not path.is_file():
        return [], [{"source_path": "vllm/envs.py", "reject_reason": "missing"}]
    try:
        registry = _assigned_dict(_parse(path), "environment_variables")
    except SyntaxError as exc:
        return [], [
            {
                "source_path": "vllm/envs.py",
                "reject_reason": "syntax_error",
                "line": exc.lineno,
            }
        ]
    if registry is None:
        return [], [
            {
                "source_path": "vllm/envs.py",
                "reject_reason": "registry_not_found",
            }
        ]

    rows: list[dict] = []
    rejected: list[dict] = []
    for key_node, value_node in zip(registry.keys, registry.values, strict=True):
        key = _literal(key_node)
        if not isinstance(key, str) or not key.startswith("VLLM_"):
            continue
        if not isinstance(value_node, ast.Lambda):
            rejected.append(
                {
                    "entity": key,
                    "source_path": "vllm/envs.py",
                    "reject_reason": "resolver_not_lambda",
                    "line": value_node.lineno,
                }
            )
            continue
        rows.append(
            {
                "entity_type": "env_var",
                "entity": key,
                "display_entity": key,
                "scope": "vllm.envs.environment_variables",
                "contract": {
                    "default": _getenv_default(value_node.body),
                    "resolver": ast.unparse(value_node.body),
                },
                "evidence": _evidence(path, snapshot, key_node),
                "verifier": {
                    "id": "python_ast_env_registry_v1",
                    "result": "pass",
                },
            }
        )
    return sorted(rows, key=lambda row: row["entity"]), rejected


def extract_public_exports(snapshot: Path) -> tuple[list[dict], list[dict]]:
    path = snapshot / "vllm" / "__init__.py"
    if not path.is_file():
        return [], [{"source_path": "vllm/__init__.py", "reject_reason": "missing"}]
    try:
        registry = _assigned_dict(_parse(path), "MODULE_ATTRS")
    except SyntaxError as exc:
        return [], [
            {
                "source_path": "vllm/__init__.py",
                "reject_reason": "syntax_error",
                "line": exc.lineno,
            }
        ]
    if registry is None:
        return [], [
            {
                "source_path": "vllm/__init__.py",
                "reject_reason": "module_attrs_not_found",
            }
        ]
    rows: list[dict] = []
    for key_node, value_node in zip(registry.keys, registry.values, strict=True):
        name, target = _literal(key_node), _literal(value_node)
        if not isinstance(name, str) or not isinstance(target, str):
            continue
        rows.append(
            {
                "entity_type": "public_export",
                "entity": name,
                "display_entity": name,
                "scope": "vllm",
                "contract": {"target": target},
                "evidence": _evidence(path, snapshot, key_node),
                "verifier": {
                    "id": "python_ast_module_attrs_v1",
                    "result": "pass",
                },
            }
        )
    return sorted(rows, key=lambda row: row["entity"]), []


def _is_config_class(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "config":
            return True
        if isinstance(decorator, ast.Call):
            func = decorator.func
            if isinstance(func, ast.Name) and func.id == "config":
                return True
    return False


def _is_class_var(annotation: ast.AST) -> bool:
    text = ast.unparse(annotation)
    return text == "ClassVar" or text.startswith("ClassVar[")


def _field_description(body: list[ast.stmt], index: int) -> str | None:
    if index + 1 >= len(body):
        return None
    next_node = body[index + 1]
    if (
        isinstance(next_node, ast.Expr)
        and isinstance(next_node.value, ast.Constant)
        and isinstance(next_node.value.value, str)
    ):
        return " ".join(next_node.value.value.split())
    return None


def _field_value(node: ast.AST | None) -> tuple[Any, dict[str, Any]]:
    if not isinstance(node, ast.Call):
        return _literal(node), {}
    func_name = node.func.id if isinstance(node.func, ast.Name) else None
    if func_name not in {"Field", "field"}:
        return _literal(node), {}
    keywords = {kw.arg: _literal(kw.value) for kw in node.keywords if kw.arg}
    default = keywords.pop("default", _literal(node.args[0]) if node.args else None)
    return default, keywords


def extract_config_fields(snapshot: Path) -> tuple[list[dict], list[dict]]:
    root = snapshot / "vllm" / "config"
    if not root.is_dir():
        return [], [{"source_path": "vllm/config", "reject_reason": "missing"}]
    rows: list[dict] = []
    rejected: list[dict] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = _parse(path)
        except SyntaxError as exc:
            rejected.append(
                {
                    "source_path": path.relative_to(snapshot).as_posix(),
                    "reject_reason": "syntax_error",
                    "line": exc.lineno,
                }
            )
            continue
        module = path.relative_to(snapshot).with_suffix("").as_posix().replace("/", ".")
        for class_node in [
            node for node in tree.body if isinstance(node, ast.ClassDef)
        ]:
            if not _is_config_class(class_node):
                continue
            for index, field_node in enumerate(class_node.body):
                if not isinstance(field_node, ast.AnnAssign):
                    continue
                if not isinstance(field_node.target, ast.Name):
                    continue
                name = field_node.target.id
                if name.startswith("_") or _is_class_var(field_node.annotation):
                    continue
                entity = f"{module}.{class_node.name}.{name}"
                default, constraints = _field_value(field_node.value)
                rows.append(
                    {
                        "entity_type": "config_field",
                        "entity": entity,
                        "display_entity": f"{class_node.name}.{name}",
                        "scope": f"{module}.{class_node.name}",
                        "contract": {
                            "annotation": ast.unparse(field_node.annotation),
                            "default": default,
                            "constraints": constraints,
                            "description": _field_description(class_node.body, index),
                        },
                        "evidence": _evidence(path, snapshot, field_node),
                        "verifier": {
                            "id": "python_ast_vllm_config_v1",
                            "result": "pass",
                        },
                    }
                )
    return sorted(rows, key=lambda row: row["entity"]), rejected
