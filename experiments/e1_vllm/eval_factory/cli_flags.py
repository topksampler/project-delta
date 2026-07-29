"""Static, executable extraction of argparse CLI flag contracts."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return {"dynamic": ast.unparse(node)}
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
        return value
    return repr(value)


class _ArgumentVisitor(ast.NodeVisitor):
    def __init__(self, *, source_path: str, source_sha256: str) -> None:
        self.source_path = source_path
        self.source_sha256 = source_sha256
        self.scope: list[str] = []
        self.rows: list[dict] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        is_add_argument = (
            isinstance(func, ast.Attribute) and func.attr == "add_argument"
        ) or (isinstance(func, ast.Name) and func.id == "add_argument")
        if not is_add_argument:
            self.generic_visit(node)
            return

        names = [
            value
            for arg in node.args
            if isinstance((value := _literal(arg)), str) and value.startswith("--")
        ]
        if not names:
            self.generic_visit(node)
            return

        keywords = {kw.arg: _literal(kw.value) for kw in node.keywords if kw.arg}
        for name in names:
            self.rows.append(
                {
                    "entity_type": "cli_flag",
                    "entity": name,
                    "scope": ".".join(self.scope) or "<module>",
                    "contract": {
                        "default": keywords.get("default"),
                        "choices": keywords.get("choices"),
                        "required": keywords.get("required"),
                        "action": keywords.get("action"),
                        "help": keywords.get("help"),
                    },
                    "evidence": {
                        "source_path": self.source_path,
                        "line_start": node.lineno,
                        "line_end": getattr(node, "end_lineno", node.lineno),
                        "source_sha256": self.source_sha256,
                    },
                    "verifier": {
                        "id": "python_ast_argparse_v1",
                        "result": "pass",
                    },
                }
            )
        self.generic_visit(node)


def extract_cli_flags_with_rejections(snapshot: Path) -> tuple[list[dict], list[dict]]:
    """Extract literal argparse contracts and preserve parse failures."""
    package_root = snapshot / "vllm"
    if not package_root.is_dir():
        raise FileNotFoundError(f"missing vllm package under {snapshot}")

    rows: list[dict] = []
    rejected: list[dict] = []
    for path in sorted(package_root.rglob("*.py")):
        rel = path.relative_to(snapshot).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text, filename=rel)
        except SyntaxError as exc:
            rejected.append(
                {
                    "source_path": rel,
                    "reject_reason": "syntax_error",
                    "line": exc.lineno,
                    "detail": exc.msg,
                }
            )
            continue
        visitor = _ArgumentVisitor(
            source_path=rel,
            source_sha256=_source_sha(path),
        )
        visitor.visit(tree)
        rows.extend(visitor.rows)

    ordered = sorted(
        rows,
        key=lambda row: (
            row["entity"],
            row["scope"],
            row["evidence"]["source_path"],
            row["evidence"]["line_start"],
        ),
    )
    return ordered, rejected


def extract_cli_flags(snapshot: Path) -> list[dict]:
    """Compatibility wrapper returning verified rows only."""
    rows, _ = extract_cli_flags_with_rejections(snapshot)
    return rows


def semantic_contract(row: dict) -> str:
    """Stable signature excluding source location and formatting."""
    contract = dict(row["contract"])
    if row["entity_type"] == "config_field":
        contract.pop("description", None)
    payload = {
        "scope": row["scope"],
        "contract": contract,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)
