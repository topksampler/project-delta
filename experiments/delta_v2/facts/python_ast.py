from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from experiments.delta_v2.facts.model import (
    Evidence,
    FactExtractionError,
    FactObservation,
    ObservationStore,
    RejectedFactCandidate,
    canonical_ast,
    normalize_literal,
    try_literal,
)


CONFIG_FIELD_FAMILY = "python.config_field.v1"
CLI_OPTION_FAMILY = "python.cli_option.v1"
ENVIRONMENT_VARIABLE_FAMILY = "python.environment_variable.v1"
LITERAL_DOMAIN_FAMILY = "python.literal_domain.v1"
FAMILY_IDS = (
    CONFIG_FIELD_FAMILY,
    CLI_OPTION_FAMILY,
    ENVIRONMENT_VARIABLE_FAMILY,
    LITERAL_DOMAIN_FAMILY,
)
PYTHON_FEATURE_VERSION = (3, 11)


@dataclass(frozen=True)
class PythonSource:
    snapshot_role: str
    path: str
    git_blob_sha1: str
    content: bytes

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    @property
    def module(self) -> str:
        path = PurePosixPath(self.path)
        parts = list(path.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    def parse(self) -> ast.Module:
        try:
            text = self.content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FactExtractionError(
                f"Python source is not UTF-8: {self.path}"
            ) from exc
        try:
            return ast.parse(
                text,
                filename=self.path,
                mode="exec",
                type_comments=True,
                feature_version=PYTHON_FEATURE_VERSION,
            )
        except SyntaxError as exc:
            raise FactExtractionError(
                f"Python 3.11 parse failed for {self.path}:{exc.lineno}: "
                f"{exc.msg}"
            ) from exc

    def evidence(self, node: ast.AST) -> Evidence:
        line_start = getattr(node, "lineno", None)
        line_end = getattr(node, "end_lineno", None)
        if not isinstance(line_start, int):
            raise FactExtractionError(
                f"AST node has no source line in {self.path}"
            )
        if not isinstance(line_end, int):
            line_end = line_start
        return Evidence(
            snapshot_role=self.snapshot_role,
            path=self.path,
            line_start=line_start,
            line_end=line_end,
            git_blob_sha1=self.git_blob_sha1,
            source_sha256=self.source_sha256,
        )

    def span_evidence(self, first: ast.AST, second: ast.AST) -> Evidence:
        first_evidence = self.evidence(first)
        second_evidence = self.evidence(second)
        return Evidence(
            snapshot_role=self.snapshot_role,
            path=self.path,
            line_start=min(
                first_evidence.line_start,
                second_evidence.line_start,
            ),
            line_end=max(
                first_evidence.line_end,
                second_evidence.line_end,
            ),
            git_blob_sha1=self.git_blob_sha1,
            source_sha256=self.source_sha256,
        )


def _terminal_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _expression_contract(node: ast.AST | None) -> dict[str, Any]:
    if node is None:
        return {"kind": "absent"}
    resolved, value = try_literal(node)
    if resolved:
        return {"kind": "literal", "value": value}
    return {"kind": "expression", "ast": canonical_ast(node)}


def _reject(
    *,
    repository_id: str,
    source: PythonSource,
    store: ObservationStore,
    family_id: str,
    node: ast.AST,
    reason: str,
) -> None:
    store.reject(
        RejectedFactCandidate.create(
            repository_id=repository_id,
            family_id=family_id,
            reason=reason,
            evidence=source.evidence(node),
        )
    )


def _add_observation(
    *,
    repository_id: str,
    source: PythonSource,
    store: ObservationStore,
    family_id: str,
    semantic_key: str,
    value: Any,
    evidence_node: ast.AST,
) -> None:
    store.add(
        FactObservation.create(
            repository_id=repository_id,
            family_id=family_id,
            semantic_key=semantic_key,
            value=value,
            evidence=source.evidence(evidence_node),
        )
    )


def _is_class_var(annotation: ast.AST) -> bool:
    candidate = (
        annotation.value
        if isinstance(annotation, ast.Subscript)
        else annotation
    )
    return _terminal_name(candidate) == "ClassVar"


def _config_decorator_state(node: ast.ClassDef) -> str:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "config":
            return "exact"
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "config"
        ):
            return "exact"
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if _terminal_name(target) == "config":
            return "ambiguous"
    return "absent"


def _field_default_contract(
    node: ast.AnnAssign,
) -> tuple[str, Any, Any, dict[str, Any]] | None:
    if node.value is None:
        return "required", None, {"kind": "absent"}, {}

    default_kind = "assignment"
    static_constraints: dict[str, Any] = {}
    literal_node: ast.AST | None = node.value
    if isinstance(node.value, ast.Call):
        terminal_name = _terminal_name(node.value.func)
        if terminal_name == "Field":
            default_kind = "pydantic-field"
        elif terminal_name == "field":
            default_kind = "dataclasses-field"
        if default_kind != "assignment":
            if any(keyword.arg is None for keyword in node.value.keywords):
                return None
            keywords = {
                keyword.arg: keyword.value
                for keyword in node.value.keywords
                if keyword.arg is not None
            }
            literal_node = (
                keywords.get("default")
                or (node.value.args[0] if node.value.args else None)
            )
            static_constraints = {
                name: _expression_contract(value)
                for name, value in sorted(keywords.items())
                if name not in {"default", "default_factory"}
            }

    return (
        default_kind,
        canonical_ast(node.value),
        _expression_contract(literal_node),
        static_constraints,
    )


def _extract_config_fields(
    *,
    repository_id: str,
    source: PythonSource,
    tree: ast.Module,
    store: ObservationStore,
) -> None:
    for statement in tree.body:
        if not isinstance(statement, ast.ClassDef):
            continue
        decorator_state = _config_decorator_state(statement)
        if decorator_state == "ambiguous":
            _reject(
                repository_id=repository_id,
                source=source,
                store=store,
                family_id=CONFIG_FIELD_FAMILY,
                node=statement,
                reason="config decorator is aliased or attributed",
            )
            continue
        if decorator_state != "exact":
            continue

        for class_statement in statement.body:
            if not isinstance(class_statement, ast.AnnAssign):
                continue
            if not isinstance(class_statement.target, ast.Name):
                _reject(
                    repository_id=repository_id,
                    source=source,
                    store=store,
                    family_id=CONFIG_FIELD_FAMILY,
                    node=class_statement,
                    reason="configuration field target is not a simple name",
                )
                continue
            if _is_class_var(class_statement.annotation):
                _reject(
                    repository_id=repository_id,
                    source=source,
                    store=store,
                    family_id=CONFIG_FIELD_FAMILY,
                    node=class_statement,
                    reason="ClassVar is not an instance configuration field",
                )
                continue

            default_contract = _field_default_contract(class_statement)
            if default_contract is None:
                _reject(
                    repository_id=repository_id,
                    source=source,
                    store=store,
                    family_id=CONFIG_FIELD_FAMILY,
                    node=class_statement,
                    reason="configuration field uses expanded keyword arguments",
                )
                continue
            (
                default_kind,
                default_ast,
                literal_default,
                static_constraints,
            ) = default_contract
            field_name = class_statement.target.id
            semantic_key = (
                f"python-config-field:{source.module}:"
                f"{statement.name}.{field_name}"
            )
            _add_observation(
                repository_id=repository_id,
                source=source,
                store=store,
                family_id=CONFIG_FIELD_FAMILY,
                semantic_key=semantic_key,
                value={
                    "annotation_ast": canonical_ast(
                        class_statement.annotation
                    ),
                    "default_kind": default_kind,
                    "default_ast": default_ast,
                    "literal_default": literal_default,
                    "static_field_constraints": static_constraints,
                },
                evidence_node=class_statement,
            )


class _CliOptionVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        repository_id: str,
        source: PythonSource,
        store: ObservationStore,
    ) -> None:
        self.repository_id = repository_id
        self.source = source
        self.store = store
        self.scope: list[str] = []

    def _visit_scope(self, node: ast.AST, name: str) -> None:
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._visit_scope(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_scope(node, node.name)

    def visit_AsyncFunctionDef(  # noqa: N802
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:
        self._visit_scope(node, node.name)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            self._extract_call(node)
        self.generic_visit(node)

    def _extract_call(self, node: ast.Call) -> None:
        if any(isinstance(argument, ast.Starred) for argument in node.args):
            self._reject(node, "add_argument uses expanded positional arguments")
            return
        if any(keyword.arg is None for keyword in node.keywords):
            self._reject(node, "add_argument uses expanded keyword arguments")
            return

        option_strings: list[str] = []
        for argument in node.args:
            if (
                not isinstance(argument, ast.Constant)
                or not isinstance(argument.value, str)
                or not argument.value.startswith("-")
            ):
                self._reject(
                    node,
                    "add_argument option strings must be hyphen-prefixed literals",
                )
                return
            option_strings.append(argument.value)
        if not option_strings:
            self._reject(node, "add_argument declares no option strings")
            return

        long_options = [
            option for option in option_strings if option.startswith("--")
        ]
        if long_options:
            primary_option = sorted(
                long_options,
                key=lambda option: -len(option),
            )[0]
        else:
            primary_option = option_strings[0]

        keywords = {
            keyword.arg: keyword.value
            for keyword in node.keywords
            if keyword.arg is not None
        }
        derived_destination = primary_option.lstrip("-").replace("-", "_")
        destination = (
            _expression_contract(keywords["dest"])
            if "dest" in keywords
            else {"kind": "derived", "value": derived_destination}
        )
        lexical_scope = ".".join(self.scope) if self.scope else "<module>"
        semantic_key = (
            f"python-cli-option:{self.source.module}:"
            f"{lexical_scope}:{primary_option}"
        )
        _add_observation(
            repository_id=self.repository_id,
            source=self.source,
            store=self.store,
            family_id=CLI_OPTION_FAMILY,
            semantic_key=semantic_key,
            value={
                "option_strings": option_strings,
                "destination": destination,
                "action": _expression_contract(keywords.get("action")),
                "type_expression": _expression_contract(keywords.get("type")),
                "choices": _expression_contract(keywords.get("choices")),
                "default_expression": _expression_contract(
                    keywords.get("default")
                ),
                "required": _expression_contract(keywords.get("required")),
            },
            evidence_node=node,
        )

    def _reject(self, node: ast.AST, reason: str) -> None:
        _reject(
            repository_id=self.repository_id,
            source=self.source,
            store=self.store,
            family_id=CLI_OPTION_FAMILY,
            node=node,
            reason=reason,
        )


def _assignment_target_and_value(
    statement: ast.stmt,
) -> tuple[ast.Name | None, ast.AST | None]:
    if isinstance(statement, ast.AnnAssign):
        target = (
            statement.target
            if isinstance(statement.target, ast.Name)
            else None
        )
        return target, statement.value
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1:
            return None, statement.value
        target = (
            statement.targets[0]
            if isinstance(statement.targets[0], ast.Name)
            else None
        )
        return target, statement.value
    return None, None


def _type_checking_annotations(tree: ast.Module) -> dict[str, ast.AST]:
    annotations: dict[str, ast.AST] = {}
    for statement in tree.body:
        if not (
            isinstance(statement, ast.If)
            and isinstance(statement.test, ast.Name)
            and statement.test.id == "TYPE_CHECKING"
        ):
            continue
        for child in statement.body:
            if (
                isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
            ):
                if child.target.id in annotations:
                    raise FactExtractionError(
                        "duplicate TYPE_CHECKING annotation for "
                        f"{child.target.id}"
                    )
                annotations[child.target.id] = child.annotation
    return annotations


def _extract_environment_variables(
    *,
    repository_id: str,
    source: PythonSource,
    tree: ast.Module,
    store: ObservationStore,
) -> None:
    registry_nodes: list[tuple[ast.stmt, ast.AST | None]] = []
    for statement in tree.body:
        target, value = _assignment_target_and_value(statement)
        if target is not None and target.id == "environment_variables":
            registry_nodes.append((statement, value))
    if len(registry_nodes) > 1:
        raise FactExtractionError(
            f"multiple environment_variables assignments in {source.path}"
        )
    if not registry_nodes:
        return

    statement, registry = registry_nodes[0]
    if not isinstance(registry, ast.Dict):
        _reject(
            repository_id=repository_id,
            source=source,
            store=store,
            family_id=ENVIRONMENT_VARIABLE_FAMILY,
            node=statement,
            reason="environment_variables registry is not a dictionary literal",
        )
        return

    annotations = _type_checking_annotations(tree)
    for key, getter in zip(registry.keys, registry.values, strict=True):
        if key is None:
            _reject(
                repository_id=repository_id,
                source=source,
                store=store,
                family_id=ENVIRONMENT_VARIABLE_FAMILY,
                node=getter,
                reason="environment registry uses dictionary expansion",
            )
            continue
        if not (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
        ):
            _reject(
                repository_id=repository_id,
                source=source,
                store=store,
                family_id=ENVIRONMENT_VARIABLE_FAMILY,
                node=key,
                reason="environment registry key is not a string literal",
            )
            continue
        evidence = source.span_evidence(key, getter)
        store.add(
            FactObservation.create(
                repository_id=repository_id,
                family_id=ENVIRONMENT_VARIABLE_FAMILY,
                semantic_key=f"python-environment-variable:{key.value}",
                value={
                    "key": key.value,
                    "getter_expression_ast": canonical_ast(getter),
                    "type_checking_annotation_ast": (
                        canonical_ast(annotations[key.value])
                        if key.value in annotations
                        else None
                    ),
                },
                evidence=evidence,
            )
        )


def _direct_literal_elements(value: ast.AST) -> list[ast.AST] | None:
    if not (
        isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Name)
        and value.value.id == "Literal"
    ):
        return None
    if isinstance(value.slice, ast.Tuple):
        return list(value.slice.elts)
    return [value.slice]


def _contains_literal_name(value: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id == "Literal"
        for node in ast.walk(value)
    )


def _literal_alias_parts(
    statement: ast.stmt,
) -> tuple[ast.Name | None, ast.AST | None, ast.AST | None]:
    if isinstance(statement, ast.AnnAssign):
        target = (
            statement.target
            if isinstance(statement.target, ast.Name)
            else None
        )
        return target, statement.value, statement.annotation
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1:
            return None, statement.value, None
        target = (
            statement.targets[0]
            if isinstance(statement.targets[0], ast.Name)
            else None
        )
        return target, statement.value, None
    return None, None, None


def _extract_literal_domains(
    *,
    repository_id: str,
    source: PythonSource,
    tree: ast.Module,
    store: ObservationStore,
) -> None:
    allowed_types = (str, int, float, bool, bytes, type(None))
    for statement in tree.body:
        target, value, annotation = _literal_alias_parts(statement)
        if value is None or not _contains_literal_name(value):
            continue
        elements = _direct_literal_elements(value)
        if target is None:
            _reject(
                repository_id=repository_id,
                source=source,
                store=store,
                family_id=LITERAL_DOMAIN_FAMILY,
                node=statement,
                reason="Literal alias target is not one simple name",
            )
            continue
        if elements is None:
            _reject(
                repository_id=repository_id,
                source=source,
                store=store,
                family_id=LITERAL_DOMAIN_FAMILY,
                node=statement,
                reason="Literal is nested inside a union or wrapper",
            )
            continue

        literal_values: list[Any] = []
        invalid = False
        for element in elements:
            try:
                raw_value = ast.literal_eval(element)
            except (ValueError, TypeError, SyntaxError, MemoryError):
                invalid = True
                break
            if not isinstance(raw_value, allowed_types):
                invalid = True
                break
            literal_values.append(normalize_literal(raw_value))
        if invalid:
            _reject(
                repository_id=repository_id,
                source=source,
                store=store,
                family_id=LITERAL_DOMAIN_FAMILY,
                node=statement,
                reason="Literal domain contains a computed or unsupported value",
            )
            continue

        semantic_key = f"python-literal-domain:{source.module}:{target.id}"
        _add_observation(
            repository_id=repository_id,
            source=source,
            store=store,
            family_id=LITERAL_DOMAIN_FAMILY,
            semantic_key=semantic_key,
            value={
                "ordered_literal_values": literal_values,
                "annotation_ast": (
                    canonical_ast(annotation)
                    if annotation is not None
                    else None
                ),
            },
            evidence_node=statement,
        )


def should_parse_path(path: str) -> bool:
    return path.startswith("vllm/") and path.endswith(".py")


def extract_python_source(
    *,
    repository_id: str,
    source: PythonSource,
    store: ObservationStore,
) -> None:
    if not should_parse_path(source.path):
        return
    tree = source.parse()
    _CliOptionVisitor(
        repository_id=repository_id,
        source=source,
        store=store,
    ).visit(tree)

    if source.path.startswith("vllm/config/"):
        _extract_config_fields(
            repository_id=repository_id,
            source=source,
            tree=tree,
            store=store,
        )
        _extract_literal_domains(
            repository_id=repository_id,
            source=source,
            tree=tree,
            store=store,
        )
    if source.path == "vllm/envs.py":
        _extract_environment_variables(
            repository_id=repository_id,
            source=source,
            tree=tree,
            store=store,
        )
