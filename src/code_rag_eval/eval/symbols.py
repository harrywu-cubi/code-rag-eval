from __future__ import annotations
import ast
from dataclasses import dataclass


@dataclass
class Symbol:
    qualified_name: str
    file: str
    start_line: int
    end_line: int
    kind: str            # "function" | "class"
    signature: str
    docstring: str | None


def _signature(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    args = [a.arg for a in node.args.args]  # type: ignore[attr-defined]
    return f"def {node.name}({', '.join(args)})"  # type: ignore[attr-defined]


def enumerate_symbols(text: str, file: str, module_prefix: str) -> list[Symbol]:
    tree = ast.parse(text)
    out: list[Symbol] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qn = f"{prefix}.{child.name}"
                kind = "class" if isinstance(child, ast.ClassDef) else "function"
                out.append(Symbol(
                    qualified_name=qn,
                    file=file,
                    start_line=child.lineno,
                    end_line=child.end_lineno or child.lineno,
                    kind=kind,
                    signature=_signature(child),
                    docstring=ast.get_docstring(child),
                ))
                visit(child, qn)

    visit(tree, module_prefix)
    return out
