"""Generate Sphinx API pages from the FEcrys source tree without importing it.

The research modules initialise optional molecular-simulation and machine-
learning dependencies at import time.  Static AST extraction keeps the
documentation build lightweight while preserving signatures, docstrings, and
source locations.
"""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass
from pathlib import Path


GITHUB_BASE = "https://github.com/mme-ucl/FEcrys/blob/main"


@dataclass(frozen=True)
class CallableDoc:
    """Documentation metadata for one class or function definition."""

    name: str
    qualified_name: str
    kind: str
    signature: str
    docstring: str | None
    line: int
    depth: int


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Return a compact source-style function signature."""

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({ast.unparse(node.args)})"


def _constructor_signature(node: ast.ClassDef) -> str:
    """Return the class constructor signature when ``__init__`` is present."""

    initializer = next(
        (
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name == "__init__"
        ),
        None,
    )
    if initializer is None:
        return f"class {node.name}"
    arguments = ast.unparse(initializer.args)
    arguments = re.sub(r"^self\s*,?\s*", "", arguments)
    return f"class {node.name}({arguments})"


def _collect_callables(tree: ast.Module) -> list[CallableDoc]:
    """Collect active classes, functions, methods, and nested helpers."""

    callables: list[CallableDoc] = []

    def visit(nodes: list[ast.stmt], parents: tuple[str, ...] = ()) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                qualified = ".".join((*parents, node.name))
                callables.append(
                    CallableDoc(
                        name=node.name,
                        qualified_name=qualified,
                        kind="class",
                        signature=_constructor_signature(node),
                        docstring=ast.get_docstring(node, clean=True),
                        line=node.lineno,
                        depth=len(parents),
                    )
                )
                visit(node.body, (*parents, node.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = ".".join((*parents, node.name))
                kind = "method" if parents else "function"
                if len(parents) > 1:
                    kind = "nested helper"
                callables.append(
                    CallableDoc(
                        name=node.name,
                        qualified_name=qualified,
                        kind=kind,
                        signature=_signature(node),
                        docstring=ast.get_docstring(node, clean=True),
                        line=node.lineno,
                        depth=len(parents),
                    )
                )
                visit(node.body, (*parents, node.name))

    visit(tree.body)
    return callables


def _heading(title: str, marker: str) -> list[str]:
    """Return a reStructuredText heading."""

    return [title, marker * len(title), ""]


def _render_docstring(docstring: str) -> list[str]:
    """Render a docstring verbatim without reStructuredText reinterpretation.

    Existing research docstrings contain executable examples, wildcard
    arguments, underscore-suffixed function names, and historical indentation.
    A literal text block preserves that information and prevents Sphinx from
    treating scientific notation or identifiers as markup.
    """

    cleaned = inspect.cleandoc(docstring).strip()
    if not cleaned:
        cleaned = "Empty docstring."
    lines = [".. rubric:: Docstring", "", ".. code-block:: text", ""]
    lines.extend(f"   {line}" if line else "" for line in cleaned.splitlines())
    return lines


def _render_callable(item: CallableDoc, source_path: str) -> list[str]:
    """Render one callable as a readable Sphinx section."""

    title = f"``{item.qualified_name}`` ({item.kind})"
    source_url = f"{GITHUB_BASE}/{source_path}#L{item.line}"
    lines = _heading(title, "^")
    lines.extend(
        [
            f"`View source on GitHub <{source_url}>`__",
            "",
            ".. code-block:: python",
            "",
            f"   {item.signature}",
            "",
        ]
    )
    if item.docstring:
        lines.extend(_render_docstring(item.docstring))
    else:
        lines.append(".. warning:: Docstring pending.")
    lines.extend(["", ""])
    return lines


def _module_name(source_root: Path, path: Path) -> str:
    """Convert a Python path below ``source_root`` into an import-style name."""

    relative = path.relative_to(source_root.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _page_name(module_name: str) -> str:
    """Return a filesystem-safe page name for a module."""

    return module_name.replace(".", "-")


def generate(source_root: Path, output_dir: Path) -> dict[str, int]:
    """Generate one API page per module and an API index.

    Parameters
    ----------
    source_root : pathlib.Path
        Root of the ``O`` Python source tree.
    output_dir : pathlib.Path
        Destination for generated reStructuredText pages.

    Returns
    -------
    dict
        Counts for modules, total callables, documented callables, and pending
        docstrings.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    pages: list[tuple[str, str]] = []
    total = 0
    documented = 0

    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_name = _module_name(source_root, path)
        callables = _collect_callables(tree)
        source_path = path.relative_to(source_root.parent).as_posix()
        page_name = _page_name(module_name)
        pages.append((module_name, page_name))

        total += len(callables)
        documented += sum(item.docstring is not None for item in callables)

        title = module_name
        lines = [f".. _api-{page_name}:", "", *_heading(title, "=")]
        lines.extend(
            [
                f"`View module on GitHub <{GITHUB_BASE}/{source_path}>`__",
                "",
            ]
        )
        module_docstring = ast.get_docstring(tree, clean=True)
        if module_docstring:
            lines.extend([*_render_docstring(module_docstring), "", ""])
        else:
            lines.extend([".. warning:: Module docstring pending.", "", ""])

        if callables:
            lines.extend(_heading("Classes and functions", "-"))
            for item in callables:
                lines.extend(_render_callable(item, source_path))
        else:
            lines.append("This module defines no active Python classes or functions.")

        (output_dir / f"{page_name}.rst").write_text(
            "\n".join(lines).rstrip() + "\n", encoding="utf-8"
        )

    pending = total - documented
    index_lines = [
        "API reference",
        "=============",
        "",
        "These pages are generated directly from the active Python syntax tree.",
        "This avoids importing optional simulation and machine-learning stacks",
        "during a documentation-only build.",
        "",
        f"**Coverage:** {documented} of {total} active classes and functions have docstrings; "
        f"{pending} are marked as pending.",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "   :caption: Modules",
        "",
    ]
    index_lines.extend(f"   {module} <{page}>" for module, page in pages)
    (output_dir / "index.rst").write_text(
        "\n".join(index_lines).rstrip() + "\n", encoding="utf-8"
    )

    return {
        "modules": len(pages),
        "callables": total,
        "documented": documented,
        "pending": pending,
    }


if __name__ == "__main__":
    docs_dir = Path(__file__).resolve().parent
    repository_root = docs_dir.parent
    counts = generate(repository_root / "O", docs_dir / "api")
    print(
        "Generated {modules} modules: {documented}/{callables} documented, "
        "{pending} pending.".format(**counts)
    )
