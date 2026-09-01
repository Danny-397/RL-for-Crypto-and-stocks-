"""Every tool must be runnable the way its own docstring says to run it.

``tools/hyperparameter_sweep.py`` shipped with a sweep the site had a panel for,
a summariser, and passing tests — and had never once been executed. It imports
``tools.real_significance``, and running ``python tools/hyperparameter_sweep.py``
puts ``tools/`` on the path but not the repository root, so the import raised
``ModuleNotFoundError`` before argparse ever ran. Under pytest it worked, because
pytest puts the root on the path. The tests therefore exercised every function in
the file while the entry point was broken.

This checks the thing pytest cannot notice about itself: a module that imports a
sibling under the ``tools.`` package must arrange for the root to be importable
first.
"""

from __future__ import annotations

import ast
import io
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")

TOOL_FILES = sorted(
    f for f in os.listdir(TOOLS)
    if f.endswith(".py") and not f.startswith("_")
)


def _source(name: str) -> str:
    with io.open(os.path.join(TOOLS, name), encoding="utf-8") as fh:
        return fh.read()


def test_there_are_tools_to_check():
    assert len(TOOL_FILES) > 10, "the tools directory looks wrong"


@pytest.mark.parametrize("name", TOOL_FILES)
def test_a_tool_importing_a_sibling_puts_the_repo_root_on_the_path(name: str):
    """The import must be reachable when the file is run as a script."""
    source = _source(name)
    tree = ast.parse(source, filename=name)

    first_tools_import = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tools"):
            if first_tools_import is None or node.lineno < first_tools_import:
                first_tools_import = node.lineno
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("tools"):
                    if first_tools_import is None or node.lineno < first_tools_import:
                        first_tools_import = node.lineno

    if first_tools_import is None:
        pytest.skip(f"{name} imports no sibling tool")

    # Any of the shapes used in this repo count, as long as one runs first.
    lines = source.replace("\r\n", "\n").split("\n")[:first_tools_import - 1]
    prelude = "\n".join(lines)
    fixed = ("sys.path.insert" in prelude) or ("sys.path.append" in prelude)
    assert fixed, (
        f"{name} imports a sibling tool at line {first_tools_import} but never "
        "puts the repository root on sys.path, so `python tools/"
        f"{name}` raises ModuleNotFoundError. Tests will not catch this: pytest "
        "puts the root on the path for you."
    )


@pytest.mark.parametrize("name", TOOL_FILES)
def test_a_tool_that_inserts_a_path_imports_sys_first(name: str):
    """A path fix that itself raises NameError is not a fix."""
    source = _source(name).replace("\r\n", "\n")
    if "sys.path.insert" not in source and "sys.path.append" not in source:
        pytest.skip(f"{name} does not touch sys.path")
    insert_at = min(
        i for i, line in enumerate(source.split("\n"))
        if "sys.path.insert" in line or "sys.path.append" in line
    )
    before = "\n".join(source.split("\n")[:insert_at])
    assert "import sys" in before, f"{name} uses sys.path before importing sys"
