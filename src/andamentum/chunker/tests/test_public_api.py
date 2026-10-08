"""Pin the public surface of andamentum.chunker."""

import ast
import pathlib

import andamentum.chunker as chunker


def test_public_all():
    assert set(chunker.__all__) == {"Chunk", "ChunkerError", "chunk"}


def test_public_imports_are_resolvable():
    for name in chunker.__all__:
        assert hasattr(chunker, name), f"chunker.{name} is missing"


def test_chunker_imports_only_standard_library_and_itself():
    import sys

    root = pathlib.Path(chunker.__file__).parent
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                mods = [node.module or ""]
            else:
                continue
            for mod in mods:
                top = mod.split(".")[0]
                assert top in sys.stdlib_module_names or top == "andamentum", (
                    f"{path.name} imports {mod}"
                )
                if top == "andamentum":
                    assert path.name == "cli.py" and mod == "andamentum"
