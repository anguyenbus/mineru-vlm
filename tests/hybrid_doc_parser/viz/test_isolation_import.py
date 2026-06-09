"""``isolation_import`` (view.md §9): the HARD isolation rule, CI-enforced.

The runtime library MUST import with ZERO ``viz`` optional deps installed, and
``import hybrid_doc_parser`` must NOT pull in any ``viz`` module. The pure viz
modules (``coords``/``html``/``normalize``) must themselves stay importable even
when ``pypdfium2``/``pdfplumber``/``PIL`` are absent — their heavy imports are
deferred into function bodies.

This runs in the FAST suite: it imports nothing heavy. The absence of the viz
deps is SIMULATED with a meta-path finder that raises ``ModuleNotFoundError`` for
``pypdfium2``/``pdfplumber``/``PIL`` (the deps may well be installed in the dev
env), so the test asserts the deferred-import contract regardless of what is on
disk. sys.modules and sys.meta_path are restored on teardown.
"""

from __future__ import annotations

import importlib
import sys

import pytest

# The three ``viz`` optional deps that must NOT be required to import the library
# or the pure viz modules.
_VIZ_DEPS = ("pypdfium2", "pdfplumber", "PIL")

_PURE_VIZ_MODULES = (
    "hybrid_doc_parser.viz.coords",
    "hybrid_doc_parser.viz.html",
    "hybrid_doc_parser.viz.normalize",
)


class _BlockFinder:
    """Meta-path finder that makes the named top-level packages unimportable."""

    def __init__(self, blocked: tuple[str, ...]) -> None:
        self._blocked = set(blocked)

    def find_spec(self, name, path=None, target=None):  # noqa: ARG002, ANN001
        if name.split(".")[0] in self._blocked:
            raise ModuleNotFoundError(f"blocked for isolation test: {name}")
        return None  # defer to the rest of the meta path for everything else


@pytest.fixture
def deps_absent():
    """Make the viz deps unimportable and evict cached modules; restore after."""
    saved_modules = dict(sys.modules)
    saved_meta = list(sys.meta_path)

    # Evict any already-imported viz deps + the package under test so the blocker
    # is consulted on the fresh re-import.
    for mod in list(sys.modules):
        top = mod.split(".")[0]
        if top in _VIZ_DEPS or mod.startswith("hybrid_doc_parser"):
            del sys.modules[mod]

    sys.meta_path.insert(0, _BlockFinder(_VIZ_DEPS))
    try:
        yield
    finally:
        sys.meta_path[:] = saved_meta
        sys.modules.clear()
        sys.modules.update(saved_modules)


def test_viz_deps_are_actually_blocked(deps_absent) -> None:
    """Sanity: the blocker really prevents importing each viz dep."""
    for dep in _VIZ_DEPS:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(dep)


def test_library_imports_with_zero_viz_deps(deps_absent) -> None:
    """``import hybrid_doc_parser`` succeeds with pypdfium2/pdfplumber/PIL absent."""
    mod = importlib.import_module("hybrid_doc_parser")
    assert mod is not None


def test_library_import_pulls_in_no_viz_module(deps_absent) -> None:
    """The library never imports ``viz`` — nothing under viz is loaded."""
    importlib.import_module("hybrid_doc_parser")
    leaked = [m for m in sys.modules if m.startswith("hybrid_doc_parser.viz")]
    assert leaked == [], f"library leaked viz modules: {leaked}"


@pytest.mark.parametrize("module", _PURE_VIZ_MODULES)
def test_pure_viz_module_imports_with_deps_absent(deps_absent, module: str) -> None:
    """Pure viz modules import even with the heavy deps blocked (deferred import)."""
    mod = importlib.import_module(module)
    assert mod is not None


def test_render_module_imports_with_deps_absent(deps_absent) -> None:
    """``viz.render`` imports too (its pypdfium2/PIL imports are deferred)."""
    mod = importlib.import_module("hybrid_doc_parser.viz.render")
    # The pure page-selection helper is usable without any rasterizer.
    assert mod.select_pages(3)[0] == [0, 1, 2]
