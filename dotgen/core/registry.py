"""Engine selection.

Phases 3-9 add a subclass in ``engines_real``; :func:`default_engines` names the
most advanced one, and that is what the app and the tests both run on -- testing
against stubs the user never sees would hide exactly the regressions this layer
exists to catch.  Everything else calls :func:`get_engines` per use -- never
caching the result -- so a swap takes effect immediately.
"""

from __future__ import annotations

from .engines import Engines, StubEngines


def default_engines() -> Engines:
    """The newest real engine.  Imported lazily to keep this module leaf-like."""
    from .engines_real import RealSeparateEngines

    return RealSeparateEngines()


_engines: Engines = default_engines()


def get_engines() -> Engines:
    return _engines


def set_engines(e: Engines) -> None:
    global _engines
    _engines = e


def reset_engines() -> None:
    set_engines(default_engines())


def use_stub_engines() -> None:
    """Force the fake data path -- for tests that assert on stub behaviour."""
    set_engines(StubEngines())
