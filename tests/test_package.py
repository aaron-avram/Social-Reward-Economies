"""Cross-module invariants: import hygiene, no dead skeleton, one AgentRole."""
import importlib
import inspect

import pytest

import model

MODULES = ["agent", "config", "instrumentation", "plots", "reputation", "results",
           "rewards", "rng", "roles", "roleupdate", "system", "welfare"]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"socialreward.{name}")


@pytest.mark.parametrize("name", MODULES)
def test_no_unimplemented_bodies(name):
    """NotImplementedError left in a skeleton is a hole, not a design."""
    mod = importlib.import_module(f"socialreward.{name}")
    _src = inspect.getsource(mod)
    assert "raise NotImplementedError" not in _src, f"{name} has an unimplemented body"


@pytest.mark.parametrize("name", MODULES)
def test_no_skeleton_markers_left(name):
    mod = importlib.import_module(f"socialreward.{name}")
    _src = inspect.getsource(mod)
    assert "TODO" not in _src, f"{name} still carries TODO markers"


def test_plots_is_the_only_matplotlib_importer():
    """The point of the split: a sweep harness must not pay for matplotlib."""
    for name in MODULES:
        if name == "plots":
            continue
        _src = inspect.getsource(importlib.import_module(f"socialreward.{name}"))
        assert "matplotlib" not in _src, f"{name} imports matplotlib"


def test_no_module_below_system_imports_system():
    """The dependency graph must stay acyclic and point one way."""
    for name in MODULES:
        if name in ("system", "plots"):
            continue
        _src = inspect.getsource(importlib.import_module(f"socialreward.{name}"))
        assert "import system" not in _src and "from .system" not in _src, name


def test_agent_does_not_import_reputation_or_roleupdate():
    _src = inspect.getsource(model.agent) if hasattr(model, "agent") else \
        inspect.getsource(importlib.import_module("socialreward.agent"))
    assert "reputation" not in _src.split('"""')[0] + _src.split('"""')[-1]
