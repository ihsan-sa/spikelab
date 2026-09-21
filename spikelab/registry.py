"""Registries for the swappable parts.

Each kind (neuron, synapse, topology, architecture, rule, task) has its own table.
A component registers with `@register(kind, name)`. On registration it is checked
against its kind's contract (a tiny smoke run). A component that fails is kept in
`broken` with the reason, by name, and is not usable; everything else still is.
Plugin files (config key `plugins`) are imported the same way: an import error marks
that file broken and nothing else.
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

KINDS = ("neuron", "synapse", "topology", "architecture", "rule", "task")

_ok: dict[str, dict[str, type]] = {k: {} for k in KINDS}
broken: dict[str, dict[str, str]] = {k: {} for k in (*KINDS, "module")}  # "module": files that failed to import
contracts: dict[str, object] = {}  # kind -> callable(cls) raising on failure


class ComponentError(Exception):
    """A component named in a config is unknown, broken or badly configured."""


def _short(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def register(kind: str, name: str):
    """Class decorator. Validates the class against its kind's contract."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")

    def deco(cls):
        cls.kind, cls.name = kind, name
        try:
            if not isinstance(getattr(cls, "defaults", None), dict):
                raise TypeError("class needs a `defaults` dict")
            check = contracts.get(kind)
            if check is not None:
                check(cls)
        except Exception as e:  # a broken component is reported, never raised
            broken[kind][name] = _short(e)
            _ok[kind].pop(name, None)
            return cls
        broken[kind].pop(name, None)
        _ok[kind][name] = cls
        return cls

    return deco


def names(kind: str) -> list[str]:
    return sorted(_ok[kind])


def get(kind: str, name: str) -> type:
    if name in _ok[kind]:
        return _ok[kind][name]
    if name in broken[kind]:
        raise ComponentError(f"{kind} {name!r} is broken: {broken[kind][name]}")
    raise ComponentError(f"unknown {kind} {name!r}; known: {', '.join(names(kind)) or 'none'}")


def params_for(kind: str, name: str, given: dict) -> dict:
    """Defaults overlaid with `given`; an unknown key is an error that names the component."""
    cls = get(kind, name)
    extra = set(given) - set(cls.defaults)
    if extra:
        raise ComponentError(f"{kind} {name!r} has no parameter(s) {sorted(extra)}; it takes {sorted(cls.defaults)}")
    return {**cls.defaults, **given}


def describe() -> dict:
    """Everything the interface needs: healthy components with defaults, broken ones with reasons."""
    return {
        k: {
            "ok": {n: {"defaults": c.defaults, "doc": (c.__doc__ or "").strip().split("\n")[0]}
                   | ({"drive": c.drive} if hasattr(c, "drive") else {}) for n, c in sorted(_ok[k].items())},
            "broken": dict(broken[k]),
        }
        for k in KINDS
    } | {"module": {"ok": {}, "broken": dict(broken["module"])}}


_loaded = False
BUILTIN_MODULES = ("neurons", "synapses", "topology", "architecture", "rules", "tasks")


def load_builtins() -> None:
    """Import the built-in component modules, each on its own so one failure is contained."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    from . import contracts as _c  # noqa: F401  (fills `contracts`)

    for mod in BUILTIN_MODULES:
        try:
            importlib.import_module(f"spikelab.{mod}")
        except Exception as e:
            broken["module"][mod] = _short(e)


def load_plugin(path: str | Path) -> str | None:
    """Import a plugin file that registers components. Returns an error string, or None."""
    load_builtins()
    path = Path(path)
    try:
        spec = importlib.util.spec_from_file_location(f"spikelab_plugin_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        msg = _short(e)
        broken["module"][str(path)] = msg
        return msg
    return None
