"""What the web page may ask for. The CLI has no limits; every run from the page goes through check() and runner.

Every number is here. A resolved config over any cap is refused before anything runs, naming each cap it broke.
The wall-clock limit is the backstop for what the caps cannot see: the run happens in a child process that is
killed when it runs out of time, which also keeps a crash out of the server. The defaults keep a run inside
about a minute on a shared 6-core box (the two example configs take 12 to 16 s).
"""
from __future__ import annotations

import math

from .registry import ComponentError

CAPS = {
    "layer_size": 256,        # any entry of architecture.sizes
    "weights": 20_000,        # feed-forward plus recurrent weights, all layers
    "steps": 1000,            # task.steps (time steps per sample)
    "train": 1024,            # task.train samples
    "test": 512,              # task.test samples
    "batch": 128,             # task.batch
    "batches": 100,           # task.batches (tasks that draw batches instead of a fixed train set)
    "epochs": 30,             # rule.epochs
    "activity": 2_000_000,    # steps x batch x neurons: the size of one batch's recorded state
}
SECONDS = 60                  # wall clock per run, then the run is killed
MEMORY = 3 * 2**30            # address space of the run's process, bytes


def weights(sizes: list[int], recurrent: bool) -> int:
    ff = sum(a * b for a, b in zip(sizes, sizes[1:]))
    return ff + (sum(n * n for n in sizes[1:-1]) if recurrent else 0)


def _short(value) -> str:
    """A value quoted back at whoever sent it, short enough to be safe in an error."""
    s = repr(value)
    return s if len(s) <= 40 else s[:40] + "..."


def _whole(value) -> bool:
    """A real whole number. True is an int to Python and not a count to anyone else."""
    return isinstance(value, int) and not isinstance(value, bool)


def check(cfg: dict, caps: dict = CAPS) -> None:
    """Raise ComponentError naming every cap a RESOLVED config breaks, and every value that cannot be capped.

    Type first, by name: a count must be a whole number (1000.0, "1000" and True are refused), because a
    float or a string compares wrong or not at all against a cap, and the cap would pass a run it should stop.
    """
    over = []

    def cap(name, value, where):
        if value > caps[name]:
            over.append(f"{where} = {value} is over the {name} limit of {caps[name]}")

    def count(name, value, where):
        """A capped count: a real positive int, under its cap. Returns it, or None if it was refused."""
        if not _whole(value):
            over.append(f"{where} must be a whole number, got {_short(value)}")
            return None
        if value < 1:
            over.append(f"{where} = {value} must be 1 or more")
            return None
        cap(name, value, where)
        return value

    dt = cfg.get("dt")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        over.append(f"dt must be a positive number, got {_short(dt)}")

    arch, task, rule = cfg["architecture"], cfg["task"], cfg["rule"]
    given = arch.get("sizes", [])
    if not isinstance(given, list):
        over.append(f"architecture.sizes must be a list of whole numbers, got {_short(given)}")
        given = []
    sizes = [count("layer_size", n, f"architecture.sizes[{i}]") for i, n in enumerate(given)]
    for k in ("steps", "train", "test", "batch", "batches"):
        if k in task:
            count(k, task[k], f"task.{k}")
    if "epochs" in rule:
        count("epochs", rule["epochs"], "rule.epochs")
    if all(n is not None for n in sizes):  # derived caps run on checked values only
        cap("weights", weights(sizes, bool(arch.get("recurrent"))), "total weights")
        steps, batch = task.get("steps", 0), task.get("batch", 1)
        if _whole(steps) and _whole(batch):
            cap("activity", steps * batch * sum(sizes), "steps x batch x neurons")
    if over:
        raise ComponentError("; ".join(over))
