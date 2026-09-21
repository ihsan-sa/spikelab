"""What the web page may ask for. The CLI has no limits; every run from the page goes through check() and runner.

Every number is here. A resolved config over any cap is refused before anything runs, naming each cap it broke.
The wall-clock limit is the backstop for what the caps cannot see: the run happens in a child process that is
killed when it runs out of time, which also keeps a crash out of the server. The defaults keep a run inside
about a minute on a shared 6-core box (the two example configs take 12 to 16 s).
"""
from __future__ import annotations

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


def check(cfg: dict, caps: dict = CAPS) -> None:
    """Raise ComponentError naming every cap a RESOLVED config breaks."""
    over = []

    def cap(name, value, where):
        if isinstance(value, (int, float)) and value > caps[name]:
            over.append(f"{where} = {value} is over the {name} limit of {caps[name]}")

    arch, task, rule = cfg["architecture"], cfg["task"], cfg["rule"]
    sizes = [int(n) for n in arch.get("sizes", [])]
    for i, n in enumerate(sizes):
        cap("layer_size", n, f"architecture.sizes[{i}]")
    cap("weights", weights(sizes, bool(arch.get("recurrent"))), "total weights")
    for k in ("steps", "train", "test", "batch", "batches"):
        cap(k, task.get(k), f"task.{k}")
    cap("epochs", rule.get("epochs"), "rule.epochs")
    steps, batch = task.get("steps", 0), task.get("batch", 1)
    if isinstance(steps, int) and isinstance(batch, int):
        cap("activity", steps * batch * sum(sizes), "steps x batch x neurons")
    if over:
        raise ComponentError("; ".join(over))
