"""One web run in its own process: `python -m spikelab.worker <config.toml> <out>`. Prints the result as JSON.

The server kills this process at the time limit; the memory limit is set here, before torch allocates anything.
"""
import json
import resource
import sys

from . import config, experiment, limits
from .registry import ComponentError


def main(path: str, out: str) -> int:
    resource.setrlimit(resource.RLIMIT_AS, (limits.MEMORY, limits.MEMORY))
    try:
        res = experiment.run(config.load(path), out)
    except ComponentError as e:
        print(e, file=sys.stderr)
        return 2
    except MemoryError:
        print(f"out of memory (limit {limits.MEMORY // 2**30} GB)", file=sys.stderr)
        return 3
    print(json.dumps({"final": res["final"], "seconds": res["seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
