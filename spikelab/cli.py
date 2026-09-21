"""spikelab run <config> | spikelab serve [config] | spikelab list"""
from __future__ import annotations

import argparse
import json
import sys

from . import config, registry
from .registry import ComponentError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="spikelab", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a config and write figures to its output folder")
    r.add_argument("config")
    r.add_argument("--out", help="output folder (default: the config's `out`)")
    s = sub.add_parser("serve", help="web page on 127.0.0.1 over the same configs")
    s.add_argument("config", nargs="?", default="configs/surrogate.toml", help="config the page starts from")
    s.add_argument("--port", type=int, default=8765)
    sub.add_parser("list", help="show every registered component, and any broken one with its reason")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        registry.load_builtins()
        print(json.dumps(registry.describe(), indent=1))
        return 0
    if a.cmd == "serve":
        from . import web

        web.serve(a.config, a.port)
        return 0
    from . import experiment

    try:
        res = experiment.run(config.load(a.config), a.out)
    except ComponentError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"final": res["final"], "seconds": res["seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
