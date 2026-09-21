"""spikelab run <config> | spikelab serve [config] | spikelab list

serve is local (127.0.0.1, no sign-in) unless a public host is given, from flags or from --public-config
(a TOML with public_host, access_team, access_aud, allow_email = [...]; flags win). Public mode needs all four.
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib

from . import config, registry
from .registry import ComponentError


def public(a):
    """(public_host, Access) from the flags and --public-config, or (None, None) for local mode."""
    keys = ("public_host", "access_team", "access_aud", "allow_email")
    got = {}
    if a.public_config:
        with open(a.public_config, "rb") as f:
            got = tomllib.load(f)
        if set(got) - set(keys):
            raise ValueError(f"unknown key(s) {sorted(set(got) - set(keys))} in {a.public_config}")
    got |= {k: getattr(a, k) for k in keys if getattr(a, k)}
    if not got:
        return None, None
    if missing := [k for k in keys if not got.get(k)]:
        raise ValueError(f"missing {', '.join(missing)}")
    from .access import Access

    return got["public_host"], Access(got["access_team"], got["access_aud"], list(got["allow_email"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="spikelab", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a config and write figures to its output folder")
    r.add_argument("config")
    r.add_argument("--out", help="output folder (default: the config's `out`)")
    s = sub.add_parser("serve", help="web page on 127.0.0.1 over the same configs")
    s.add_argument("config", nargs="?", default="configs/surrogate.toml", help="config the page starts from")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--public-config", help="TOML with the public-mode keys below (see public.example.toml)")
    s.add_argument("--public-host", help="public mode: the hostname Cloudflare Access serves this page on")
    s.add_argument("--access-team", help="public mode: the Cloudflare Zero Trust team name")
    s.add_argument("--access-aud", help="public mode: the Access application's AUD tag")
    s.add_argument("--allow-email", action="append", help="public mode: an email allowed in (repeat for more)")
    sub.add_parser("list", help="show every registered component, and any broken one with its reason")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        registry.load_builtins()
        print(json.dumps(registry.describe(), indent=1))
        return 0
    if a.cmd == "serve":
        from . import web

        try:
            host, access = public(a)
        except ValueError as e:
            print(f"public mode: {e}", file=sys.stderr)
            return 2
        web.serve(a.config, a.port, host, access)
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
