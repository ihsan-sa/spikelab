"""Public mode: every request needs a Cloudflare Access assertion, verified against a throwaway key served here."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from spikelab import cli, config, web
from spikelab.access import Access
from test_web import raw, req

TEAM, AUD, HOST = "testteam", "test-aud", "spike.example.com"
ISS = f"https://{TEAM}.cloudflareaccess.com"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER = rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def certs():
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(KEY.public_key())) | {"kid": "k1", "alg": "RS256"}
    body = json.dumps({"keys": [jwk]}).encode()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/certs"
    srv.shutdown()


@pytest.fixture
def public(tmp_path, tiny_patterns, certs):
    cfg = config.resolve(tiny_patterns)
    cfg["out"] = str(tmp_path / "run")
    p = tmp_path / "c.toml"
    p.write_text(config.dumps(cfg))
    srv = web.make_server(str(p), 0, HOST, Access(TEAM, AUD, ["Friend@Example.com", "me@example.com"], certs))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def token(key=KEY, kid="k1", **over):
    claims = {"aud": AUD, "iss": ISS, "email": "friend@example.com", "exp": int(time.time()) + 300} | over
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, key, algorithm="RS256",
                      headers={"kid": kid})


def get(srv, tok, path="/", host=HOST):
    return req(srv, "GET", path, host=host, headers={"Cf-Access-Jwt-Assertion": tok} if tok else {})


def test_good_token_is_let_in_everywhere(public, tiny_patterns):
    tok = token()
    assert get(public, tok)[0] == 200
    assert get(public, tok, "/api/state")[0] == 200
    assert get(public, tok, "/guide.pdf")[0] == 200
    assert get(public, tok, host=f"127.0.0.1:{public.server_address[1]}")[0] == 200
    code, body = req(public, "POST", "/api/run", json.dumps(tiny_patterns), host=HOST,
                     origin=f"https://{HOST}", headers={"Cf-Access-Jwt-Assertion": tok})
    assert code == 200, body
    for url in (json.loads(body)["figures"][0], json.loads(body)["weights"]):
        assert get(public, tok, url)[0] == 200
        assert get(public, None, url)[0] == 403
        assert get(public, token(email="stranger@example.com"), url)[0] == 403


@pytest.mark.parametrize("why, tok", [
    ("no assertion", None),
    ("garbage", "not.a.jwt"),
    ("bad signature", lambda: token(key=OTHER)),
    ("wrong aud", lambda: token(aud="another-app")),
    ("wrong iss", lambda: token(iss="https://evil.cloudflareaccess.com")),
    ("expired", lambda: token(exp=int(time.time()) - 60)),
    ("no exp", lambda: token(exp=None)),
    ("email not listed", lambda: token(email="stranger@example.com")),
    ("no email", lambda: token(email=None)),
    ("unknown kid", lambda: token(kid="k2")),
    ("hs256 with the public key", lambda: jwt.encode({"aud": AUD, "iss": ISS, "email": "friend@example.com",
                                                      "exp": int(time.time()) + 300}, "x" * 32, algorithm="HS256",
                                                     headers={"kid": "k1"})),
])
def test_everything_else_is_403_with_no_detail(public, tiny_patterns, why, tok):
    tok = tok() if callable(tok) else tok
    headers = {"Cf-Access-Jwt-Assertion": tok} if tok else {}
    for method, path, body in (("GET", "/", None), ("GET", "/api/state", None),
                               ("POST", "/api/run", json.dumps(tiny_patterns)), ("GET", "/runs/1/raster.png", None),
                               ("GET", "/runs/1/weights.json", None),
                               ("GET", "/guide.pdf", None)):
        code, out = req(public, method, path, body, host=HOST, headers=headers)
        assert (code, json.loads(out)) == (403, {"error": "forbidden"}), (why, path)


@pytest.mark.parametrize("method", ["HEAD", "OPTIONS", "PUT", "DELETE", "WHAT"])
def test_another_method_takes_the_access_check_too(public, method):
    """The base class used to answer these itself, before the gate, with its version in the header."""
    code, headers, body = raw(public, method, "/", headers={"Host": HOST})
    assert code == 403 and headers["Server"] == "spikelab" and b"Python" not in body
    code, headers, _ = raw(public, method, "/", headers={"Host": HOST, "Cf-Access-Jwt-Assertion": token()})
    assert code == 405 and headers["Server"] == "spikelab"


def test_public_mode_still_refuses_foreign_host_and_origin(public):
    tok = token()
    assert get(public, tok, host="evil.example")[0] == 403
    code, _ = req(public, "GET", "/", host=HOST, origin="https://evil.example",
                  headers={"Cf-Access-Jwt-Assertion": tok})
    assert code == 403
    assert public.server_address[0] == "127.0.0.1"


def test_local_mode_does_not_accept_the_public_host(tmp_path, tiny_patterns):
    p = tmp_path / "c.toml"
    p.write_text(config.dumps(config.resolve(tiny_patterns)))
    srv = web.make_server(str(p), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert get(srv, token())[0] == 403
    finally:
        srv.shutdown()


def test_cli_public_settings(tmp_path):
    ap = cli_args
    assert cli.public(ap()) == (None, None)
    with pytest.raises(ValueError, match="access_aud, allow_email"):
        cli.public(ap("--public-host", HOST, "--access-team", TEAM))
    f = tmp_path / "public.toml"
    f.write_text(f'public_host = "{HOST}"\naccess_team = "{TEAM}"\naccess_aud = "{AUD}"\nallow_email = ["a@x"]\n')
    host, acc = cli.public(ap("--public-config", str(f), "--public-host", "other.example"))
    assert host == "other.example" and acc.aud == AUD and acc.allow == {"a@x"}
    with pytest.raises(ValueError, match="bare team name"):
        cli.public(ap("--public-config", str(f), "--access-team", "evil.example/x?"))


def cli_args(*a):
    import argparse

    ns = argparse.Namespace(public_config=None, public_host=None, access_team=None, access_aud=None, allow_email=None)
    it = iter(a)
    for flag in it:
        k, v = flag[2:].replace("-", "_"), next(it)
        setattr(ns, k, [v] if k == "allow_email" else v)
    return ns
