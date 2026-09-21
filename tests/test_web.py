import http.client
import json
import threading

import pytest

from spikelab import config, web


@pytest.fixture
def server(tmp_path, tiny_patterns):
    cfg = config.resolve(tiny_patterns)
    cfg["out"] = str(tmp_path / "run")
    p = tmp_path / "c.toml"
    p.write_text(config.dumps(cfg))
    srv = web.make_server(str(p), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def req(srv, method, path, body=None, host=None):
    port = srv.server_address[1]
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request(method, path, body=body, headers={"Host": host or f"127.0.0.1:{port}"})
    r = c.getresponse()
    return r.status, r.read()


def test_binds_loopback_and_refuses_foreign_host(server):
    assert server.server_address[0] == "127.0.0.1"
    assert req(server, "GET", "/", host="evil.example")[0] == 403


def test_page_lists_registries_and_runs(server, tiny_patterns):
    code, body = req(server, "GET", "/api/state")
    state = json.loads(body)
    assert code == 200 and {"lif", "adlif"} <= set(state["registry"]["neuron"]["ok"])
    tiny_patterns["neuron"] = {"name": "adlif"}
    code, body = req(server, "POST", "/api/run", json.dumps(tiny_patterns))
    res = json.loads(body)
    assert code == 200, res
    assert 'name = "adlif"' in res["toml"]
    code, png = req(server, "GET", res["figures"][0])
    assert code == 200 and png[:4] == b"\x89PNG"


def test_bad_config_is_a_400_naming_it(server, tiny_patterns):
    tiny_patterns["rule"] = {"name": "nope"}
    code, body = req(server, "POST", "/api/run", json.dumps(tiny_patterns))
    assert code == 400 and "unknown rule 'nope'" in json.loads(body)["error"]
