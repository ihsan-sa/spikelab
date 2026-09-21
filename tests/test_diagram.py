"""The network diagram, in a real headless Chrome driven over its DevTools pipe (no port), and its weights endpoint."""
import fcntl
import json
import os
import shutil
import subprocess
import tempfile
import time

import pytest

from spikelab import config
from test_web import req, server  # noqa: F401  (the fixture)


class Chrome:
    """Just enough of the DevTools protocol over --remote-debugging-pipe (fd 3 in, fd 4 out) to evaluate JS."""

    def __init__(self):
        self.prof = tempfile.mkdtemp()
        to_r, self.to_w = os.pipe()
        self.from_r, from_w = os.pipe()
        # lift the child's ends above 4 first, so dup2 onto 3 and 4 cannot clobber one of them
        lo = (to_r, from_w)
        to_r, from_w = (fcntl.fcntl(fd, fcntl.F_DUPFD, 10) for fd in lo)
        for fd in lo:
            os.close(fd)

        def fds():  # the child's fd 3 reads commands, fd 4 writes replies
            os.dup2(to_r, 3)
            os.dup2(from_w, 4)

        self.p = subprocess.Popen(["google-chrome", "--headless", "--disable-gpu", "--no-first-run",
                                   f"--user-data-dir={self.prof}", "--remote-debugging-pipe", "about:blank"],
                                  preexec_fn=fds, close_fds=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.close(to_r)
        os.close(from_w)
        self.buf, self.n, self.session = b"", 0, None

    def send(self, method, check=True, **params):
        self.n += 1
        msg = {"id": self.n, "method": method, "params": params} | ({"sessionId": self.session} if self.session else {})
        os.write(self.to_w, json.dumps(msg).encode() + b"\0")
        while True:
            while b"\0" not in self.buf:
                chunk = os.read(self.from_r, 1 << 16)
                assert chunk, "chrome closed the pipe"
                self.buf += chunk
            line, self.buf = self.buf.split(b"\0", 1)
            r = json.loads(line)
            if r.get("id") == self.n:
                assert not check or "error" not in r, r
                return r.get("result")

    def open(self, url):
        target = self.send("Target.createTarget", url=url)["targetId"]
        self.session = self.send("Target.attachToTarget", targetId=target, flatten=True)["sessionId"]

    def js(self, expr):
        r = self.send("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        assert "exceptionDetails" not in r, r
        return r["result"].get("value")

    def close(self):
        self.p.kill()
        self.p.wait()
        shutil.rmtree(self.prof, ignore_errors=True)


@pytest.fixture
def page(server):  # noqa: F811
    c = Chrome()
    c.open(f"http://127.0.0.1:{server.server_address[1]}/")
    for _ in range(100):  # the first evaluations can land on about:blank, or in a page mid-load
        r = c.send("Runtime.evaluate", check=False, expression="!!document.querySelector('#sketch .node')", returnByValue=True)
        if r and r["result"].get("value"):
            break
        time.sleep(0.1)
    yield c
    c.close()


def counts(page, **fields):
    """Set architecture/topology fields as a user would (value, then an input event) and count what is drawn."""
    for key, value in fields.items():
        page.js(f"""(()=>{{const i=document.querySelector('#p-architecture input[data-key="{key}"],#p-topology input[data-key="{key}"]');
            if(i.type==='checkbox') i.checked={json.dumps(value)}; else i.value={json.dumps(str(value) if not isinstance(value, list) else json.dumps(value))};
            i.dispatchEvent(new Event('input',{{bubbles:true}}));}})()""")
    return page.js("""({nodes:document.querySelectorAll('#sketch .node').length, edges:document.querySelectorAll('#sketch .edge').length,
        rec:document.querySelectorAll('#sketch .rec').length, more:[...document.querySelectorAll('#sketch .more')].map(t=>t.textContent),
        text:document.getElementById('sketch').textContent})""")


def select(page, kind, name):
    page.js(f"(()=>{{const s=document.getElementById('s-{kind}'); s.value='{name}'; s.dispatchEvent(new Event('change',{{bubbles:true}}));}})()")


def test_diagram_follows_sizes_topology_and_recurrence(page):
    c = counts(page, sizes=[10, 8, 2])
    assert (c["nodes"], c["edges"], c["rec"]) == (20, 10 * 8 + 8 * 2, 0)
    assert "input 10" in c["text"] and "spikes" in c["text"] and "lif" in c["text"] and "delta" in c["text"]
    assert "rule surrogate: trains all weights" in c["text"]
    c = counts(page, sizes=[40, 64, 4])  # past 12 a layer shows its first and last 5 and a marker
    assert c["nodes"] == 10 + 10 + 4 and c["more"] == ["… +30", "… +54"] and "hidden 64" in c["text"]
    assert c["edges"] == 10 * 10 + 10 * 4
    assert counts(page, recurrent=True)["rec"] == 10 * 9 // 2
    select(page, "topology", "sparse")
    edges = [counts(page, p=p)["edges"] for p in (0.1, 0.5, 0.9)]
    assert edges[0] < edges[1] < edges[2] < 10 * 10 + 10 * 4 + 45
    assert "sparse p=0.5" in counts(page, p=0.5)["text"]
    assert counts(page, recurrent=False)["rec"] == 0


def test_diagram_marks_a_current_input_and_the_stdp_caption(page):
    select(page, "task", "current")
    select(page, "rule", "stdp")
    c = counts(page, sizes=[10, 3])
    assert "current" in c["text"] and "spikes" not in c["text"]
    assert "rule stdp: trains feed-forward weights only" in c["text"]


def test_weights_endpoint_respects_the_mask(server, tiny_patterns):  # noqa: F811
    tiny_patterns["topology"] = {"name": "sparse", "p": 0.3}
    tiny_patterns["architecture"]["recurrent"] = True
    code, body = req(server, "POST", "/api/run", json.dumps(tiny_patterns))
    assert code == 200, body
    url = json.loads(body)["weights"]
    code, body = req(server, "GET", url)
    assert code == 200
    layers = json.loads(body)["layers"]
    assert [(len(L["W"]), len(L["W"][0])) for L in layers] == [(10, 8), (8, 2)]
    assert layers[1]["V"] is None and len(layers[0]["V"]) == 8
    _, net, *_ = config.build(config.load(server.app.root / "1" / "config.toml"))  # same seed, same mask
    for L, got in zip(net.layers, layers):
        assert [[w is not None for w in row] for row in got["W"]] == L.mask.tolist()
    assert not any(got for got in (layers[0]["V"][i][i] for i in range(8)))
    assert req(server, "GET", "/runs/9/weights.json")[0] == 404


def test_learned_weights_toggle_draws_the_runs_weights(page):
    page.js("document.getElementById('go').click()")
    page.js("new Promise(ok=>{const t=setInterval(()=>{if(!document.getElementById('learned').disabled){clearInterval(t);ok();}},100);})")
    page.js("(()=>{const l=document.getElementById('learned'); l.checked=true; l.dispatchEvent(new Event('change',{bubbles:true}));})()")
    page.js("new Promise(ok=>{const t=setInterval(()=>{if(document.getElementById('sketch').textContent.includes('learned weights:')){clearInterval(t);ok();}},50);})")
    colours = page.js("[...new Set([...document.querySelectorAll('#sketch .edge')].map(e=>e.getAttribute('stroke')))]")
    assert set(colours) <= {"#2563eb", "#dc2626"} and colours  # coloured by sign, dense: every shown pair
    assert counts(page)["edges"] == 10 * 8 + 8 * 2
    counts(page, recurrent=True)  # changing a setting goes back to the sketch
    assert not page.js("document.getElementById('learned').checked")
