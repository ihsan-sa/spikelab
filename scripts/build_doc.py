"""Screenshot the web page and render docs/GUIDE.md to docs/spikelab-guide.pdf with headless Chrome.

Only the markdown GUIDE.md uses is converted: headings, paragraphs, lists, code blocks, images,
inline code, bold and links.
"""
import html
import re
import subprocess
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CHROME = "google-chrome"


def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)


def to_html(md):
    out, para, in_list, in_code = [], [], False, False

    def flush():
        nonlocal para, in_list
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para = []
        if in_list:
            out.append("</ul>")
            in_list = False

    for line in md.splitlines():
        if line.startswith("```"):
            if not in_code:
                flush()
            out.append("<pre><code>" if not in_code else "</code></pre>")
            in_code = not in_code
        elif in_code:
            out.append(html.escape(line))
        elif m := re.match(r"(#+) (.*)", line):
            flush()
            n = len(m[1])
            out.append(f"<h{n}>{inline(m[2])}</h{n}>")
        elif line.startswith("- "):
            if para:
                out.append(f"<p>{inline(' '.join(para))}</p>")
                para = []
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
        elif not line.strip():
            flush()
        else:
            para.append(line)
    flush()
    return "\n".join(out)


CSS = """body{font:11pt/1.5 'DejaVu Sans',sans-serif;color:#1f2937;max-width:760px;margin:auto}
h1{font-size:20pt}h2{font-size:14pt;margin-top:1.4em;border-bottom:1px solid #e5e7eb}
code{background:#f3f4f6;padding:0 .2em;font-size:.92em}pre{background:#f3f4f6;padding:.6em}
img{max-width:100%;display:block;margin:.6em auto;page-break-inside:avoid}p,li{text-align:left}"""


def screenshot():
    from spikelab import web

    srv = web.make_server(str(ROOT / "configs" / "surrogate.toml"), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([CHROME, "--headless", "--disable-gpu", f"--user-data-dir={prof}", "--hide-scrollbars",
                        "--window-size=1100,760", "--virtual-time-budget=4000",
                        f"--screenshot={DOCS / 'img' / 'web.png'}", url], check=True, capture_output=True, timeout=120)
    srv.shutdown()


def pdf():
    page = DOCS / "_guide.html"
    page.write_text(f"<!doctype html><meta charset=utf-8><style>{CSS}</style>{to_html((DOCS / 'GUIDE.md').read_text())}")
    try:
        with tempfile.TemporaryDirectory() as prof:
            subprocess.run([CHROME, "--headless", "--disable-gpu", f"--user-data-dir={prof}", "--no-pdf-header-footer",
                            f"--print-to-pdf={DOCS / 'spikelab-guide.pdf'}", page.as_uri()],
                           check=True, capture_output=True, timeout=120)
    finally:
        page.unlink()


if __name__ == "__main__":
    screenshot()
    pdf()
    print("wrote", DOCS / "img" / "web.png", "and", DOCS / "spikelab-guide.pdf")
