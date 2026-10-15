"""Self-contained static dashboard: out/dashboard.html (inline CSS/JS/SVG, embedded fonts, no network)."""

from __future__ import annotations

import base64
import json
import re
import urllib.request
from pathlib import Path

from . import config as C

TEMPLATE = Path(__file__).with_name("dashboard_template.html")

_FONT_FACES = [
    # (file, family, weight)
    ("switzer-300.woff2", "Switzer", 300),
    ("switzer-400.woff2", "Switzer", 400),
    ("switzer-500.woff2", "Switzer", 500),
    ("switzer-600.woff2", "Switzer", 600),
    ("geistmono-latin.woff2", "Geist Mono", "400 600"),
]


def fetch_fonts() -> list[Path]:
    """Download the fonts once (Switzer: Fontshare, ITF FFL; Geist Mono: Google Fonts, OFL)."""
    C.FONTS_DIR.mkdir(parents=True, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"}

    def get(url: str) -> bytes:
        return urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=30).read()

    css = get("https://api.fontshare.com/v2/css?f[]=switzer@300,400,500,600&display=swap").decode()
    written = []
    for block in css.split("@font-face")[1:]:
        w = re.search(r"font-weight:\s*(\d+)", block)
        u = re.search(r"url\('(//[^']+\.woff2)'\)", block)
        if w and u:
            p = C.FONTS_DIR / f"switzer-{w.group(1)}.woff2"
            p.write_bytes(get("https:" + u.group(1)))
            written.append(p)
    css = get("https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600&display=swap").decode()
    m = re.search(r"/\* latin \*/.*?url\((https://[^)]+\.woff2)\)", css, re.S)
    if m:
        p = C.FONTS_DIR / "geistmono-latin.woff2"
        p.write_bytes(get(m.group(1)))
        written.append(p)
    return written


def _font_css() -> str:
    faces = []
    for fname, family, weight in _FONT_FACES:
        p = C.FONTS_DIR / fname
        if not p.exists():
            continue
        b64 = base64.b64encode(p.read_bytes()).decode()
        faces.append(f"@font-face{{font-family:'{family}';src:url(data:font/woff2;base64,{b64}) format('woff2');"
                     f"font-weight:{weight};font-style:normal;font-display:block}}")
    if not faces:
        print("[dashboard] fonts not found in assets/fonts (run `python run.py fonts`); using system fallbacks")
    return "\n".join(faces)


def _slim_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        g = r["guardrail"]
        out.append({
            "id": r["item_id"], "q": r["question"], "a": r["answer"], "k": r["knowledge"], "gold": r["gold"],
            "v": {k: v["verdict"] for k, v in r["configs"].items()},
            "lat": {k: round(v["latency_s"], 2) for k, v in r["configs"].items()},
            "g": {
                "mv": g["model_verdict"], "ov": g["overridden"], "rule": g["rule"], "fired": g["rules_fired"],
                "detail": g["detail"],
                "spans": [{"t": s["text"], "ok": s["verbatim"], "s": s["start"], "e": s["end"]} for s in g["spans"]],
                "facts": [{"t": f["text"], "kind": f["kind"], "s": f["start"], "e": f["end"], "ok": f["covered"]}
                          for f in g["facts"]],
                "raw": r["configs"]["haiku_guardrail"]["raw_text"],
            },
        })
    return out


def write_dashboard(rows: list[dict], summary: dict) -> Path:
    payload = {"summary": summary, "rows": _slim_rows(rows),
               "configs": [{"key": c.key, "label": c.label, "model": c.model, "mode": c.mode} for c in C.configs()]}
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__FONTS__*/", _font_css()).replace("__DATA__", data)
    C.OUT_DIR.mkdir(exist_ok=True)
    path = C.OUT_DIR / "dashboard.html"
    path.write_text(html)
    return path
