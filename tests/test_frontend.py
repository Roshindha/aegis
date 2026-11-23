"""Static checks for the docs/ front end: design rules, assets, size, and WCAG AA contrast of the palette."""
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
CSS = (DOCS / "style.css").read_text()
HTML = (DOCS / "index.html").read_text()
JS = (DOCS / "app.js").read_text()


def test_no_shadows_blur_or_gradients():
    for banned in ("box-shadow", "text-shadow", "drop-shadow", "backdrop-filter", "gradient("):
        assert banned not in CSS, banned
        assert banned not in JS and banned not in HTML, banned
    assert not re.search(r"(?<![\w-])filter\s*:", CSS)


def test_no_generic_fonts_and_block_radius():
    assert "Inter" not in CSS and "Roboto" not in CSS and "Arial" not in CSS and "system-ui" not in CSS
    for r in re.findall(r"border-radius:\s*([^;]+);", CSS):
        assert r.strip() in {"2px", "3px", "4px", "50%", "0"}, r        # chips/inputs small; 50% only for the stage node


def test_fonts_and_assets_exist():
    for f in re.findall(r'url\("(fonts/[^"]+)"\)', CSS):
        assert (DOCS / f).exists(), f
    for ref in re.findall(r'(?:src|href)="([a-z][\w.-]*\.(?:js|css))"', HTML):
        assert (DOCS / ref).exists(), ref
    assert "font-display: swap" in CSS


def test_page_weight_under_150kb_excluding_fonts():
    total = sum((DOCS / n).stat().st_size for n in ("index.html", "style.css", "app.js", "config.js"))
    assert total < 150_000, total


def _vars(block: str) -> dict:
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9A-Fa-f]{6})", block))


def _theme_vars():
    light = _vars(CSS[CSS.index(":root {"):CSS.index(':root[data-theme="dark"]')])
    dark = {**light, **_vars(CSS[CSS.index(':root[data-theme="dark"]'):CSS.index("* { box-sizing")])}
    return light, dark


def _lum(hex_):
    c = [int(hex_[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_text_pairs_meet_wcag_aa():
    for name, v in zip(("light", "dark"), _theme_vars()):
        paper = v["paper"]
        for fg in ("ink", "muted", "pass-ink", "refuse-ink"):
            assert contrast(v[fg], paper) >= 4.5, (name, fg, contrast(v[fg], paper))
        assert contrast(paper, v["ink"]) >= 4.5, name                    # button: paper text on ink fill
        assert contrast(v["focus"], paper) >= 3, name                    # focus ring
        assert contrast(v["hl"], paper) >= 3, name                       # highlighted path
        assert contrast(v["ink"], v["paper"]) >= 7, name
