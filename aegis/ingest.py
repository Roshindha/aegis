"""Turn the IM8 Reform OSCAL catalogue into control-aware chunks.

One control = one chunk. Controls are short (35-160 words), so we never split
them; instead we attach rich metadata so retrieval can filter and cite.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path("data/raw/tech-standards")
CATALOG = REPO / "catalogs" / "im8-reform.json"
PROFILES = REPO / "profiles"
SOURCE_URL = "https://github.com/GovTechSG/tech-standards"

LEVEL_NAMES = {0: "must-have", 1: "should-have", 2: "good-to-have"}

# OSCAL placeholders look like: {{ insert: param, tp-4_prm_2 }}
PARAM_RE = re.compile(r"\{\{\s*insert:\s*param,\s*([\w-]+)\s*\}\}")
# OSCAL cross-links look like: [SP 800-63b](#e59c5a7c-8b1f-...) -- keep the label, drop the UUID
LINK_RE = re.compile(r"\[([^\]]+)\]\(#[^)]*\)")


@dataclass
class Reference:
    framework: str   # e.g. "NIST SP 800-53", "MVSP", "IM8 (legacy)"
    ref_id: str      # e.g. "AC-2(3)", "2.7", "1.7/S1c"
    raw: str         # original text, kept for display


@dataclass
class Chunk:
    chunk_id: str
    control_id: str
    title: str
    group_id: str
    group_title: str
    statement: str
    guidance: str
    risk_statement: str
    levels: dict[str, int]
    references: list[Reference] = field(default_factory=list)
    published: str = ""
    last_modified: str = ""
    catalog_version: str = ""
    source_url: str = SOURCE_URL
    text: str = ""   # what gets embedded and keyword-indexed


# ---------- small helpers ----------

def _render(part: dict) -> str:
    """A part's own prose plus any nested list items, e.g. NS-6's four certificate rules."""
    lines = [LINK_RE.sub(r"\1", part.get("prose", ""))]
    for item in part.get("parts", []):
        label = next((p["value"] for p in item.get("props", []) if p["name"] == "label"), "-")
        lines.append(f"  ({label}) {_render(item)}")
    return "\n".join(lines)


def _part(control: dict, name: str) -> str:
    for part in control.get("parts", []):
        if part["name"] == name:
            return _render(part)
    return ""


def _prop(control: dict, name: str) -> str:
    for prop in control.get("props", []):
        if prop["name"] == name:
            return prop["value"]
    return ""


def resolve_params(text: str, control: dict) -> str:
    """Replace OSCAL placeholders with a readable label.

    No profile in this release sets parameter values, so the honest rendering is
    "[agency-defined: <label>]" rather than inventing a number.
    """
    labels = {p["id"]: p.get("label", "value") for p in control.get("params", [])}
    return PARAM_RE.sub(lambda m: f"[agency-defined: {labels.get(m.group(1), 'value')}]", text)


def parse_reference(raw: str) -> Reference:
    """Normalise messy reference strings into (framework, ref_id)."""
    head = raw.split(":")[0].strip()
    tail = raw.split(":", 1)[1].strip() if ":" in raw else ""

    if head.startswith("IM8 ") or head == "Data Protection (Data Security)":
        return Reference("IM8 (legacy)", f"{head.removeprefix('IM8 ')} {tail}".strip(), raw)
    if m := re.match(r"[A-Z]{2}\.[A-Z]{2}-\d+", head):         # check CSF first: "PR.AA-01"
        return Reference("NIST CSF 2.0", m.group(0), raw)
    if m := re.search(r"(?:^|\s)([A-Z]{2}-\d+(?:\(\d+\))?)$", head):
        return Reference("NIST SP 800-53", m.group(1), raw)   # "AC-2", "NIST SP 800-53 IA-2(1)"
    if m := re.match(r"MVSP (\d+\.\d+)", head):
        return Reference("MVSP", m.group(1), raw)
    if m := re.match(r"SLSA (Build L\d)", head):
        return Reference("SLSA", m.group(1), raw)
    if m := re.match(r"CIS Critical Security Controls v8 (\d+\.\d+)", head):
        return Reference("CIS Controls v8", m.group(1), raw)
    if m := re.match(r"Cloud Controls Matrix ([A-Z]{3}-\d+)", head):
        return Reference("CCM", m.group(1), raw)
    if head.startswith(("NIST", "SP 800")):
        return Reference("NIST (other)", head, raw)
    return Reference("Other", head, raw)


def load_profile_levels() -> dict[str, dict[str, int]]:
    levels: dict[str, dict[str, int]] = {}
    for path in sorted(PROFILES.glob("*.json")):
        risk, level = path.stem.rsplit("-level-", 1)
        profile = json.loads(path.read_text())["profile"]
        for imp in profile["imports"]:
            for inc in imp.get("include-controls", []):
                for cid in inc.get("with-ids", []):
                    levels.setdefault(cid, {})[risk] = int(level)
    return levels


def describe_levels(levels: dict[str, int]) -> str:
    if not levels:
        return "Applicability: no profile level assigned in this catalogue release."
    parts = [f"{risk} systems: Level {lvl} ({LEVEL_NAMES[lvl]})" for risk, lvl in sorted(levels.items())]
    return "Applicability: " + "; ".join(parts) + "."


def build_text(c: Chunk) -> str:
    """The text we embed. A contextual header (ID, title, group) helps both
    keyword search (exact IDs) and semantic search (topic words)."""
    return "\n".join([
        f"{c.control_id}: {c.title}",
        f"Group: {c.group_title}",
        describe_levels(c.levels),
        f"Statement: {c.statement}",
        f"Guidance: {c.guidance}",
        f"Risk: {c.risk_statement}",
    ])


# ---------- main entry point ----------

def build_chunks() -> list[Chunk]:
    catalog = json.loads(CATALOG.read_text())["catalog"]
    version = catalog["metadata"]["version"]
    levels = load_profile_levels()
    chunks: list[Chunk] = []

    for group in catalog["groups"]:
        for ctl in group.get("controls", []):
            chunk = Chunk(
                chunk_id=f"im8-{version}-{ctl['id']}",
                control_id=ctl["id"].upper(),
                title=ctl["title"],
                group_id=group["id"],
                group_title=group["title"],
                statement=resolve_params(_part(ctl, "statement"), ctl),
                guidance=resolve_params(_part(ctl, "guidance"), ctl),
                risk_statement=_prop(ctl, "risk-statement"),
                levels=levels.get(ctl["id"], {}),
                references=[parse_reference(l["text"]) for l in ctl.get("links", [])],
                published=_prop(ctl, "published"),
                last_modified=_prop(ctl, "last-modified"),
                catalog_version=version,
            )
            chunk.text = build_text(chunk)
            chunks.append(chunk)
    return chunks


def save_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")