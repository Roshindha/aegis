"""Step 1: explore the IM8 Reform OSCAL catalogue before designing chunks."""
import json
from collections import Counter
from pathlib import Path

REPO = Path("data/raw/tech-standards")
CATALOG = REPO / "catalogs" / "im8-reform.json"
PROFILES = REPO / "profiles"


def load_profile_levels() -> dict[str, dict[str, int]]:
    """Return {control_id: {"low-risk": 1, "medium-risk": 0, ...}}."""
    levels: dict[str, dict[str, int]] = {}
    for path in sorted(PROFILES.glob("*.json")):
        risk, level = path.stem.rsplit("-level-", 1)   # "low-risk-level-1" -> ("low-risk", "1")
        profile = json.loads(path.read_text())["profile"]
        for imp in profile["imports"]:                  # loop over BOTH imports
            for inc in imp.get("include-controls", []):
                for cid in inc.get("with-ids", []):
                    levels.setdefault(cid, {})[risk] = int(level)
    return levels


def get_part(control: dict, name: str) -> str:
    for part in control.get("parts", []):
        if part["name"] == name:
            return part.get("prose", "")
    return ""          # lined up with `for`, NOT inside the loop


def get_prop(control: dict, name: str) -> str:
    for prop in control.get("props", []):
        if prop["name"] == name:
            return prop["value"]
    return ""


def main() -> None:
    catalog = json.loads(CATALOG.read_text())["catalog"]
    levels = load_profile_levels()
    meta = catalog["metadata"]
    print(f"{meta['title']}  version={meta['version']}\n")

    total = 0
    ref_sources = Counter()
    for group in catalog["groups"]:
        controls = group.get("controls", [])
        total += len(controls)
        print(f"{group['id']:3} {group['title']:45} {len(controls):3} controls")
        for c in controls:
            for link in c.get("links", []):
                ref_sources[link["text"].split(":")[0].split(" ")[0]] += 1

    print(f"\nTotal controls: {total}")
    print(f"Controls with a profile level: {len(levels)}")
    print(f"Reference sources: {ref_sources.most_common()}\n")

    # Show one full control, the way a chunk might look
    sample = catalog["groups"][0]["controls"][0]
    print("--- sample control ---")
    print(f"ID:        {sample['id'].upper()}")
    print(f"Title:     {sample['title']}")
    print(f"Levels:    {levels.get(sample['id'])}")
    print(f"Statement: {get_part(sample, 'statement')}")
    print(f"Guidance:  {get_part(sample, 'guidance')[:200]}...")
    print(f"Risk:      {get_prop(sample, 'risk-statement')[:200]}...")
    print(f"Refs:      {[l['text'] for l in sample.get('links', [])]}")


if __name__ == "__main__":
    main()