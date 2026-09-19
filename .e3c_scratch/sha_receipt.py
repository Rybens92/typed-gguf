"""E3c scratch: the run's model-integrity receipt (hashes before and after the measurement).

The card's gates say "don't modify either model file (SHA before/after)". The probe pins
`model_sha256` in every run record *before* it loads the model; this script re-hashes both files
after the campaign and reports whether they are unchanged, next to E3's published digest for the
same file (`docs/evidence/e3_sha256_before.txt`, card `t_6d2e084d`) — the cross-card pin.

Writes `docs/evidence/e3c_cue_sha256_after.txt` and prints the same facts.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = {
    "4B": pathlib.Path("/var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf"),
    "occamy": pathlib.Path("/var/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf"),
}
RECORDS = {
    "4B": [ROOT / ".e3c" / "spark4b.json"],
    "occamy": sorted(glob for glob in (ROOT / ".e3c").glob("occamy_*.json")
                     if glob.name not in ("occamy.json", "occamy_dry.json")),
}
PUBLISHED = ROOT / "docs" / "evidence" / "e3_sha256_before.txt"


def digest(path: pathlib.Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            sha.update(block)
    return sha.hexdigest()


def recorded(paths: list[pathlib.Path]) -> set[str]:
    found: set[str] = set()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r'"model_sha256":\s*"([0-9a-f]{64})"', text):
            found.add(match.group(1))
    return found


def main() -> int:
    published: dict[str, str] = {}
    if PUBLISHED.exists():
        for line in PUBLISHED.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) == 2:
                published[fields[1]] = fields[0]
    lines = ["E3c model-integrity receipt (card t_6c119626)",
             "before = the digest every probe run recorded before loading",
             "after  = sha256 of the file on disk now (this file)", ""]
    receipt: dict[str, object] = {}
    for name, path in MODELS.items():
        after = digest(path)
        before = recorded(RECORDS[name])
        entry = {"path": str(path), "bytes": path.stat().st_size, "after": after,
                 "before": sorted(before), "identical": before == {after}}
        pinned = published.get(str(path))
        if pinned:
            entry["published"] = pinned
            entry["published_matches"] = pinned == after
        receipt[name] = entry
        lines.append(f"{name}: {path}")
        lines.append(f"  before: {', '.join(sorted(before)) or '(no record)'}")
        lines.append(f"  after : {after}")
        lines.append(f"  identical: {entry['identical']}")
        if pinned:
            lines.append(f"  E3's published digest: {pinned} "
                         f"(matches: {entry['published_matches']})")
        lines.append("")
    out = ROOT / "docs" / "evidence" / "e3c_cue_sha256_after.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "docs" / "evidence" / "e3c_cue_sha256_receipt.json").write_text(
        json.dumps(receipt, indent=1), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
