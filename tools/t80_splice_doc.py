"""Splice the generated artifacts into the evidence document's placeholders (card t_80f1a4c6)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/evidence/e3_fix_t_80f1a4c6_serving_backend.md"
RAW = ROOT / "docs/evidence/e2e/t_80f1a4c6-serving-backend"

doc = DOC.read_text()
before_after = RAW.joinpath("before_after.txt").read_text().rstrip()
replay = RAW.joinpath("replay_survivors.txt").read_text().rstrip()

block = "```\n" + before_after + "\n```"
assert "<!-- @@BEFORE_AFTER@@ -->" in doc
doc = doc.replace("<!-- @@BEFORE_AFTER@@ -->", block)

replay_block = "```\n" + replay + "\n```"
assert "<!-- @@REPLAY@@ -->" in doc
doc = doc.replace("<!-- @@REPLAY@@ -->", replay_block)

DOC.write_text(doc)
print("spliced before/after and replay into the doc")
print("remaining placeholders:", [line for line in doc.splitlines() if "@@" in line])
