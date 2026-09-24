"""E3c scratch: verify the probe read the row *after* each shape's opener (not the shipped cue).

If the openers were not in the prompt, every 'at the cue' shape would report the same readout
position and the same top token; the card's numbers depend on the opposite.
"""
import json
import pathlib

record = json.loads(pathlib.Path(".e3c/occamy_dry.json").read_text())
for item in record["items"]:
    print("item", item["id"], "prefix_tokens", item["prefix_tokens"])
    for name, block in item["shapes"].items():
        readout = block["readout"]
        top = readout["top_tokens"][0]
        print(f"  {name:22s} opener={item['opener'][name]!r:30s} "
              f"suffix_tokens={block['suffix_tokens']:3d} readout_at={readout['position']:4d} "
              f"({readout['row']}) top={top['piece']!r} p={top['p_full']:.6f}")
