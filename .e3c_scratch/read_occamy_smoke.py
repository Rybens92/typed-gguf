"""E3c scratch: the Occamy smoke — where the readout sits and what the model does there."""
import json
import pathlib

record = json.loads(pathlib.Path(".e3c/occamy_smoke.json").read_text())
item = record["items"][0]
print("prefix_tokens", item["prefix_tokens"], "n_ctx", item["n_ctx"],
      "n_seq_max", item["n_seq_max"])
print("prefix_tail", repr(item["prefix_tail"]))
print("opener", {k: repr(v) for k, v in item["opener"].items()})
for name, shape in item["shapes"].items():
    readout = shape["readout"]
    print("\n==", name, "row", readout["row"], "at", readout["position"])
    if "advance" in shape:
        print("   advance:", shape["advance"])
    print("   top:", [(t["piece"], round(t["p_full"], 6)) for t in readout["top_tokens"]])
    print("   closers:", {k: round(v, 6) for k, v in readout["closer_mass"].items()})
    print("   cue:", readout["cue"])
    for variant, block in readout["labels"].items():
        print("   labels", variant, block["texts"], [f"{m:.3g}" for m in block["first_mass"]],
              "coverage", f"{block['coverage']:.3g}")
for key, ranked in item["ranked"].items():
    print("\nranked", key, "got", ranked["got"], "correct", ranked["correct"],
          "top_coverage", ranked["top_coverage"], "agrees", ranked["agrees"],
          "low_mass" if ranked["reliability"] == "low_mass" else ranked["reliability"])
