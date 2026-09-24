"""E3c scratch: what the smoke run measured (the shape of the numbers behind the report)."""
import json
import pathlib

path = pathlib.Path(".e3c/smoke.json")
record = json.loads(path.read_text())
item = record["items"][0]
print("prefix_tokens", item["prefix_tokens"], "n_ctx", item["n_ctx"],
      "n_seq_max", item["n_seq_max"])
print("prefix_tail", repr(item["prefix_tail"]))
print("backend", record["backend_claim"], record["backend_source"])
print("device_log_tail", repr(item["device_log_tail"])[:400])
for name, shape in item["shapes"].items():
    readout = shape["readout"]
    print("\n==", name, "opener", repr(item["opener"][name]), "readout row", readout["row"],
          "at", readout["position"])
    print("   top:", [(t["piece"], round(t["p_full"], 4)) for t in readout["top_tokens"]])
    print("   closers:", {k: round(v, 6) for k, v in readout["closer_mass"].items()})
    print("   cue:", readout["cue"])
    for variant, block in readout["labels"].items():
        print("   labels", variant, block["texts"], [round(m, 5) for m in block["first_mass"]],
              "coverage", f"{block['coverage']:.6f}")
    if "advance" in shape:
        print("   advance:", shape["advance"])
