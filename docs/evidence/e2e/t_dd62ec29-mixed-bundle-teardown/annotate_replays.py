"""Append the two-direction note to the replay artifact (harness proof, per the triage reference)."""
import pathlib

path = pathlib.Path(".e2e/t_dd62ec29-mixed-bundle-teardown/logs/survivor_replays.txt")
path.write_text(path.read_text(encoding="utf-8") + (
    "\nTwo directions are proven above: the six keys the triage's pins were written for are now\n"
    "KILLED (they were r2 survivors), and `x__child_verdict__mutmut_120` —\n"
    "`report.get(\"ok\", False)`, equivalent because every report carries the key — SURVIVES, so the\n"
    "harness is not simply failing everything it touches.\n"))
print(path.read_text(encoding="utf-8")[-400:])
