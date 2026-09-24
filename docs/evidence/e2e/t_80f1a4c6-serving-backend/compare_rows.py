"""Show the answer-row shape of a serving response and match the AFTER row inside BEFORE."""
import json
import pathlib
import sys

before = json.loads(pathlib.Path(sys.argv[1]).read_text())
after = json.loads(pathlib.Path(sys.argv[2]).read_text())


def rows(payload):
    answers = payload["answers"]
    return list(answers.items()) if isinstance(answers, dict) else list(enumerate(answers))


print("BEFORE rows:", len(rows(before)), " AFTER rows:", len(rows(after)))
label, first = rows(before)[0]
print("BEFORE row key:", label, "keys:", sorted(first.keys()))
print("BEFORE row:", json.dumps({k: first[k] for k in list(first)[:6]})[:600])
label2, second = rows(after)[0]
print("AFTER row key:", label2, "keys:", sorted(second.keys()))
print("AFTER row:", json.dumps({k: second[k] for k in list(second)[:6]})[:600])
