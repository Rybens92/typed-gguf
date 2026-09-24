import json

s = json.load(open('.e3c_tiel/tiel_label_sweep.json'))
print("sweep top keys:", list(s.keys()))
for k in ('aggregate', 'summary', 'notes', 'policy', 'placement', 'wall_s', 'n_items', 'meta'):
    if k in s:
        print(" ", k, "=", json.dumps(s[k])[:500])
rows = s.get('rows') or s.get('cells') or s.get('items') or []
print("rows:", len(rows))
if rows:
    print(" row0:", json.dumps(rows[0])[:400])

b = json.load(open('.e3c_tiel/batch_response.json'))
print("=== batch top keys:", list(b.keys()))
for k, v in b.items():
    if k in ('answers', 'response', 'questions'):
        print(" ", k, "->", json.dumps(v)[:400])
    else:
        print(" ", k, "=", json.dumps(v)[:300])
