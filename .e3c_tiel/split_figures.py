#!/usr/bin/env python3
"""§5.1 figures: the same 20 items read by the bench shape and the serving shape."""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path("/var/home/rybens/workspace/ggufone")
sys.path.insert(0, str(ROOT / "src"))

from ggufone.bench import compare  # noqa: E402


def main() -> int:
    quality = json.loads((ROOT / "docs" / "evidence" / "tiel_quality.json").read_text("utf-8"))
    qrows = {str(r.get("id")): r for r in compare.rows_of(quality)}
    batch = json.loads((ROOT / ".e3c_tiel" / "batch_response.json").read_text("utf-8"))["answers"]
    ids = sorted(batch)
    qcov = [float(qrows[i].get("coverage") or 0.0) for i in ids]
    bcov = [float(batch[i].get("coverage") or 0.0) for i in ids]
    qlow = [i for i in ids if qrows[i].get("reliability") == "low_mass"]
    blow = [i for i in ids if batch[i].get("reliability") == "low_mass"]
    print(f"ids {ids[0]}..{ids[-1]} (n={len(ids)})")
    print(f"bench shape:   measured {len(ids) - len(qlow)}/{len(ids)} · low_mass {qlow} · "
          f"coverage min {min(qcov):.3e} median {statistics.median(qcov):.3e} max {max(qcov):.3e}")
    print(f"serving shape: measured {len(ids) - len(blow)}/{len(ids)} · low_mass {len(blow)}/{len(ids)}"
          f" · coverage min {min(bcov):.3e} median {statistics.median(bcov):.3e} max {max(bcov):.3e}")
    qtok: dict[str, int] = {}
    for i in ids:
        token = str((qrows[i].get("cue") or {}).get("token"))
        qtok[token] = qtok.get(token, 0) + 1
    print(f"bench shape cue tokens: {qtok}")
    btok: dict[str, int] = {}
    for i in ids:
        token = str((batch[i].get("cue") or {}).get("token"))
        btok[token] = btok.get(token, 0) + 1
    print(f"serving shape cue tokens: {btok}")
    print("\n| id | bench coverage / verdict | serving coverage / verdict | serving cue token (mass) |")
    for i in ids:
        q, b = qrows[i], batch[i]
        cue = b.get("cue") or {}
        print(f"| {i} | {float(q.get('coverage') or 0):.3e} / {q.get('reliability')} | "
              f"{float(b.get('coverage') or 0):.3e} / {b.get('reliability')} | "
              f"`{cue.get('token')}` ({cue.get('mass'):.2f}) |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
