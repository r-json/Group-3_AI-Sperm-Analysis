"""One-off provenance correction for the runs written on 2026-10-05 (documented in REPORT.md).

Before commit c61a69b ("fix(provenance)"), `_write_run` recorded the git commit at the
moment each run *finished*. Because code was being committed while training ran, some runs
recorded a later commit or a "-dirty" flag. The code that actually ran is HEAD when each training
process *started*, recovered here from the training log:

* HuSHeM, all models: process started 2026-10-05 17:55 at commit c865547 (clean tree)
* SMIDS linear probes: process started 2026-10-05 22:37 at commit cafbaff (clean tree)

`git diff c865547 cafbaff -- src/spermtriage/{training,models,data,config.py}` contains
only formatting, typing, an identical refactor and a `pretrained` flag that defaults to
True, so both commits train identically. This script adds `code_commit` and a note; it
leaves the originally recorded value in `git_commit_recorded_at_write`. It is idempotent.
"""

from __future__ import annotations

import json

from spermtriage.config import project_root

PROCESS_COMMIT = {
    ("hushem", None): "c865547ffce7fd754b1d01f645f8efd9448c890b",
    ("smids", "lp"): "cafbaff721b5976ce1c4ce114bd1ec619216e7ef",
}


def main() -> None:
    root = project_root() / "results" / "runs" / "main"
    for prov_path in sorted(root.glob("*/*/fold*/provenance.json")):
        prov = json.loads(prov_path.read_text())
        if "code_commit" in prov:
            continue
        dataset, model_id = prov_path.parts[-4], prov_path.parts[-3]
        key = ("hushem", None) if dataset == "hushem" else ("smids", model_id.split("-")[-1])
        if key not in PROCESS_COMMIT:
            continue
        prov["git_commit_recorded_at_write"] = prov["git_commit"]
        prov["code_commit"] = PROCESS_COMMIT[key]
        prov["git_commit"] = PROCESS_COMMIT[key]
        prov["provenance_note"] = (
            "code_commit = HEAD when the training process started (see "
            "scripts/annotate_provenance_2026_10_05.py and REPORT.md)."
        )
        prov_path.write_text(json.dumps(prov, indent=2, default=str))
        print("annotated", prov_path.relative_to(root))


if __name__ == "__main__":
    main()
