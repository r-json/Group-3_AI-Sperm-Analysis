# Superseded documents (kept for the record)

These files describe the project as it was at commit `fa4ada5` (July 2025). They are
**superseded** and several of their claims were found to be unsupported (see
[`docs/research/01_audit.md`](../research/01_audit.md)): dataset sizes, results for models
that were never run, and features that were never implemented. Do not cite them.

| File | Status |
| --- | --- |
| `PROJECT_SUMMARY.md` | Superseded by the README, model card and `results/main/tables.md` |
| `GITHUB_SETUP.md`, `COMMIT_INSTRUCTIONS.md`, `MANUAL_COMMIT_GUIDE.md`, `auto_commit.bat`, `auto_commit.ps1` | Obsolete helper material for the original upload |
| `verify_setup.py` | Checked the legacy file layout; replaced by CI |

The legacy notebook, PyQt5 GUI and committed dataset copies were removed from the default
branch. They remain in git history at commit `fa4ada5` (`Finish.ipynb`, `Interface/`,
`HuSHeM-*/`, `SMIDS-*/`). Parts of the notebook and GUI were adapted from an unlicensed
third-party project; see [`NOTICE.md`](../../NOTICE.md).
