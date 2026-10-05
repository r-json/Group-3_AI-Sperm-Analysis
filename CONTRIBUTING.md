# Contributing

## Development setup

```bash
git clone https://github.com/r-json/sperm-morphology-triage.git
cd sperm-morphology-triage
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[gui,data,dev]"
pre-commit install
```

## Before opening a pull request

```bash
ruff check src tests scripts && ruff format --check src tests scripts
mypy
QT_QPA_PLATFORM=offscreen pytest --cov
```

CI runs the same checks on Python 3.11-3.13. The tests use synthetic fixtures only: they
need neither the datasets nor network access.

## Rules for research changes

* **No hand-written numbers.** Every number in the README, docs or manuscript must come from
  a file under `results/` that `spermtriage report` generated, with its run id and commit.
  If a run did not produce a number, write `TODO`.
* **Protocol changes need a new ADR** in `docs/adr/` and a new split version (`v2`).
  Examples: splits, deduplication, the role of each subset.
* **New experiments go in a new YAML** under `configs/experiments/`, never as edits to
  `main.yaml` after its results are published.
* **Commit before training.** Runs record the commit checked out when training starts; a
  dirty tree is flagged.
* Use conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
