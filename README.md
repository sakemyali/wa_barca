# wa_barca — More than a Hack 2027, phase 1

Per-frame ball possession (`H` / `A` / `D` / `X`) for FC Barcelona vs Athletic Club, scored on
`spellF1` by the organiser's scorer in `evaluation/` (vendored from
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027), MIT).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export MTAH_DATA_DIR=/path/to/data   # folder containing tactical/ and broadcast/
```

The match data is gated and never committed. `games.py` resolves every file path from
`MTAH_DATA_DIR`.

## Layout

```
games.py                 organiser constants: frame alignment, graded halves, asset paths
evaluation/              organiser scorer and validator (do not edit)
src/possession/          our code
tests/
```

## Score a submission

```bash
python evaluation/validate_submission.py evaluation/results/<team>.csv
python evaluation/run_evaluation.py            # public 5-minute window only
```
