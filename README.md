# wa_barca — More than a Hack 2027, phase 1: ball possession

One label per frame (`H` / `A` / `D` / `X`) for FC Barcelona vs Athletic Club, 199,319 frames,
ranked on `spellF1` by the organisers' scorer (vendored unchanged in `evaluation/`).

**How it works, in five lines.** Two gradient-boosted classifiers answer *is the ball in play?* and
*which team has it?* for every frame; three more detect the instant of a change (team change,
stoppage, restart); a three-state Viterbi decoder turns those into possessions. The classifiers are
trained only on **public tracking of other matches** (7 DFL Bundesliga games, 3 Metrica sample
games), after those matches have been degraded to look like the provided automatic tracking. No
label of the provided match is used anywhere, for training or for tuning.

The full account — what was measured, what was tried, what was dropped and why — is in
[`docs/TECHNICAL.md`](docs/TECHNICAL.md). Every experiment is logged in
[`experiments/README.md`](experiments/README.md).

## Reproduce the submission

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MTAH_DATA_DIR=/path/to/data          # the folder that contains tactical/
export PYTHONPATH=src

python scripts/predict.py                    # -> evaluation/results/wa_barca.csv   (30 s)
python evaluation/validate_submission.py evaluation/results/wa_barca.csv
```

`predict.py` uses the trained models committed under `models/`. It needs only the provided
tactical tracking (`BAR-ATH_tactical_atd.txt` and `.xml`); the video, the broadcast feed, the
homography and the Smart Tagging are not used.

## Rebuild everything from scratch

```bash
python scripts/prepare_external.py           # download + convert the 10 public matches (≈2.8 GB streamed, 350 MB kept)
python scripts/bootstrap.py                  # label-free first guess at the provided match's stoppages  (≈4 min)
python scripts/train.py                      # the five classifiers -> models/            (≈8 min, 8 cores)
python scripts/predict.py                    # the submission
```

Validation, on held-out public matches only:

```bash
python scripts/run_cv.py v3                  # 3-fold cross-validation, out-of-fold probabilities
python scripts/tune_decoder.py v3            # choose the decoder's settings by spellF1
python scripts/error_analysis.py v3          # where the possessions are lost
python scripts/baseline_check.py             # organisers' baseline on the simulated matches
python -m pytest -q                          # 29 unit tests, a few seconds
```

Seeds are fixed everywhere and LightGBM runs in deterministic mode, so a rebuild on the same
machine and library versions regenerates the same models and the same CSV.

## Layout

```
games.py                    organisers' constants: frame alignment, graded halves, asset paths
evaluation/                 organisers' scorer and validator (unchanged), public 5-minute ground truth
src/possession/
  atd.py tagging.py homography.py   loaders for the provided assets
  schema.py matches.py              one layout for every tracking source
  external/                         converters for the public matches (DFL, Metrica sample games)
  degrade.py                        make public tracking look like the provided tracking
  features.py                       per-frame features: players, ball, visibility
  kickoff.py                        opening-kickoff detector
  models.py                         LightGBM wrappers and feature views
  decode.py                         Viterbi decoder with event-aware transition costs
  metrics.py                        fast exact copy of spellF1 for tuning loops
  pipeline.py                       the steps, shared by training, validation and inference
scripts/                    the submission chain: prepare_external, bootstrap, train, predict
                            validation: run_cv, tune_decoder, error_analysis, ablate_decoder
                            checks: calibrate, baseline_check, domain_check; experiment: run_tcn
models/                     trained classifiers (LightGBM text files) and the decoder setting
experiments/                experiment log, calibration notes, decoder tuning results
docs/TECHNICAL.md           the technical document
tests/
```

## Data and licences

The match data is gated, licensed for this competition only, and never committed (`data/`,
`data_external/`, `*.npz`, `*.mp4` are git-ignored). The public training matches are downloaded by
`scripts/prepare_external.py` from their publishers:

- **DFL** — Bassek et al., *An integrated dataset of spatiotemporal and event data in elite
  soccer*, Scientific Data (2025), CC BY 4.0, <https://doi.org/10.6084/m9.figshare.28196177>
- **Metrica Sports sample data** — three anonymised matches,
  <https://github.com/metrica-sports/sample-data>

The scorer in `evaluation/` is the organisers' (MIT, see `evaluation/LICENSE-upstream`).
