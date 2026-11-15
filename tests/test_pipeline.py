import os

import numpy as np
import pytest

from possession import pipeline
from possession.schema import D, H, A, X


def test_event_targets_mark_each_kind_of_change_and_respect_halftime():
    label = np.array([H] * 10 + [A] * 10 + [D] * 10 + [H] * 10 + [A] * 10, np.int8)
    period = np.array([1] * 40 + [2] * 10, np.int8)
    ev = pipeline.event_targets(label, period)
    w = pipeline.EVENT_HALF_WIDTH
    assert ev[10 - w:10 + w + 1, 0].all() and ev[:, 0].sum() == 2 * w + 1      # one team change
    assert ev[20 - w:20 + w + 1, 1].all() and ev[:, 1].sum() == 2 * w + 1      # one stoppage
    assert ev[30 - w:30 + w + 1, 2].all() and ev[:, 2].sum() == 2 * w + 1      # one restart
    # the H -> A step at frame 40 is the halftime boundary, not an event
    assert not ev[38:43].any()


def test_submission_writer_and_scorer_round_trip(tmp_path):
    codes = np.array([X, H, H, A, D, X], np.int8)
    path = pipeline.write_submission(str(tmp_path / "s.csv"), codes)
    table = pipeline.pm.read_label_csv(path)
    assert pipeline.pm.validate_submission(table, 6) is None
    assert pipeline.pm.label_codes(table).tolist() == codes.tolist()


def test_perfect_prediction_scores_one_and_halves_are_scored_separately():
    table = {"label": np.array([H] * 50 + [D] * 20 + [A] * 50 + [A] * 40 + [H] * 40, np.int8),
             "period": np.array([1] * 120 + [2] * 80, np.int8)}
    gt, cut = pipeline.table_gt(table)
    r = pipeline.score(gt, pipeline.with_gap(table["label"], cut))
    assert r["spell_f1"] == 1.0 and r["macro_f1"] == 1.0
    assert r["n_gt_spells"] == 4            # the A run that straddles halftime counts twice


@pytest.mark.skipif(not os.path.isfile(os.path.join(pipeline.ROOT, "models", "team.txt")), reason="no trained models")
def test_committed_models_match_the_feature_list():
    import json
    from possession import models
    with open(os.path.join(pipeline.ROOT, "models", "features.json")) as f:
        names = json.load(f)
    team = models.load(os.path.join(pipeline.ROOT, "models", "team.txt"))
    alive = models.load(os.path.join(pipeline.ROOT, "models", "alive.txt"))
    assert team.num_feature() == len(models.team_columns(names))
    assert alive.num_feature() == len(models.alive_columns(names))
