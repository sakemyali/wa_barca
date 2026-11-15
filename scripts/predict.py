"""Label the provided match and write the submission.

    python scripts/predict.py [out.csv]

Uses the trained models in models/ and the decoder setting in models/decoder.json. Frames outside
the two halves are X (never scored); every frame inside them gets H, A or D.
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import games  # noqa: E402
from possession import decode as dec  # noqa: E402
from possession import matches, models, pipeline  # noqa: E402
from possession.schema import X  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models")
DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "wa_barca.csv")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def load_models():
    alive = models.load(os.path.join(MODEL_DIR, "alive.txt"))
    team = models.load(os.path.join(MODEL_DIR, "team.txt"))
    events = {k: models.load(os.path.join(MODEL_DIR, "event_{}.txt".format(k))) for k in pipeline.EVENTS}
    with open(os.path.join(MODEL_DIR, "decoder.json")) as f:
        params = dec.DecoderParams(*json.load(f)["params"])
    return alive, team, events, params


def predict(prov=None):
    """(codes [n_video_frames], probabilities dict, rows) for the provided match."""
    prov = prov or matches.provided()
    tab = pipeline.tables([prov], bank=None, swaps=(False,))[0]
    with open(os.path.join(MODEL_DIR, "features.json")) as f:
        assert json.load(f) == tab["names"], "feature list differs from the one the models were trained on"
    alive, team, events, params = load_models()
    p_dead, p_home = pipeline.infer(tab, alive, team)
    p_events = pipeline.infer_events(tab, events)
    pre = pipeline.before_kickoff(prov, tab, p_dead)
    p_dead = np.where(pre, 1.0, p_dead)
    path = dec.decode(p_dead, p_home, pipeline.blocks_of(tab["period"]), params, p_events=p_events)
    codes = np.full(prov.n_frames, X, np.int8)
    codes[tab["rows"]] = path
    probs = {"p_dead": p_dead, "p_home": p_home, "p_events": p_events, "pre_kickoff": pre}
    return codes, probs, tab["rows"]


def main(out=DEFAULT_OUT):
    t0 = time.time()
    codes, probs, rows = predict()
    print("frames forced to D before an opening kickoff: {}".format(int(probs["pre_kickoff"].sum())))
    pipeline.write_submission(out, codes)
    counts = {k: int((codes == i).sum()) for i, k in enumerate("HADX")}
    print("wrote {} ({} rows, {}) in {:.0f}s".format(out, len(codes), counts, time.time() - t0))
    # Pipeline check against the public five minutes. A check, not a tuning target: nothing in
    # this repository reads this number back.
    gt = pipeline.pm.label_codes(pipeline.pm.read_label_csv(PUBLIC_GT))
    r = pipeline.score(gt, codes)
    print("public 5-minute window (sanity check only): spellF1 {spell_f1:.4f} macroF1 {macro_f1:.4f} "
          "possessions {n_pred_spells} vs {n_gt_spells}".format(**r))


if __name__ == "__main__":
    main(*sys.argv[1:2])
