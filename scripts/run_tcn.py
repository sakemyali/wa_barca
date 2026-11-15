"""Experiment: a temporal convolutional network on top of (or instead of) the tree models.

    python scripts/run_tcn.py [tag] [inputs=stack|features|probs] [oof=v3] [epochs=8] [folds=3]

  inputs=features  the TCN sees the engineered per-frame features only (a pure temporal network)
  inputs=probs     it sees only the tree models' out-of-fold probabilities (a learned smoother)
  inputs=stack     it sees both (the hybrid)

Same three folds as run_cv.py. Writes experiments/oof_<tag>.npz in the format tune_decoder.py
reads, so the network's output is decoded and scored exactly like the trees'.

PyTorch is needed for this script only. It is not a dependency of the submission pipeline.
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
extra = os.environ.get("MTAH_TORCH_PATH")
if extra:
    sys.path.insert(0, extra)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import games  # noqa: E402
from possession import matches, models, pipeline  # noqa: E402
from run_cv import FOLDS  # noqa: E402
from tune_decoder import load_oof  # noqa: E402

CACHE = os.path.join(games.DATA_DIR, ".cache")
CROP = 2000
BATCH = 12


class Block(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        pad = 2 * dilation
        self.c1 = nn.Conv1d(channels, channels, 5, padding=pad, dilation=dilation)
        self.c2 = nn.Conv1d(channels, channels, 5, padding=pad, dilation=dilation)
        self.n1 = nn.GroupNorm(8, channels)
        self.n2 = nn.GroupNorm(8, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.c1(F.gelu(self.n1(x)))
        y = self.c2(self.drop(F.gelu(self.n2(y))))
        return x + y


class TCN(nn.Module):
    """Non-causal dilated residual network, receptive field about +/- 10 s."""

    def __init__(self, n_in, channels=64, dilations=(1, 2, 4, 8, 16, 32, 64), dropout=0.1):
        super().__init__()
        self.inp = nn.Conv1d(n_in, channels, 1)
        self.blocks = nn.ModuleList([Block(channels, d, dropout) for d in dilations])
        self.out = nn.Conv1d(channels, 3, 1)

    def forward(self, x):                      # x [batch, time, features]
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        return self.out(F.gelu(h)).transpose(1, 2)


def logit(p):
    p = np.clip(p.astype(np.float32), 1e-4, 1 - 1e-4)
    return np.log(p) - np.log1p(-p)


def make_inputs(tab, probs, cols, mode):
    """Per-frame input matrix [n, d] (float16) for one table."""
    parts = []
    if mode in ("stack", "features"):
        parts.append(tab["X"][:, cols])
    if mode in ("stack", "probs"):
        p_dead, p_home, p_ev = probs
        parts.append(np.column_stack([logit(p_dead), logit(p_home), logit(p_ev)]).astype(np.float16))
    return np.concatenate(parts, 1)


def standardiser(arrays):
    sample = np.concatenate([a[::25] for a in arrays]).astype(np.float32)
    mean = np.nanmean(sample, 0)
    std = np.nanstd(sample, 0)
    return mean, np.where(std > 1e-6, std, 1.0)


def prepare(a, mean, std):
    x = (a.astype(np.float32) - mean) / std
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -8, 8)


def halves(period):
    return [(s, e) for _, s, e in pipeline.blocks_of(period)]


def train_tcn(train_items, n_in, epochs, seed):
    """train_items: [(inputs float16 [n, d], label, period, weight)]."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean, std = standardiser([it[0] for it in train_items])
    net = TCN(n_in)
    segs = [(i, s, e) for i, it in enumerate(train_items) for s, e in halves(it[2]) if e - s > CROP]
    steps = epochs * sum(e - s for _, s, e in segs) // (CROP * BATCH)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    net.train()
    t0, running = time.time(), 0.0
    for step in range(steps):
        xs, ys, ws = [], [], []
        for _ in range(BATCH):
            i, s, e = segs[rng.integers(len(segs))]
            a = int(rng.integers(s, e - CROP))
            inp, lab, _, w = train_items[i]
            xs.append(prepare(inp[a:a + CROP], mean, std))
            ys.append(lab[a:a + CROP])
            ws.append(w[a:a + CROP])
        x = torch.from_numpy(np.stack(xs))
        y = torch.from_numpy(np.stack(ys).astype(np.int64))
        w = torch.from_numpy(np.stack(ws))
        loss = (F.cross_entropy(net(x).reshape(-1, 3), y.reshape(-1), reduction="none") * w.reshape(-1)).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        running += float(loss)
        if (step + 1) % 100 == 0:
            print("    step {}/{} loss {:.4f} ({:.0f}s)".format(step + 1, steps, running / 100, time.time() - t0), flush=True)
            running = 0.0
    return net, mean, std


@torch.no_grad()
def predict_tcn(net, mean, std, inputs, period):
    net.eval()
    out = np.zeros((len(inputs), 3), np.float32)
    for s, e in halves(period):
        x = torch.from_numpy(prepare(inputs[s:e], mean, std))[None]
        out[s:e] = F.softmax(net(x)[0], -1).numpy()
    return out


def main(tag="tcn", **opts):
    t0 = time.time()
    mode = opts.get("inputs", "stack")
    epochs = int(opts.get("epochs", 8))
    torch.set_num_threads(8)
    oof = load_oof(opts.get("oof", "v3"))
    folds = FOLDS[:int(opts.get("folds", len(FOLDS)))]
    ext = matches.external()
    # cache=<dir>: the bootstrap cache the out-of-fold probabilities were computed with, so that
    # the features rebuilt here carry the same simulated visibility as those probabilities
    tabs = pipeline.tables(ext, bank=pipeline.load_bank(opts.get("cache", CACHE)))
    names = tabs[0]["names"]
    cols = models.team_columns(names)
    items = {}
    for t in tabs:
        base = t["name"].replace("_swap", "")
        p_dead, p_home, _, _, p_ev = oof[base]
        if t["name"].endswith("_swap"):
            p_home = 1.0 - p_home            # the same match seen from the other bench
        inputs = make_inputs(t, (p_dead, p_home, p_ev), cols, mode)
        items[t["name"]] = (inputs, t["label"], t["period"], models.spell_weights(t["label"]))
    n_in = next(iter(items.values()))[0].shape[1]
    print("inputs={} ({} per frame), {} sequences, ready in {:.0f}s".format(mode, n_in, len(items), time.time() - t0), flush=True)

    out = {}
    for k, held in enumerate(folds):
        train = [v for n, v in items.items() if n.replace("_swap", "") not in held]
        net, mean, std = train_tcn(train, n_in, epochs, seed=k)
        for name in held:
            inputs, lab, per, _ = items[name]
            p = predict_tcn(net, mean, std, inputs, per)
            p_dead = p[:, 2]
            p_home = p[:, 0] / np.maximum(p[:, 0] + p[:, 1], 1e-6)
            out[name] = (p_dead, p_home, lab, per, oof[name][4])
            al = lab < 2
            print("  fold {} {:<18} dead acc {:.4f} | team acc on live frames {:.4f}".format(
                k, name, ((p_dead > 0.5) == (lab == 2)).mean(), ((p_home[al] > 0.5) == (lab[al] == 0)).mean()), flush=True)
    np.savez_compressed(os.path.join(ROOT, "experiments", "oof_{}.npz".format(tag)),
                        **{"{}|{}".format(n, f): v for n, vals in out.items()
                           for f, v in zip(("p_dead", "p_home", "label", "period", "p_events"), vals)})
    with open(os.path.join(ROOT, "experiments", "results.jsonl"), "a") as f:
        f.write(json.dumps({"tag": tag, "opts": opts, "seconds": round(time.time() - t0),
                            "time": time.strftime("%Y-%m-%d %H:%M")}) + "\n")
    print("done in {:.0f}s; now: python scripts/tune_decoder.py {}".format(time.time() - t0, tag))


if __name__ == "__main__":
    args = sys.argv[1:]
    tag = args[0] if args and "=" not in args[0] else "tcn"
    main(tag, **dict(a.split("=", 1) for a in args if "=" in a))
