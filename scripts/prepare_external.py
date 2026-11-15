"""Download the public training matches and convert them to data_external/*.npz.

    python scripts/prepare_external.py            # everything (about 2.8 GB streamed, 300 MB kept)
    python scripts/prepare_external.py dfl        # only the DFL matches
    python scripts/prepare_external.py metrica    # only Metrica's sample games

None of this is the provided match: these are other games, published by their owners.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from possession import schema  # noqa: E402
from possession.external import dfl  # noqa: E402

OUT = os.environ.get("MTAH_EXTERNAL_DIR") or os.path.join(ROOT, "data_external")


def main(which):
    os.makedirs(OUT, exist_ok=True)
    if which in ("all", "dfl"):
        for match_id in dfl.FILES:
            path = os.path.join(OUT, "dfl_{}.npz".format(match_id))
            if os.path.isfile(path):
                print("have", path)
                continue
            t0 = time.time()
            m = dfl.fetch(match_id)
            schema.save(path, m)
            print("{}: {} frames in {:.0f}s -> {}".format(m.name, m.n_frames, time.time() - t0, path), flush=True)
    if which in ("all", "metrica"):
        from possession.external import metrica_sample
        metrica_sample.prepare(OUT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
