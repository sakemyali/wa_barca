import os

import numpy as np
import pytest

from possession.tagging import frame_mask, load_tagging, parse_tagging

XML = """<file><ALL_INSTANCES>
<instance><ID>1</ID><start>10.0</start><end>12.04</end><code>SET PIECES</code>
<label><group>Team</group><text>N/A</text></label>
<label><group>Type</group><text>Throw-in</text></label>
<label><group>Half</group><text>1st Half</text></label></instance>
<instance><ID>2</ID><start>100.52</start><end>105.52</end><code>CROSS</code>
<label><group>Team</group><text>Barcelona</text></label>
<label><group>Half</group><text>2nd Half</text></label>
<label><group>Max Players in the box</group><text>4</text></label></instance>
</ALL_INSTANCES></file>"""


def test_parse(tmp_path):
    p = tmp_path / "t.xml"
    p.write_text(XML)
    df = parse_tagging(p)
    assert list(df["id"]) == [1, 2]
    assert list(df["code"]) == ["set pieces", "cross"]
    assert df["team"].tolist() == [None, "H"]
    assert list(df["half"]) == [1, 2]
    assert list(df["start_frame"]) == [250, 2513]
    assert list(df["end_frame"]) == [301, 2638]
    assert df["type"].tolist() == ["Throw-in", None]
    assert df["max_players_in_box"].tolist() == [None, "4"]
    m = frame_mask(df, 400)
    assert m[250] and m[301] and not m[249] and not m[302] and m.sum() == 52


@pytest.mark.skipif(not os.environ.get("MTAH_DATA_DIR"), reason="MTAH_DATA_DIR unset")
def test_real_file():
    df = load_tagging()
    assert len(df) == 638
    assert df["team"].isna().sum() == 36
