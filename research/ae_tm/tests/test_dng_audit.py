"""Protocol and RAW-domain checks independent of training scores."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import tifffile

SOURCE=Path(__file__).parents[1]/"diagnostics"/"dng_audit.py"
spec=importlib.util.spec_from_file_location("dng_audit",SOURCE)
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_stale_external_path_resolves_inside_declared_root(tmp_path):
    folder=tmp_path/"s25"/"DNG"
    folder.mkdir(parents=True)
    expected=folder/"source.dng"
    expected.write_bytes(b"declared source")
    result=audit.resolve_moved_file(
        tmp_path,"s25","DNG",r"X:\unrelated\external\source.dng")
    assert result==expected
    assert result.read_bytes()==b"declared source"


def test_ambiguous_same_basename_is_rejected(tmp_path):
    folder=tmp_path/"s25"/"DNG"
    for child,value in (("first",b"a"),("second",b"b")):
        dest=folder/child/"same.dng"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(value)
    with pytest.raises(ValueError,match="Ambiguous"):
        audit.resolve_moved_file(tmp_path,"s25","DNG",r"X:\gone\same.dng")


def test_preview_root_is_not_mistaken_for_raw_subifd(tmp_path):
    path=tmp_path/"stacked.dng"
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((4,4,3),np.uint8),photometric="rgb",subifds=1)
        writer.write(np.zeros((4,4,3),np.uint16),photometric=34892,
                     extratags=[(50717,"I",3,(4095,4095,4095),False)])
    records=audit.ifd_metadata(path)
    assert records[0]["raw_kind"]=="non-raw"
    assert records[1]["raw_kind"]=="processed-linear-rgb"
    assert records[1]["ifd"]=="0/0"
    assert records[1]["WhiteLevel"]==[4095,4095,4095]


def test_source_group_coverage_and_duplicate_split_are_enforced():
    protocol=json.loads((SOURCE.parent/"DNG_PROTOCOL.json").read_text())
    rows=[{"pair_id":i} for i in range(34)]
    mapping=audit.validate_groups(rows,protocol)
    assert mapping[2]==mapping[9] # conservatively merged room
    assert mapping[12]==mapping[18] # conservatively merged lobby
    protocol["source_groups"][-1]["pair_ids"].append(0)
    with pytest.raises(ValueError,match="multiple"):
        audit.validate_groups(rows,protocol)
