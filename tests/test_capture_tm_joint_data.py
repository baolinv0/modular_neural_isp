from capture_tm.joint_data import write_joint_dataset
from capture_tm.data import load_manifest


def test_all_splits_include_four_scene_conditions_and_sufficient_time_support(tmp_path):
    path = write_joint_dataset(tmp_path, scenes=20, size=16, seed=5)
    _, scenes = load_manifest(path)
    for split in ('train', 'val', 'test'):
        rows = [s for s in scenes if s.split == split]
        assert {s.provenance['condition'] for s in rows} == {'dark_static', 'dark_motion', 'backlit', 'ordinary'}
        assert all(float(s.frame_times_s[0]) <= -.2 and float(s.frame_times_s[-1]) >= .1 for s in rows)
    assert any(s.frames.max() > 1 for s in scenes)
    dark = [s.frames.mean() for s in scenes if s.provenance['condition'] == 'dark_static']
    bright = [s.frames.mean() for s in scenes if s.provenance['condition'] == 'ordinary']
    assert sum(dark) / len(dark) < sum(bright) / len(bright)
