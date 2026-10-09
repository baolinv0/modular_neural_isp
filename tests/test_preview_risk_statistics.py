import pytest


def test_highlight_contrast_excludes_scenes_without_highlights():
    from research.ae_tm.diagnostics.preview_risk_ablation import interval
    a = [{'scene_id': 'bright', 'highlight_mse': .1, 'bright_pixel_count': 12}]
    b = [{'scene_id': 'bright', 'highlight_mse': .3, 'bright_pixel_count': 12}]
    for i in range(6):
        a.append({'scene_id': str(i), 'highlight_mse': 0., 'bright_pixel_count': 0})
        b.append({'scene_id': str(i), 'highlight_mse': 0., 'bright_pixel_count': 0})
    contrast = interval(a, b, 'highlight_mse')
    assert contrast['mean'] == pytest.approx(-.2)
    assert contrast['independent_scenes'] == 1
    assert contrast['ci95_descriptive'] == pytest.approx([-.2, -.2])


def test_absent_regions_have_no_numerical_quality_score():
    from research.ae_tm.diagnostics.preview_risk_ablation import mean_present, interval
    assert mean_present([None, None]) is None
    assert mean_present([None, .2, .4]) == pytest.approx(.3)
    rows = [{'scene_id': 'dark', 'bright_radiance_mse': None, 'bright_pixel_count': 0}]
    result = interval(rows, rows, 'bright_radiance_mse')
    assert result['mean'] is None and result['independent_scenes'] == 0
