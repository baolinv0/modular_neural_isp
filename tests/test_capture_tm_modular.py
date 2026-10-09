"""The actual shipped renderer must load and remain frozen."""
import importlib
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def backend_module():
    def construct(name, *args, **kwargs):
        try:
            module = importlib.import_module("capture_tm.modular")
        except ModuleNotFoundError:
            pytest.fail("capture_tm.modular has not been implemented")
        return getattr(module, name)(*args, **kwargs)
    return SimpleNamespace(
        ModularPhotofinishingBackend=lambda *a, **k: construct("ModularPhotofinishingBackend", *a, **k),
        AnalyticBackend=lambda *a, **k: construct("AnalyticBackend", *a, **k),
    )


def test_missing_checkpoint_never_falls_back_to_random_weights(backend_module, tmp_path):
    with pytest.raises(FileNotFoundError):
        backend_module.ModularPhotofinishingBackend(tmp_path / "missing.pth")


def test_malformed_checkpoint_fails_strict_loading(backend_module, tmp_path):
    path = tmp_path / "empty.pth"
    torch.save({}, path)
    with pytest.raises(RuntimeError):
        backend_module.ModularPhotofinishingBackend(path)


def test_renderer_fingerprint_identifies_checkpoint_bytes(backend_module):
    backend = backend_module.ModularPhotofinishingBackend()
    expected = hashlib.sha256(Path(backend.checkpoint).read_bytes()).hexdigest()
    assert backend.checkpoint_sha256 == expected
    assert expected in backend.renderer_identity


def test_default_shipped_checkpoint_runs_frozen_independent_of_cwd(backend_module, tmp_path, monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        monkeypatch.chdir(tmp_path)
        backend = backend_module.ModularPhotofinishingBackend()
        assert Path(backend.checkpoint).name == "photofinishing_s24-style-0.pth"
        assert not any(p.requires_grad for p in backend.parameters())
        backend.train()
        assert not backend.model.training
        x = torch.linspace(.02, .7, 32 * 32).reshape(1, 1, 32, 32).repeat(1, 3, 1, 1)
        first = backend(x)
        second = backend(x)
        assert first.shape == x.shape
        assert torch.isfinite(first).all()
        assert torch.all((first >= 0) & (first <= 1))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert not first.requires_grad
    finally:
        torch.set_num_threads(old_threads)


def test_backend_rejects_unbounded_input_instead_of_destroying_hdr(backend_module):
    backend = backend_module.AnalyticBackend()
    with pytest.raises(ValueError):
        backend(torch.full((1, 3, 2, 2), 2.))


@pytest.mark.parametrize("size", [(1, 4), (8, 8), (11, 16), (16, 11)])
def test_real_backend_rejects_undersized_images_with_readable_error(backend_module, size):
    backend = backend_module.ModularPhotofinishingBackend()
    with pytest.raises(ValueError, match="H and W >= 12"):
        backend(torch.full((1, 3, *size), .1))


def test_real_backend_accepts_twelve_pixel_minimum(backend_module):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        backend = backend_module.ModularPhotofinishingBackend()
        x = torch.full((1, 3, 12, 12), .1)
        out = backend(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        assert torch.all((out >= 0) & (out <= 1))
    finally:
        torch.set_num_threads(old_threads)


def test_analytic_backend_is_explicit_srgb_oetf(backend_module):
    x = torch.tensor([0., .0031308, .18, 1.]).reshape(1, 1, 1, 4).repeat(1, 3, 1, 1)
    out = backend_module.AnalyticBackend()(x)
    torch.testing.assert_close(out[0, 0, 0], torch.tensor([0., .040449936, .46135613, 1.]), rtol=1e-5, atol=1e-7)


def test_both_patent_branches_run_the_same_real_shipped_renderer(backend_module):
    from capture_tm.tone import PatentToneMapper
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        backend = backend_module.ModularPhotofinishingBackend()
        state = torch.load(backend.checkpoint, map_location="cpu", weights_only=True)
        first_key = next(iter(state))
        torch.testing.assert_close(backend.model.state_dict()[first_key], state[first_key], rtol=0, atol=0)
        x = torch.linspace(.02, 2., 32 * 32).reshape(1, 1, 32, 32).repeat(1, 3, 1, 1)
        for mode in ("apple", "samsung"):
            out = PatentToneMapper(mode, backend=backend)(x, capture_bias_ev=1.)
            assert out["output"].shape == x.shape
            assert torch.isfinite(out["output"]).all()
            assert torch.all((out["output"] >= 0) & (out["output"] <= 1))
            assert out["pre_backend"][0, 0, -1, -1] > out["pre_backend"][0, 0, 16, 0]
        assert not backend.model.training
        assert not any(p.requires_grad for p in backend.parameters())
    finally:
        torch.set_num_threads(old_threads)
