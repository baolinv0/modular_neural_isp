import copy
import hashlib
import json

import numpy as np
import torch

from photofinishing.evaluate.config import ModelSpec
from photofinishing.evaluate.model_loader import infer_rgb, load_model_from_spec
from photofinishing.photofinishing_model import PhotofinishingModule
from photofinishing.unpaired_chroma_heads import ChromaHead, FrozenLUTAffineResidual, configure_chroma_head


def _sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_affine_checkpoint_round_trip_reconstructs_wrapper_and_infers(tmp_path):
    device = torch.device("cpu")
    stage1 = PhotofinishingModule(device=device, use_3d_lut=False).eval()
    adapted = copy.deepcopy(stage1)
    configure_chroma_head(adapted, ChromaHead.AFFINE_RESIDUAL)
    checkpoint = tmp_path / "best.pth"
    torch.save(adapted.state_dict(), checkpoint)
    run_config = tmp_path / "run_config.json"
    run_config.write_text(json.dumps({
        "stage": "chroma",
        "chroma_head": "affine_residual",
        "affine_matrix_limit": 0.15,
        "affine_bias_limit": 0.05,
        "best_checkpoint_sha256": _sha256(checkpoint),
        "last_checkpoint_sha256": _sha256(checkpoint),
    }), encoding="utf-8")

    spec = ModelSpec(
        name="affine",
        checkpoint=checkpoint,
        run_config=run_config,
        use_3d_lut=False,
    )
    loaded, metadata = load_model_from_spec(spec, role="stage2", device=device)
    assert isinstance(loaded._lut_net, FrozenLUTAffineResidual)
    assert metadata["chroma_head"] == "affine_residual"

    image = np.full((128, 128, 3), 0.2, np.float32)
    expected = infer_rgb(adapted, image, device)
    actual = infer_rgb(loaded, image, device)
    assert actual.shape == (128, 128, 3)
    assert np.array_equal(actual, expected)
