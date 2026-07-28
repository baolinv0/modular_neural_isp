import copy

import torch

from photofinishing.photofinishing_model import PhotofinishingModule
from photofinishing.unpaired_chroma_heads import ChromaHead, FrozenLUTAffineResidual, configure_chroma_head
from photofinishing.unpaired_stage_control import (
    AdaptationStage,
    assert_trainable_scope,
    configure_trainable_scope,
    set_stage_train_mode,
)
from photofinishing.unpaired_style_losses import Stage1UnpairedLoss, Stage2UnpairedLoss


def test_real_photofinishing_stage1_interface_and_gradients():
    model = PhotofinishingModule(device=torch.device("cpu"), use_3d_lut=False)
    configure_trainable_scope(model, AdaptationStage.LUMINANCE)
    image = torch.rand(1, 3, 256, 256)
    reference = torch.rand(1, 3, 256, 256)
    output = model(image, training_mode=True)
    assert "output" in output and "cbcr_lut" in output
    anchor = sum((parameter * 0).sum() for parameter in model.parameters() if parameter.requires_grad)
    loss, _ = Stage1UnpairedLoss()(output["output"], reference, anchor)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model._gain_net.parameters())
    assert any(parameter.grad is not None for parameter in model._gtm_net.parameters())
    assert all(parameter.grad is None for parameter in model._ltm_net.parameters())
    assert all(parameter.grad is None for parameter in model._lut_net.parameters())
    assert all(parameter.grad is None for parameter in model._gamma_net.parameters())


def test_real_photofinishing_affine_stage2_preserves_initial_output_and_gradients():
    stage1 = PhotofinishingModule(device=torch.device("cpu"), use_3d_lut=False).eval()
    adapted = copy.deepcopy(stage1)
    for parameter in stage1.parameters():
        parameter.requires_grad = False

    image = torch.rand(1, 3, 128, 128)
    reference = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        stage1_output = stage1(image, training_mode=True)

    configure_chroma_head(adapted, ChromaHead.AFFINE_RESIDUAL)
    configure_trainable_scope(adapted, AdaptationStage.CHROMA, ChromaHead.AFFINE_RESIDUAL)
    assert_trainable_scope(adapted, AdaptationStage.CHROMA, ChromaHead.AFFINE_RESIDUAL)
    set_stage_train_mode(adapted, AdaptationStage.CHROMA, ChromaHead.AFFINE_RESIDUAL)

    assert isinstance(adapted._lut_net, FrozenLUTAffineResidual)
    trainable = [parameter for parameter in adapted.parameters() if parameter.requires_grad]
    assert sum(parameter.numel() for parameter in trainable) == 6

    with torch.no_grad():
        initial = adapted(image, training_mode=True)
    assert torch.equal(initial["cbcr_lut"], stage1_output["cbcr_lut"])
    assert torch.equal(initial["output"], stage1_output["output"])

    result = adapted(image, training_mode=True)
    loss, _ = Stage2UnpairedLoss()(
        result["output"],
        reference,
        stage1_output["output"],
        result["cbcr_lut"],
        stage1_output["cbcr_lut"],
    )
    loss.backward()
    assert adapted._lut_net.matrix_raw.grad is not None
    assert adapted._lut_net.bias_raw.grad is not None
    assert all(parameter.grad is None for parameter in adapted._lut_net.base_lut_net.parameters())
    assert all(parameter.grad is None for parameter in adapted._gain_net.parameters())
    assert all(parameter.grad is None for parameter in adapted._gtm_net.parameters())
    assert all(parameter.grad is None for parameter in adapted._ltm_net.parameters())
    assert all(parameter.grad is None for parameter in adapted._gamma_net.parameters())
