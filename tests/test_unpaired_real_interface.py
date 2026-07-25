import torch

from photofinishing.photofinishing_model import PhotofinishingModule
from photofinishing.unpaired_stage_control import AdaptationStage, configure_trainable_scope
from photofinishing.unpaired_style_losses import Stage1UnpairedLoss


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
