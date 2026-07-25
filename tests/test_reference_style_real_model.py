import torch

from photofinishing.photofinishing_model import PhotofinishingModule
from photofinishing.reference_style_training import configure_trainable_stage


def test_real_photofinishing_stage_contract():
  model = PhotofinishingModule(device=torch.device("cpu"), use_3d_lut=False)

  tone_names = configure_trainable_stage(model, "tone")
  assert tone_names
  assert all(name.startswith(("_gain_net", "_gtm_net")) for name in tone_names)
  assert not any(name.startswith(("_ltm_net", "_lut_net", "_gamma_net")) for name in tone_names)

  chroma_names = configure_trainable_stage(model, "chroma")
  assert chroma_names
  assert all(name.startswith("_lut_net") for name in chroma_names)
  assert not any(
    name.startswith(("_gain_net", "_gtm_net", "_ltm_net", "_gamma_net"))
    for name in chroma_names
  )
