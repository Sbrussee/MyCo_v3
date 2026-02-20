import pytest
import torch

from myco.model import MoCoV3Lit


def test_contrastive_loss_matches_mocov3_temperature_scaling_single_process() -> None:
    module = object.__new__(MoCoV3Lit)
    module.temperature = 0.2

    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    loss = MoCoV3Lit._contrastive_loss(module, q, k)

    logits = (
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32) / module.temperature
    )
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    expected = expected * (2.0 * module.temperature)
    assert float(loss) == pytest.approx(float(expected), abs=1e-7)


def test_on_load_checkpoint_remaps_legacy_projector_keys_and_drops_mismatches() -> None:
    module = object.__new__(MoCoV3Lit)
    module.state_dict = lambda: {
        "q_proj.net.0.weight": torch.zeros((4, 4), dtype=torch.float32),
        "q_proj.net.1.weight": torch.zeros((4,), dtype=torch.float32),
        "q_enc.pos_embed": torch.zeros((1, 26, 384), dtype=torch.float32),
    }

    checkpoint = {
        "state_dict": {
            "q_proj.fc1.weight": torch.ones((4, 4), dtype=torch.float32),
            "q_proj.bn1.weight": torch.ones((4,), dtype=torch.float32),
            "q_enc.pos_embed": torch.ones((1, 785, 384), dtype=torch.float32),
        }
    }

    MoCoV3Lit.on_load_checkpoint(module, checkpoint)

    restored = checkpoint["state_dict"]
    assert "q_proj.net.0.weight" in restored
    assert "q_proj.net.1.weight" in restored
    assert "q_enc.pos_embed" not in restored
