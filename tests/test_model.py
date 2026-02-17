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
