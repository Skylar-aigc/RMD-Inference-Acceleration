"""Tests for the distillation loss functions (require torch)."""

import pytest

torch = pytest.importorskip("torch")

from rmd.losses.distillation import (
    compute_huber_c,
    compute_revised_latents,
    compute_score_loss,
    compute_student_loss,
    compute_weighting_factor,
)


def test_compute_revised_latents():
    m = torch.zeros(1, 1)
    r = torch.ones(1, 1)
    f = torch.full((1, 1), 2.0)
    expected = torch.full((1, 1), -1.0)
    assert torch.allclose(compute_revised_latents(m, r, f), expected)


def test_compute_huber_c():
    c = compute_huber_c((1, 16, 21, 90, 160))
    assert isinstance(c, float) and c > 0
    # larger latents -> larger huber_c
    small = compute_huber_c((1, 16, 21, 60, 104))
    assert c > small
    assert c == compute_huber_c((3, 16, 21, 90, 160))


def test_compute_score_loss_scalar():
    fake = torch.zeros(4, 16)
    target = torch.zeros(4, 16)
    sigmas = torch.full((4, 1), 0.5)
    loss = compute_score_loss(fake, target, sigmas)
    assert torch.is_tensor(loss) and loss.ndim == 0
    assert loss.item() == 0.0  # identical latents -> zero loss


def test_compute_score_loss_nonzero():
    fake = torch.zeros(4, 16)
    target = torch.ones(4, 16)
    sigmas = torch.full((4, 1), 0.5)
    loss = compute_score_loss(fake, target, sigmas)
    assert loss.item() > 0.0


def test_compute_score_loss_supports_inverse_sigma_mode():
    fake = torch.zeros(1, 1, dtype=torch.float64)
    target = torch.ones_like(fake)
    sigmas = torch.tensor([[0.5]], dtype=torch.float64)

    loss = compute_score_loss(fake, target, sigmas, mode="inverse_sigma")

    assert torch.allclose(loss, torch.tensor(1.0), rtol=1e-6)


def test_compute_score_loss_supports_capped_legacy_snr_mode():
    fake = torch.zeros(1, 1, dtype=torch.float64)
    target = torch.ones_like(fake)
    sigmas = torch.tensor([[0.9]], dtype=torch.float64)

    loss = compute_score_loss(fake, target, sigmas, mode="legacy_snr")

    assert torch.allclose(loss, torch.tensor(5.0), rtol=1e-6)


def test_compute_score_loss_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown score weighting mode"):
        compute_score_loss(torch.zeros(1, 1), torch.ones(1, 1), torch.ones(1, 1), mode="unknown")


def test_weighting_factor_clip():
    m = torch.zeros(1, 1, 4, 4)
    r = torch.zeros(1, 1, 4, 4)  # zero distance -> clipped to clip_min
    factor = compute_weighting_factor(m, r, clip_min=0.2)
    assert factor.min().item() >= 0.2


def test_student_loss_shape():
    m = torch.zeros(2, 16)
    rev = torch.ones(2, 16)
    w = torch.ones(2, 1, 1)
    loss = compute_student_loss(m, rev, w, huber_c=1e-3)
    assert loss.ndim == 0 and loss.item() > 0
