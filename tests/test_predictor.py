"""Tests for the sampling/solver helpers (require torch, CPU is enough)."""

import pytest

torch = pytest.importorskip("torch")
import numpy as np

from rmd.engines.predictor import (
    EulerSolver,
    Predictor,
    extract_into_tensor,
    gather_trajectory_batch,
)


def test_extract_into_tensor():
    a = torch.arange(10.0)
    t = torch.tensor([3])
    out = extract_into_tensor(a, t, (2, 2))
    assert out.shape == (1, 1, 1)
    assert out[0, 0, 0] == 3.0


def test_extract_into_tensor_batch():
    a = torch.arange(10.0)
    t = torch.tensor([2, 7])
    out = extract_into_tensor(a, t, (2, 16, 21, 60, 104))
    assert out.shape == (2, 1, 1, 1, 1)
    assert out[0, 0, 0, 0, 0] == 2.0
    assert out[1, 0, 0, 0, 0] == 7.0


def test_euler_solver_shapes():
    sigmas = np.linspace(1.0, 0.0, 1001)[::-1]  # ascending schedule like FlowMatch
    solver = EulerSolver(sigmas, timesteps=1000, euler_timesteps=50)
    assert tuple(solver.sigmas.shape) == (50,)
    assert tuple(solver.euler_timesteps.shape) == (50,)
    solver = solver.to("cpu")


def test_euler_solver_euler_step():
    sigmas = np.linspace(1.0, 0.0, 1001)[::-1]
    solver = EulerSolver(sigmas, timesteps=1000, euler_timesteps=50).to("cpu")
    sample = torch.zeros(2, 4)
    pred = torch.ones(2, 4)
    idx = torch.tensor([10, 10])
    x_prev = solver.euler_step(sample, pred, idx)
    assert x_prev.shape == (2, 4)
    # sigma_prev < sigma -> x_prev = sample + (sigma_prev - sigma) * pred < 0
    assert torch.all(x_prev < 0)


def test_upsample_infer_shapes():
    # 480p latent -> 720p latent via bilinear
    latent = torch.randn(1, 16, 21, 60, 104)
    p = Predictor.__new__(Predictor)  # skip the heavy __init__ (needs transformers)
    out = p.upsample_infer(latent, target_size=(90, 160), mode="bilinear")
    assert tuple(out.shape) == (1, 16, 21, 90, 160)


def test_upsample_infer_identity_when_same_size():
    latent = torch.randn(1, 16, 21, 90, 160)
    p = Predictor.__new__(Predictor)
    out = p.upsample_infer(latent, target_size=(90, 160), mode="bilinear")
    assert tuple(out.shape) == (1, 16, 21, 90, 160)


def test_gather_trajectory_batch_uses_sampled_resolution():
    low = torch.randn(2, 16, 21, 60, 104)
    high = torch.randn(2, 16, 21, 90, 160)
    trajectory = [low, low + 1, high, high + 1]

    low_batch = gather_trajectory_batch(trajectory, torch.tensor([0, 1]))
    high_batch = gather_trajectory_batch(trajectory, torch.tensor([2, 3]))

    assert low_batch.shape == low.shape
    assert high_batch.shape == high.shape


def test_gather_trajectory_batch_rejects_mixed_resolutions():
    low = torch.randn(2, 16, 21, 60, 104)
    high = torch.randn(2, 16, 21, 90, 160)
    with pytest.raises(ValueError, match="mixed low- and high-resolution"):
        gather_trajectory_batch([low, high], torch.tensor([0, 1]))


def test_predictor_accepts_precomputed_unconditional_embedding():
    p = Predictor(
        type("Args", (), {"train_batch_size": 2})(),
        None,
        None,
        "cpu",
        torch.float32,
        solver=None,
        uncond_prompt_embeds=torch.zeros(1, 512, 4),
    )
    assert p.uncond_prompt_embeds.shape == (1, 512, 4)


class _DtypeCheckingTransformer:
    def __init__(self, expected_dtype):
        self.expected_dtype = expected_dtype
        self.calls = 0

    def __call__(self, hidden_states, **kwargs):
        assert hidden_states.dtype == self.expected_dtype
        self.calls += 1
        return (torch.zeros_like(hidden_states),)


def test_generate_new_keeps_reinjected_noise_in_model_dtype():
    solver = EulerSolver(np.linspace(0.0, 1.0, 1001), timesteps=1000, euler_timesteps=1000).to("cpu")
    predictor = Predictor.__new__(Predictor)
    predictor.solver = solver
    predictor.solver_hrs = solver
    predictor.weight_dtype = torch.bfloat16
    transformer = _DtypeCheckingTransformer(torch.bfloat16)
    noise = torch.randn(1, 2, 2, 2, 2, dtype=torch.bfloat16)

    predictor.generate_new(
        transformer,
        noise_scheduler=None,
        latent=noise,
        noise=noise,
        encoder_hidden_states=torch.zeros(1, 2, 2, dtype=torch.bfloat16),
        image_rotary_emb=None,
        steps=2,
        eta=0.0,
    )

    assert transformer.calls == 2


def test_generate_new_upsample_keeps_reinjected_noise_in_model_dtype():
    solver = EulerSolver(np.linspace(0.0, 1.0, 1001), timesteps=1000, euler_timesteps=1000).to("cpu")
    predictor = Predictor.__new__(Predictor)
    predictor.solver = solver
    predictor.solver_hrs = solver
    predictor.weight_dtype = torch.bfloat16
    transformer = _DtypeCheckingTransformer(torch.bfloat16)
    noise = torch.randn(1, 2, 2, 2, 2, dtype=torch.bfloat16)

    predictor.generate_new_upsample(
        transformer,
        noise_scheduler=None,
        latent=noise,
        noise=noise,
        encoder_hidden_states=torch.zeros(1, 2, 2, dtype=torch.bfloat16),
        image_rotary_emb=None,
        steps=2,
        eta=0.0,
        flow_shift_trans=0,
    )

    assert transformer.calls == 2


def test_generate_new_upsample_reinjects_pure_gaussian_at_resolution_shift(monkeypatch):
    solver = EulerSolver(np.linspace(0.0, 1.0, 1001), timesteps=1000, euler_timesteps=1000).to("cpu")
    predictor = Predictor.__new__(Predictor)
    predictor.solver = solver
    predictor.solver_hrs = solver
    predictor.weight_dtype = torch.float32
    transformer = _DtypeCheckingTransformer(torch.float32)
    noise = torch.zeros(1, 2, 2, 2, 2)
    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.ones_like(tensor))

    _, noisy_imgs = predictor.generate_new_upsample(
        transformer,
        noise_scheduler=None,
        latent=noise,
        noise=noise,
        encoder_hidden_states=torch.zeros(1, 2, 2),
        image_rotary_emb=None,
        steps=2,
        eta=0.5,
        return_mid=True,
        flow_shift_trans=1,
        relusion_shift=0,
        target_size=(2, 2),
    )

    expected = torch.full_like(noisy_imgs[1], solver.sigmas[499].item())
    assert torch.allclose(noisy_imgs[1], expected)
