"""Tests for the NjegošGPT model (run on CPU)."""

import math

import pytest
import torch

from model import GPT, GPTConfig

V = 50  # small vocabulary for the tests


def tiny(**kw):
    cfg = {"vocab_size": V, "block_size": 16, "n_layer": 2, "n_head": 2, "n_embd": 32, "dropout": 0.0}
    return GPT(GPTConfig(**(cfg | kw)))


def test_forward_shapes():
    model = tiny()
    idx = torch.randint(V, (3, 10))
    logits, loss = model(idx)
    assert logits.shape == (3, 10, V)
    assert loss is None
    _, loss = model(idx, idx)
    assert loss.dim() == 0


def test_initial_loss_is_ln_vocab():
    # at init the model should guess uniformly: loss ≈ ln(V)
    torch.manual_seed(0)
    model = tiny()
    idx = torch.randint(V, (8, 16))
    _, loss = model(idx, torch.randint(V, (8, 16)))
    assert loss.item() == pytest.approx(math.log(V), abs=0.1)


@pytest.mark.parametrize("use_flash", [True, False])
def test_causal_masking(use_flash):
    # changing a future token must not change the logits at earlier positions
    torch.manual_seed(0)
    model = tiny(use_flash=use_flash).eval()
    idx = torch.randint(V, (1, 12))
    idx2 = idx.clone()
    idx2[0, 8] = (idx[0, 8] + 1) % V
    a, _ = model(idx)
    b, _ = model(idx2)
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-6)
    assert not torch.allclose(a[:, 8:], b[:, 8:])


def test_flash_and_explicit_agree():
    torch.manual_seed(0)
    flash = tiny(use_flash=True).eval()
    explicit = tiny(use_flash=False).eval()
    explicit.load_state_dict(flash.state_dict())
    idx = torch.randint(V, (2, 16))
    assert torch.allclose(flash(idx)[0], explicit(idx)[0], atol=1e-5)


def test_generate_length():
    model = tiny().eval()
    idx = torch.zeros((2, 3), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=20, temperature=0.8, top_k=5)
    assert out.shape == (2, 23)  # longer than block_size, so it also tests context cropping


def test_overfit_one_batch():
    # a small model must be able to "memorize" a single batch
    torch.manual_seed(0)
    model = tiny()
    x = torch.randint(V, (4, 16))
    y = torch.randint(V, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = model(x, y)[1].item()
    for _ in range(100):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2 * first
