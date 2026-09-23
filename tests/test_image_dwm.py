from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from server import image_dwm


def test_activation_projection_matches_row_weight_projection():
    generator = torch.Generator().manual_seed(42)
    weight = torch.randn(image_dwm.EXPECTED_HIDDEN, 7, generator=generator)
    values = torch.randn(3, 5, 7, generator=generator)
    direction = torch.randn(image_dwm.EXPECTED_HIDDEN, generator=generator)
    direction = direction / direction.norm()
    alpha = 0.75
    clean = torch.nn.functional.linear(values, weight)
    with image_dwm.request_scale(0.5):
        projected = image_dwm._project(clean, direction, alpha)
    edited = weight - (alpha * 0.5) * torch.outer(direction, direction @ weight)
    expected = torch.nn.functional.linear(values, edited)
    torch.testing.assert_close(projected, expected, rtol=2e-5, atol=2e-5)


def test_zero_request_scale_is_exact_identity():
    values = torch.randn(2, image_dwm.EXPECTED_HIDDEN)
    direction = torch.randn(image_dwm.EXPECTED_HIDDEN)
    with image_dwm.request_scale(0):
        projected = image_dwm._project(values, direction, 1.0)
    assert projected is values


def test_bundled_bank_contains_only_verified_runtime_vectors():
    bank = Path(__file__).resolve().parents[1] / 'server/data/qwen-image21-dwm.safetensors'
    manifest = json.loads(bank.with_suffix('.json').read_text())
    assert bank.stat().st_size == manifest['bytes']
    assert bank.stat().st_size < 1_200_000
    assert hashlib.sha256(bank.read_bytes()).hexdigest() == manifest['sha256']
    expected = {'direction.attention_output_last', 'direction.mlp_output_last'}
    assert set(manifest['tensors']) == expected
    with safe_open(str(bank), framework='pt', device='cpu') as handle:
        assert set(handle.keys()) == expected  # No pairs, means or capture activations.
        assert handle.metadata()['revision'] == manifest['source_revision']
        for key in expected:
            value = handle.get_tensor(key)
            assert value.dtype == torch.float32
            assert list(value.shape) == manifest['tensor_shape'] == [36, 4096]
            assert torch.isfinite(value).all()
            torch.testing.assert_close(value.norm(dim=-1), torch.ones(36), rtol=1e-5, atol=1e-5)


def test_bundled_profile_loads_with_reversible_hooks(monkeypatch):
    bank = Path(__file__).resolve().parents[1] / 'server/data/qwen-image21-dwm.safetensors'
    profile = json.loads(bank.with_suffix('.json').read_text())['profile']
    for name, value in {
        'FVL_IMAGE_DWM_BANK': str(bank),
        'FVL_IMAGE_DWM_LAYERS': profile['layers'],
        'FVL_IMAGE_DWM_ATTN_ALPHA': str(profile['attention_alpha']),
        'FVL_IMAGE_DWM_MLP_ALPHA': str(profile['mlp_alpha']),
    }.items():
        monkeypatch.setenv(name, value)
    layers = [SimpleNamespace(self_attn=torch.nn.Identity(), mlp=torch.nn.Identity())
              for _ in range(36)]
    encoder = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)))
    status = image_dwm.configure_from_environment(encoder)
    assert status['enabled'] and status['layers'] == list(range(19, 25))
    assert len(encoder._blackfrost_image_dwm_handles) == 12
    assert not status['weights_modified']
    values = torch.randn(1, 2, 4096, generator=torch.Generator().manual_seed(23))
    for strength in (0, profile['default_scale'], 0):
        with image_dwm.request_scale(strength):
            result = layers[19].mlp(layers[19].self_attn(values))
        assert torch.equal(result, values) == (strength == 0)
    assert layers[18].mlp(values) is values
