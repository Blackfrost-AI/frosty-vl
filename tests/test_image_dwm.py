from __future__ import annotations

import torch

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

