import torch

from LGSSM_VAE.foundation.utils import apply_segment_laplace_attacks


def test_apply_segment_laplace_attacks_handles_small_candidate_pool():
    # Only 2 dims are ever observed, but dims_fraction requests k=H.
    B, T, H = 1, 10, 5
    x = torch.zeros(B, T, H)
    m = torch.zeros(B, T, H)
    m[:, :, 0] = 1.0
    m[:, :, 1] = 1.0

    g = torch.Generator(device="cpu")
    g.manual_seed(0)

    x_attacked, a_mask = apply_segment_laplace_attacks(
        x,
        m,
        generator=g,
        clean_fraction=0.0,
        seg_len_min=T,
        seg_len_max=T,
        dims_fraction_min=1.0,
        dims_fraction_max=1.0,
        laplace_scales=(0.5,),
    )

    assert x_attacked.shape == x.shape
    assert a_mask.shape == x.shape
    assert float(a_mask.sum().item()) > 0.0
    # Missing dims must never be attacked/modified.
    assert torch.all(a_mask[:, :, 2:] == 0)
    assert torch.all(x_attacked[:, :, 2:] == 0)
    # Attacked entries should differ from the original input.
    assert torch.any((x_attacked - x).abs() * a_mask > 0)


def test_apply_segment_laplace_attacks_clean_fraction_keeps_clean():
    B, T, H = 2, 8, 3
    x = torch.randn(B, T, H)
    m = torch.ones(B, T, H)

    g = torch.Generator(device="cpu")
    g.manual_seed(123)

    x_attacked, a_mask = apply_segment_laplace_attacks(
        x,
        m,
        generator=g,
        clean_fraction=1.0,
        seg_len_min=2,
        seg_len_max=4,
        dims_fraction_min=0.5,
        dims_fraction_max=1.0,
        laplace_scales=(0.3,),
    )

    assert torch.allclose(x_attacked, x)
    assert torch.all(a_mask == 0)

