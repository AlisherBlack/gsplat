# SPDX-License-Identifier: Apache-2.0
"""RaysStructuredLidarModelParametersExt — first-class rays-defined lidar model.

Self-contained tests (no dependency on the other test modules) for the facade
that hides the spinning scaffolding behind a rays(+times) constructor:

1. equivalence — the class produces bit-identical tiling / angles map to the
   manual assembly (fit scaffolding -> compute_tiling(element_angles) ->
   compute_angles_to_values_map) it replaces;
2. spin-direction auto-inference for CW and CCW panoramas (+ explicit override);
3. timestamps_rel=None falls back to the column-index map;
4. valid_mask: full tile coverage, invalid elements land in their nominal cell;
5. azimuth recentering: a panorama crossing the ±pi seam constructs correctly
   and every element sits in the tile of its real angle;
6. interface smoke: usable wherever the spinning Ext is expected.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from gsplat.cuda._lidar import (
    SpinningDirection,
    angles_to_tile_indices,
    clamp_element_angles_to_fov,
    compute_angles_to_columns_map,
    compute_angles_to_values_map,
    compute_tiling,
)

# The public class lives in _wrapper (it inherits ``to_cpp`` and satisfies the
# isinstance check in fully_fused_projection_with_ut); _lidar only hosts the
# assembly function.
from gsplat.cuda._wrapper import (
    RaysStructuredLidarModelParametersExt,
    RowOffsetStructuredSpinningLidarModelParametersExt,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_ROWS = 8
N_COLS = 64
TILING_KW = dict(
    n_bins_elevation=4,
    max_pts_per_tile=16,
    resolution_elevation=200,
    densification_factor_azimuth=2,
)


def _angles_to_dirs(az: np.ndarray, el: np.ndarray) -> np.ndarray:
    cos_el = np.cos(el)
    return np.stack([np.cos(az) * cos_el, np.sin(az) * cos_el, np.sin(el)], axis=-1)


def _nonseparable_angles(
    az_start: float = 1.0,
    az_end: float = -1.0,
    el_start: float = 0.20,
    el_end: float = -0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Separable grid + per-row azimuth skew + per-column elevation ripple
    (mimics a galvo/prism scan). Azimuths run az_start -> az_end across columns."""
    el = np.linspace(el_start, el_end, N_ROWS)[:, None].repeat(N_COLS, axis=1)
    az = np.linspace(az_start, az_end, N_COLS)[None, :].repeat(N_ROWS, axis=0)
    rows = np.arange(N_ROWS, dtype=np.float64)
    cols = np.arange(N_COLS, dtype=np.float64)
    az = az + ((rows / (N_ROWS - 1) - 0.5) * 0.12 * abs(az_end - az_start))[:, None]
    el = el + (np.sin(2 * np.pi * cols / N_COLS) * 0.08 * abs(el_start - el_end))[None, :]
    return az, el


def _row_major_times() -> np.ndarray:
    n = N_ROWS * N_COLS
    return (np.arange(n, dtype=np.float64) / (n - 1)).reshape(N_ROWS, N_COLS)


def _class_preprocessed_angles(
    dirs_f32: torch.Tensor, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the class's own preprocessing bit-exactly:
    float32 -> float64, normalize, el = arcsin(clip(z)), az = atan2(y, x),
    recenter az around the circular mean of the valid elements.
    Used so equivalence/tile assertions do not depend on trig round-trips."""
    dirs = dirs_f32.to(torch.float64).cpu().numpy()
    dirs = dirs / np.linalg.norm(dirs, axis=-1).clip(1e-8)[..., None]
    el = np.arcsin(np.clip(dirs[..., 2], -1.0, 1.0))
    az = np.arctan2(dirs[..., 1], dirs[..., 0])
    az_valid = az[valid]
    center = float(np.arctan2(np.mean(np.sin(az_valid)), np.mean(np.cos(az_valid))))
    az = center + (az - center + np.pi) % (2.0 * np.pi) - np.pi
    return az, el


def _tile_of_each_element(tiling, device) -> torch.Tensor:
    tile_of = torch.full((N_ROWS * N_COLS,), -1, dtype=torch.long, device=device)
    tte = tiling.tiles_to_elements_map.to(device)
    pack = tiling.tiles_pack_info.to(device)
    for t in range(pack.shape[0]):
        off, cnt = int(pack[t, 0]), int(pack[t, 1])
        if cnt == 0:
            continue
        elems = tte[off : off + cnt]
        flat = elems[:, 1].long() * N_COLS + elems[:, 0].long()
        tile_of[flat] = t
    return tile_of


def _expected_tile(model, tiling, angles) -> torch.Tensor:
    clamped = clamp_element_angles_to_fov(model, angles.reshape(-1, 2))
    return angles_to_tile_indices(
        model,
        clamped,
        n_bins_azimuth=tiling.n_bins_azimuth,
        n_bins_elevation=tiling.n_bins_elevation,
        cdf_elevation=tiling.cdf_elevation,
    ).long()


# --------------------------------------------------------------------------
# 1. Equivalence with the manual assembly the class replaces.
# --------------------------------------------------------------------------


def test_equivalent_to_manual_assembly():
    az0, el0 = _nonseparable_angles()
    dirs = torch.tensor(_angles_to_dirs(az0, el0), dtype=torch.float32)
    t_rel = torch.tensor(_row_major_times(), dtype=torch.float64)

    model = RaysStructuredLidarModelParametersExt(
        dirs, timestamps_rel=t_rel, device=DEVICE, **TILING_KW
    )

    # Manual assembly on the SAME scaffolding the class fitted internally
    # (the model IS the fitted base params via inheritance) and the SAME
    # preprocessed angles/rays.
    valid = np.ones((N_ROWS, N_COLS), dtype=bool)
    az, el = _class_preprocessed_angles(dirs, valid)
    element_angles = torch.tensor(
        np.stack([az, el], axis=-1).reshape(-1, 2), dtype=torch.float32, device=DEVICE
    )
    manual_tiling = compute_tiling(model, element_angles=element_angles, **TILING_KW)

    values = torch.tensor(
        np.rint(_row_major_times().reshape(-1) * (N_COLS - 1)).astype(np.int64),
        device=DEVICE,
    )
    dirs64 = dirs.to(torch.float64).cpu().numpy()
    dirs64 = dirs64 / np.linalg.norm(dirs64, axis=-1).clip(1e-8)[..., None]
    element_rays = torch.tensor(
        dirs64.reshape(-1, 3), dtype=torch.float32, device=DEVICE
    )
    manual_map = compute_angles_to_values_map(model, element_rays, values)

    assert torch.equal(model.angles_to_columns_map.cpu(), manual_map.cpu())
    assert model.tiling.n_bins_azimuth == manual_tiling.n_bins_azimuth
    assert torch.equal(
        model.tiling.cdf_elevation.cpu(), manual_tiling.cdf_elevation.cpu()
    )
    assert torch.equal(
        model.tiling.tiles_pack_info.cpu(), manual_tiling.tiles_pack_info.cpu()
    )
    assert torch.equal(
        model.tiling.tiles_to_elements_map.cpu(),
        manual_tiling.tiles_to_elements_map.cpu(),
    )
    assert torch.equal(
        model.tiling.cdf_dense_ray_mask.cpu(), manual_tiling.cdf_dense_ray_mask.cpu()
    )


# --------------------------------------------------------------------------
# 2. Spin-direction inference.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "az_start,az_end,expected",
    [
        (1.0, -1.0, SpinningDirection.CLOCKWISE),  # azimuth decreasing
        (-1.0, 1.0, SpinningDirection.COUNTER_CLOCKWISE),  # azimuth increasing
    ],
)
def test_spin_direction_is_inferred(az_start, az_end, expected):
    az, el = _nonseparable_angles(az_start=az_start, az_end=az_end)
    dirs = torch.tensor(_angles_to_dirs(az, el), dtype=torch.float32)
    model = RaysStructuredLidarModelParametersExt(dirs, device=DEVICE, **TILING_KW)
    assert model.spinning_direction == expected

    forced = RaysStructuredLidarModelParametersExt(
        dirs, spinning_direction=expected, device=DEVICE, **TILING_KW
    )
    assert forced.spinning_direction == expected


def test_wrong_forced_direction_fails_clearly():
    az, el = _nonseparable_angles()  # CW panorama
    dirs = torch.tensor(_angles_to_dirs(az, el), dtype=torch.float32)
    with pytest.raises(AssertionError, match="not ordered"):
        RaysStructuredLidarModelParametersExt(
            dirs,
            spinning_direction=SpinningDirection.COUNTER_CLOCKWISE,
            device=DEVICE,
            **TILING_KW,
        )


def test_unsorted_rows_fail_clearly():
    az, el = _nonseparable_angles()
    el = el[::-1].copy()  # ascending mean elevation — wrong order
    dirs = torch.tensor(_angles_to_dirs(az, el), dtype=torch.float32)
    with pytest.raises(AssertionError, match="descending"):
        RaysStructuredLidarModelParametersExt(dirs, device=DEVICE, **TILING_KW)


# --------------------------------------------------------------------------
# 3. No timestamps -> column-index map (global-shutter binning semantics).
# --------------------------------------------------------------------------


def test_no_timestamps_falls_back_to_columns_map():
    az, el = _nonseparable_angles()
    dirs = torch.tensor(_angles_to_dirs(az, el), dtype=torch.float32)
    model = RaysStructuredLidarModelParametersExt(dirs, device=DEVICE, **TILING_KW)
    columns_map = compute_angles_to_columns_map(model)
    assert torch.equal(model.angles_to_columns_map.cpu(), columns_map.cpu())


# --------------------------------------------------------------------------
# 4. valid_mask: coverage + nominal fallback.
# --------------------------------------------------------------------------


def test_valid_mask_full_coverage_and_nominal_fallback():
    az0, el0 = _nonseparable_angles()
    dirs_np = _angles_to_dirs(az0, el0)
    valid = np.ones((N_ROWS, N_COLS), dtype=bool)
    valid[0, :8] = False
    valid[3, ::5] = False
    valid[-1, -1] = False
    dirs_np = dirs_np.copy()
    dirs_np[~valid] = 0.0  # garbage measurements must not leak anywhere
    dirs = torch.tensor(dirs_np, dtype=torch.float32)

    model = RaysStructuredLidarModelParametersExt(
        dirs,
        timestamps_rel=torch.tensor(_row_major_times()),
        valid_mask=torch.tensor(valid),
        device=DEVICE,
        **TILING_KW,
    )

    tile_of = _tile_of_each_element(model.tiling, model.device)
    assert torch.all(tile_of >= 0), "every element (incl. invalid) must be in a tile"
    assert int(model.tiling.tiles_pack_info[:, 1].sum()) == N_ROWS * N_COLS

    # Valid elements sit in the tile of their REAL angle...
    az, el = _class_preprocessed_angles(dirs, valid)
    real_angles = torch.tensor(
        np.stack([az, el], axis=-1), dtype=torch.float32, device=model.device
    )
    expected_real = _expected_tile(model, model.tiling, real_angles)
    valid_flat = torch.tensor(valid.reshape(-1), device=model.device)
    assert torch.equal(tile_of[valid_flat], expected_real[valid_flat])

    # ...and invalid ones in the tile of their NOMINAL cell angle.
    nominal_az = (
        model.column_azimuths_rad.cpu().numpy().astype(np.float64)[None, :]
        + model.row_azimuth_offsets_rad.cpu().numpy().astype(np.float64)[:, None]
    )
    nominal_el = np.broadcast_to(
        model.row_elevations_rad.cpu().numpy().astype(np.float64)[:, None],
        (N_ROWS, N_COLS),
    )
    nominal_angles = torch.tensor(
        np.stack([nominal_az, nominal_el], axis=-1),
        dtype=torch.float32,
        device=model.device,
    )
    expected_nominal = _expected_tile(model, model.tiling, nominal_angles)
    assert torch.equal(tile_of[~valid_flat], expected_nominal[~valid_flat])


# --------------------------------------------------------------------------
# 5. Azimuth recentering across the ±pi seam.
# --------------------------------------------------------------------------


def test_panorama_crossing_pi_seam():
    # Panorama centered at pi: raw atan2 azimuths jump between ~+pi and ~-pi.
    az0, el0 = _nonseparable_angles(az_start=math.pi + 1.0, az_end=math.pi - 1.0)
    dirs = torch.tensor(_angles_to_dirs(az0, el0), dtype=torch.float32)

    model = RaysStructuredLidarModelParametersExt(
        dirs,
        timestamps_rel=torch.tensor(_row_major_times()),
        device=DEVICE,
        **TILING_KW,
    )

    # FOV must span ~2 rad (the panorama), not ~2*pi (a wrapped mess).
    assert model.fov_horiz_rad.span < 3.0

    # Every element sits in the tile of its real (recentered) angle.
    valid = np.ones((N_ROWS, N_COLS), dtype=bool)
    az, el = _class_preprocessed_angles(dirs, valid)
    tile_of = _tile_of_each_element(model.tiling, model.device)
    real_angles = torch.tensor(
        np.stack([az, el], axis=-1), dtype=torch.float32, device=model.device
    )
    expected = _expected_tile(model, model.tiling, real_angles)
    assert torch.equal(tile_of, expected)


# --------------------------------------------------------------------------
# 6. Interface smoke.
# --------------------------------------------------------------------------


def test_is_a_drop_in_ext():
    az, el = _nonseparable_angles()
    dirs = torch.tensor(_angles_to_dirs(az, el), dtype=torch.float32)
    model = RaysStructuredLidarModelParametersExt(
        dirs,
        timestamps_rel=torch.tensor(_row_major_times()),
        device=DEVICE,
        **TILING_KW,
    )
    # The exact isinstance fully_fused_projection_with_ut asserts on (the
    # _wrapper class, NOT the _lidar base), plus the CUDA conversion hook.
    assert isinstance(model, RowOffsetStructuredSpinningLidarModelParametersExt)
    assert hasattr(model, "to_cpp")
    assert (model.n_rows, model.n_columns) == (N_ROWS, N_COLS)
    assert model.angles_to_columns_map.shape == (4 * N_ROWS, 4 * N_COLS)
    assert model.angles_to_columns_map.dtype == torch.int32
    assert model.tiling.tiles_to_elements_map.dtype == torch.int32
    # Rows descending, scaffolding valid.
    assert torch.all(torch.diff(model.row_elevations_rad) < 0)
    # Importable from the package root.
    import gsplat

    assert gsplat.RaysStructuredLidarModelParametersExt is RaysStructuredLidarModelParametersExt
