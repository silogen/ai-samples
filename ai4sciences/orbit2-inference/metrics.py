# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Standalone metric functions for evaluating downscaling predictions.

All functions take (prediction, ground_truth) as numpy arrays of shape (H, W)
and return a scalar float, except where noted.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
try:
    from scipy.special import sph_harm_y as sph_harm
except ImportError:
    from scipy.special import sph_harm as _sph_harm_legacy
    def sph_harm(m, l, phi, theta):  # noqa: E741
        return _sph_harm_legacy(m, l, phi, theta)


def rms(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction.astype(np.float64) - ground_truth.astype(np.float64)) ** 2)))


def mae(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction.astype(np.float64) - ground_truth.astype(np.float64))))


def smape(prediction: np.ndarray, ground_truth: np.ndarray, eps: float = 1e-8) -> float:
    p = prediction.astype(np.float64)
    g = ground_truth.astype(np.float64)
    return float(np.mean(np.abs(p - g) / ((np.abs(p) + np.abs(g)) / 2 + eps)))


def psnr(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    data_range = ground_truth.max() - ground_truth.min()
    if data_range == 0:
        return float("nan")
    return float(peak_signal_noise_ratio(ground_truth, prediction, data_range=data_range))


def ssim(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    data_range = ground_truth.max() - ground_truth.min()
    if data_range == 0:
        return float("nan")
    return float(structural_similarity(ground_truth, prediction, data_range=data_range))


def spectrum_diff(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    pred_spec = np.abs(np.fft.fftshift(np.fft.fft2(prediction.astype(np.float64))))
    gt_spec   = np.abs(np.fft.fftshift(np.fft.fft2(ground_truth.astype(np.float64))))
    return float(np.mean(np.abs(pred_spec - gt_spec)))


def _spherical_harmonic_transform(
    grid: np.ndarray, l_max: Optional[int] = None
) -> Dict[Tuple[int, int], complex]:
    """Expand a 2D lat-lon grid (lat from 90→-90, lon from 0→360) into SH coefficients."""
    nlat, nlon = grid.shape
    if l_max is None:
        l_max = min((nlat - 1) // 2, nlon // 2, 64)
    l_max = max(0, min(l_max, (nlat - 1) // 2, nlon // 2))

    lat_deg = np.linspace(90.0, -90.0, nlat)
    lon_deg = np.linspace(0.0, 360.0, nlon, endpoint=False)
    theta = np.deg2rad(90.0 - lat_deg)
    phi   = np.deg2rad(lon_deg)
    theta_2d, phi_2d = np.meshgrid(theta, phi, indexing="ij")

    d_theta = np.pi / max(1, nlat - 1)
    d_phi   = 2.0 * np.pi / nlon
    weights = np.sin(theta_2d) * d_theta * d_phi

    grid_f  = grid.astype(np.float64)
    coeffs  = {}
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            y_lm = sph_harm(m, l, phi_2d, theta_2d)
            coeffs[(l, m)] = np.sum(weights * grid_f * np.conj(y_lm))
    return coeffs


def _sh_coeffs_to_energy_spectrum(coeffs: Dict[Tuple[int, int], complex]) -> np.ndarray:
    """E_l = sum_m |C_lm|^2 for each degree l."""
    if not coeffs:
        return np.array([])
    l_max = max(l for l, _ in coeffs)
    e_l = np.zeros(l_max + 1, dtype=np.float64)
    for (l, m), c in coeffs.items():
        e_l[l] += np.abs(c) ** 2
    return e_l


def sh_l2(prediction: np.ndarray, ground_truth: np.ndarray, l_max: int = 10) -> float:
    """L2-norm of the difference in spherical harmonic coefficients."""
    pred_sh = _spherical_harmonic_transform(prediction, l_max=l_max)
    gt_sh   = _spherical_harmonic_transform(ground_truth, l_max=l_max)
    keys    = set(pred_sh) & set(gt_sh)
    return float(np.sqrt(sum(np.abs(pred_sh[k] - gt_sh[k]) ** 2 for k in keys)))


def sh_energy_mismatch(prediction: np.ndarray, ground_truth: np.ndarray, l_max: int = 10) -> float:
    """L2 difference of the per-degree energy spectra."""
    pred_sh = _spherical_harmonic_transform(prediction, l_max=l_max)
    gt_sh   = _spherical_harmonic_transform(ground_truth, l_max=l_max)
    e_pred  = _sh_coeffs_to_energy_spectrum(pred_sh)
    e_gt    = _sh_coeffs_to_energy_spectrum(gt_sh)
    n       = min(len(e_pred), len(e_gt))
    if n == 0:
        return 0.0
    return float(np.sqrt(np.sum((e_pred[:n] - e_gt[:n]) ** 2)))


# Registry of all metrics in evaluation order
METRICS = {
    "rms":                rms,
    "mae":                mae,
    "smape":              smape,
    "psnr":               psnr,
    "ssim":               ssim,
    "spectrum_diff":      spectrum_diff,
    "sh_l2":              sh_l2,
    "sh_energy_mismatch": sh_energy_mismatch,
}
