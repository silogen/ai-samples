# Copyright 2026 Advanced Micro Devices, Inc.  All rights reserved.
 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
 
#       http://www.apache.org/licenses/LICENSE-2.0
 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
import torch

from gsplat import rasterization

from street_gaussians_ns.data.cameras import Cameras


torch.compiler.allow_in_graph(rasterization)
# Without this we get:
#  TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. 
#  Consider setting `torch.set_float32_matmul_precision('high')` for better performance
torch.set_float32_matmul_precision("high")


@torch.compile(mode="reduce-overhead")
def get_viewmat(camera: Cameras, downscale_factor: int, optimized_camera_to_world):
    optimized_camera_to_world = optimized_camera_to_world[0, ...]
    device = optimized_camera_to_world.device

    # calculate the FOV of the camera given fx and fy, width and height
    cx = camera.cx
    cy = camera.cy
    # Prepare camera intrinsics
    fx, fy = camera.fx, camera.fy
    
    if downscale_factor != 1.:
        # For efficiency avoid changing the camera and then changing back again, but
        #  just scale the parameters here
        #camera.rescale_output_resolution(1 / camera_downscale)
        camera_scale = 1 / downscale_factor
        fx *= camera_scale
        fy *= camera_scale
        cx *= camera_scale
        cy *= camera_scale
    K = torch.zeros((3, 3), device=device, dtype=torch.float32)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    K[2, 2] = 1.

    # shift the camera to center of scene looking at center
    R = optimized_camera_to_world[:3, :3]  # 3 x 3
    T = optimized_camera_to_world[:3, 3:4]  # 3 x 1

    # flip the z and y axes to align with gsplat conventions
    R_edit = torch.diag(torch.tensor([1, -1, -1], device=device, dtype=R.dtype))
    R = R @ R_edit
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.T
    T_inv = -R_inv @ T
    viewmat = torch.eye(4, device=device, dtype=R.dtype)
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv

    return K, viewmat


def render_outputs(
        camera_width, 
        camera_height,
        K: torch.Tensor,
        viewmat: torch.Tensor,
        means: torch.Tensor, 
        quats: torch.Tensor, 
        scales: torch.Tensor, 
        opacities: torch.Tensor, 
        features_dc: torch.Tensor,
        features_rest: Optional[torch.Tensor] = None,
        sky_capture=None,
        output_names=None,
        absgrad : bool = False,
        sh_degree : Optional[int] = None,
        training : bool = False,
) -> tuple[dict[str, torch.Tensor], dict]:
    """
    Generic rendering for a given set of gaussian parameters.
    Used by both the full scene rendering and submodel rendering.
    
    :param camera: Camera to render viewpoint from
    :type camera: Cameras
    :param means: Tensor of gaussian means
    :type means: torch.Tensor
    :param quats: Tensor of quaternion parameters for each gaussian
    :type quats: torch.Tensor
    :param scales: 3D scales for each gaussian
    :type scales: torch.Tensor
    :param opacities: Opacities for each gaussian
    :type opacities: torch.Tensor
    :param features_dc: Spherical harmonic features
    :type features_dc: torch.Tensor
    :param features_rest: Spherical harmonic features. If None, features_dc is treated as a colors matrix
    :type features_rest: torch.Tensor
    :param sky_capture: Rendering of sky to superimpose the scene over
    :param output_names: Outputs to select: may contain 'rgb', 'accumulation', 'depth'. Default: all
    :return: Outputs dict and meta dict
    """
    if output_names is None:
        output_names = ["rgb", "accumulation", "depth"]

    if len(means) == 0:
        return render_empty(camera_width, camera_height, means.device, output_names, sky_capture)
    
    if features_rest is None:
        colors = features_dc
    else:
        colors = torch.cat((features_dc, features_rest), dim=1)

    sh_degree = None if sh_degree == 0 else sh_degree

    renders, alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=torch.exp(scales),
        opacities=torch.sigmoid(opacities).squeeze(-1),  # Convert from logit to probability space
        colors=colors,
        viewmats=viewmat[None, ...],
        Ks=K[None, ...],
        width=camera_width,
        height=camera_height,
        sh_degree=sh_degree,
        render_mode="RGB+ED" if "depth" in output_names else "RGB",
        absgrad=absgrad,
        packed=False,
    )

    rgb = renders[..., :3][0]
    alpha = alphas[0]

    meta["viewmat"] = viewmat

    if sky_capture is not None:
        rgb = rgb * alpha + sky_capture * (1 - alpha)

    if not training:
        rgb = rgb.clamp(0., 1.)

    outputs = {}
    if "rgb" in output_names:
        outputs["rgb"] = rgb
    if "accumulation" in output_names:
        outputs["accumulation"] = alpha
    if "depth" in output_names:
        outputs["depth"] = renders[..., 3:4][0]

    return outputs, meta


def render_empty(camera_width, camera_height, device, output_names=None, sky_capture=None):
    if output_names is None:
        output_names = ["rgb", "accumulation"]
    empty = torch.zeros(camera_height, camera_width, 1, device=device)
    outputs = {}
    if "accumulation" in output_names:
        outputs["accumulation"] = empty
    if "rgb" in output_names:
        if sky_capture is None:
            outputs["rgb"] = torch.zeros(camera_height, camera_width, 3, device=device)
        else:
            outputs["rgb"] = sky_capture
    if "depth" in output_names:
        outputs["depth"] = empty
    return outputs, {}

