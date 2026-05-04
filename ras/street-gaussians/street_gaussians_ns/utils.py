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

from pathlib import Path
import torch
import math
import torchvision.utils as vutils
import numpy as np
from street_gaussians_ns.data.utils.pytorch3d_functions import quaternion_multiply


def num_sh_bases(degree: int):
    if degree == 0:
        return 1
    if degree == 1:
        return 4
    if degree == 2:
        return 9
    if degree == 3:
        return 16
    return 25


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def idft(time, dim):
    """
    Computes the inverse discrete Fourier transform (IDFT) of a given time signal.
    """
    if isinstance(time, float):
        time = torch.tensor(time)
    t = time.view(-1, 1)
    idft = torch.zeros(t.shape[0], dim, dtype=t.dtype, device=t.device)
    indices = torch.arange(dim, dtype=torch.int, device=t.device)
    even_indices = indices[::2]
    odd_indices = indices[1::2]
    idft[:, even_indices] = torch.cos(t * even_indices * 2 * math.pi / dim)
    idft[:, odd_indices] = torch.sin(t * (odd_indices + 1) * 2 * math.pi / dim)
    return idft


def save_rgb_image(rgb: torch.Tensor, filename: str|Path = "output_rgb.png"):
    """
    Saves the rgb tensor as an image file.

    Args:
        rgb: torch.Tensor of shape [H, W, 3], values in [0, 1]
        filename: Output filename
    """
    # Move to CPU and clamp values
    img = rgb.detach().cpu().clamp(0, 1).permute(2, 0, 1)  # [3, H, W]
    vutils.save_image(img, str(filename))


def k_nearest_sklearn(x: torch.Tensor, k: int):
    """
    Find k-nearest neighbors using sklearn's NearestNeighbors.

    x: The data tensor of shape [num_samples, num_features]
    k: The number of neighbors to retrieve
    
    """
    # Convert tensor to numpy array
    x_np = x.cpu().numpy()

    # Build the nearest neighbors model
    from sklearn.neighbors import NearestNeighbors

    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

    # Find the k-nearest neighbors
    distances, indices = nn_model.kneighbors(x_np)

    # Exclude the point itself from the result and return
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)


@torch.compile(dynamic=True)
def object2world_gs(means, quats, anno):
    """
    Transform the object GS from object to world coordinate system
    
    """
    assert means.dim() == 2 and means.shape[1] == 3
    assert quats.dim() == 2 and quats.shape[1] == 4

    # TODO Original SG does not backpropagate to bboxes (annos): it converts rot mat -> quat using numpy
    # Therefore we follow this behaviour (center and quat are detached during apply_to_bbox), but I think this stops bbox optimization from working
    # Later, try removing this and test whether bbox optimization works
    # Do this by updating the apply_to_bbox method so it doesn't detach its results

    # Tsransform the object GS from object to world coordinate system
    means_w = torch.matmul(means, anno.rot.T) + anno.center[None, :] #[N_pts,3]
    quat_w = quaternion_multiply(anno.quat, quats)

    return [means_w.squeeze(), quat_w.squeeze()]
