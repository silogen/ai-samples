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

import torch
import torch.nn.functional as F


class EnvLight(torch.nn.Module):
    def __init__(self, resolution=1024):
        super().__init__()
        self.base = torch.nn.Parameter(
            0.5 * torch.ones(6, resolution, resolution, 3, requires_grad=True),
        )
    
    def get_world_directions(self, W: int, H: int, cx, cy, fx, fy, c2w, train=False):
        y_range = torch.arange(int(H), dtype=torch.float32, device=c2w.device)
        x_range = torch.arange(int(W), dtype=torch.float32, device=c2w.device)
        v, u = torch.meshgrid(y_range, x_range, indexing='ij')

        # Create 3D unit vectors pointing out into a grid whose size (i.e. separation of the angles)
        # is determined by the focal length
        if train:
            directions = torch.stack(
                [(u-cx+torch.rand_like(u))/fx, -(v-cy+torch.rand_like(v))/fy, -torch.ones_like(u)],
                dim=-1
            )
        else:
            directions = torch.stack(
                [(u-cx+0.5)/fx, -(v-cy+0.5)/fy, -torch.ones_like(u)],
                dim=-1
            )
        directions = F.normalize(directions, dim=-1)  # (H,W,3)
        # Rotate all directions to point the same way as the camera
        directions = directions @ c2w[:3, :3].T       # (H,W,3)
        return directions

    def forward(self, W: int, H: int, cx, cy, fx, fy, c2w, train=False):
        direction = self.get_world_directions(W, H, cx, cy, fx, fy, c2w, train)  # (H, W, 3)  [3: xyz vec]
        direction = direction.contiguous()

        # We could just store the parameter (base) in this order, but maintain this permutation so that old models can be loaded
        cubemap = self.base.permute(0, 3, 1, 2)

        light = sample_cubemap(cubemap, direction)

        return light


def sample_cubemap(cubemap, dirs):
    """
    Sample from a cubic texture constructed from 6 square faces defined in the
    tensor `cubemap`.

    cubemap: (6, C, R_x, R_y), 6 faces of cube, C color channels, (R_x, R_y) resolution of faces
    dirs: (H, W, 3), directions from cube center to sample from the cube surface, given as unit
        vectors for each pixel of a (H, W) image
    returns: (H, W, C), color values for each pixel
    """

    H, W, _ = dirs.shape
    C = cubemap.shape[1]
    device = dirs.device

    x, y, z = dirs.unbind(-1)
    ax, ay, az = x.abs(), y.abs(), z.abs()

    face = torch.zeros_like(x, dtype=torch.long)
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)

    # X faces
    mask = (ax >= ay) & (ax >= az)
    face[mask & (x > 0)] = 0
    face[mask & (x < 0)] = 1
    u[mask] = -z[mask] / ax[mask]
    v[mask] = -y[mask] / ax[mask]

    # Y faces
    mask = (ay > ax) & (ay >= az)
    face[mask & (y > 0)] = 2
    face[mask & (y < 0)] = 3
    u[mask] = x[mask] / ay[mask]
    v[mask] = z[mask] / ay[mask]

    # Z faces
    mask = (az > ax) & (az > ay)
    face[mask & (z > 0)] = 4
    face[mask & (z < 0)] = 5
    u[mask] = x[mask] / az[mask]
    v[mask] = -y[mask] / az[mask]

    # grid_sample coords
    grid = torch.stack([u, v], dim=-1)  # (H,W,2)

    out = torch.zeros(H, W, C, device=device)

    for f in range(6):
        mask = face == f
        if not mask.any():
            continue

        # Extract grid for this face
        g = grid[mask].reshape(1, -1, 1, 2)  # (1,F,1,2), where F is num masked coordinates
        tex = cubemap[f:f+1]  # (1,C,H,W)

        samp = F.grid_sample(
            tex, g,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        out[mask] = samp.view(C, -1).T

    return out
