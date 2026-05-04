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

import numpy as np
import open3d as o3d

CD_UNIT = 1e-4


def cv2gl(c2w: np.array):
    applied_transform = np.eye(4)
    applied_transform = applied_transform[np.array([1, 0, 2, 3]), :]
    applied_transform[2, :] *= -1
    return np.matmul(applied_transform, c2w)


def gl2cv(c2w: np.array):
    return cv2gl(c2w)


def write_points_pcd(points, filename):
    """Writes points to a PCD file."""
    import open3d as o3d

    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        pcd = points
    o3d.io.write_point_cloud(str(filename), pcd)


def read_pcd_file(pcd_path, ignore_nan=True, filter_ego=True, return_pcd=False):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    # Convert to numpy array
    points = np.asarray(pcd.points)
    if ignore_nan:
        points = points[~np.isnan(points).any(axis=1)]
    if filter_ego:
        def self_mask(points):
            x = points[:, 0]
            y = np.abs(points[:, 1])
            z = points[:, 2]
            return ~(
                x < 3
                & x > -1
                & y < 1
                & z < 2
                & z > -1
            )

        points = points[self_mask(points)]
    if return_pcd:
        return np2pcd(points)
    return points

def np2pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def calc_chamfer_distance(pred, gt):
    assert isinstance(gt, np.ndarray) and isinstance(pred, np.ndarray)
    assert gt.shape[1] == 3 and pred.shape[1] == 3
    gt_pcd = np2pcd(gt)
    pred_pcd = np2pcd(pred)
    dists1 = pred_pcd.compute_point_cloud_distance(gt_pcd)
    dists1 = np.asarray(dists1).mean()
    dists2 = gt_pcd.compute_point_cloud_distance(pred_pcd)
    dists2 = np.asarray(dists2).mean()
    # dists = 0.5 * (dists1 + dists2) / CD_UNIT
    return dists1 / CD_UNIT, dists2 / CD_UNIT
