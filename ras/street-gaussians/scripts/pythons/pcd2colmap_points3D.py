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

import argparse
import random
import numpy as np
import json
import os
import open3d as o3d
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


def get_box_corners(center, dimensions, orientation):
    length, width, height = dimensions
    q = orientation

    # Generate vertices of an axis-aligned bounding box
    dx  = length / 2.0
    dy = width / 2.0
    dz = height / 2.0

    corners = np.array(
        [
            [dx, dy, dz],
            [-dx, dy, dz],
            [-dx, -dy, dz],
            [dx, -dy, dz],
            [dx, dy, -dz],
            [-dx, dy, -dz],
            [-dx, -dy, -dz],
            [dx, -dy, -dz],
        ]
    )

    # Create rotation from quaternion and apply to vertices
    rotation = R.from_quat([q[1], q[2], q[3], q[0]])  # Note: quaternion order is [x, y, z, w]
    rotated_corners = rotation.apply(corners)

    # Add local coordinates to center point to get world coordinates
    world_corners = rotated_corners + center

    return world_corners

def undistort_nearest(cv_image, k, d, fisheye=True):
    if fisheye:
       mapx, mapy = cv2.fisheye.initUndistortRectifyMap(k, d, None, k, (cv_image.shape[1], cv_image.shape[0]), cv2.CV_32FC1)
    else:
       mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (cv_image.shape[1], cv_image.shape[0]), cv2.CV_32FC1)

    cv_image_undistorted = cv2.remap(cv_image, mapx, mapy, cv2.INTER_NEAREST)

    return cv_image_undistorted
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="/home/ubuntu/project/nerf/nerfstudio/data/waymo")
    parser.add_argument("--meta_file", default="transform.json")
    parser.add_argument("--main_lidar_in_transforms", default="lidar_TOP")
    parser.add_argument("--world_coordinate", action="store_true")
    parser.add_argument("--annotation_file", type=str, default="annotation.json")

    args = parser.parse_args()
    root_path = args.root_path+"/"
    sensor_path = root_path+args.meta_file
    annotation_path = root_path + args.annotation_file
    moving_gids = []
    with open(sensor_path, "r") as f:
        data = json.load(f)
    annotation_frames = []
    if os.path.exists(annotation_path):
        with open(annotation_path, "r") as f:
            annotation_data = json.load(f)
        annotation_frames = annotation_data["frames"]
    c2w0 = np.array(data["frames"][0]["transform_matrix"])

    T0 = (c2w0[:3,3])*0.98
    T0_OPENCV = np.array([T0[1], T0[0],-1*T0[2]])
    with open(root_path+'colmap/sparse/lidar/points3D.txt','w') as j:
        i = 1
        frames = data["frames"]
        lidar_frames = data["lidar_frames"]
        
        for lidar_frame in tqdm(lidar_frames):
            if (lidar_frame["lidar"] == args.main_lidar_in_transforms ):
                found_dict =  [d for d in frames if str(d.get('timestamp')) in str(lidar_frame["timestamp"])]
                annotation_found_dict = next((d for d in annotation_frames if str(d.get("timestamp")) in str(lidar_frame["timestamp"])), None)

                obbs = []
                if annotation_found_dict is not None:
                    for object in annotation_found_dict["objects"]:
                        if object["is_moving"] or object["gid"] in moving_gids:
                            translation = object["translation"]
                            lwh = object["size"]
                            rotation = object["rotation"]
                            world_corners = get_box_corners(translation, lwh, rotation)
                            world_corners[:] -= T0_OPENCV
                            obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(world_corners))
                            scale_x = 1.1
                            scale_y = 1.1
                            scale_z = 1.1
                            extents = np.array(obb.extent) * np.array([scale_x, scale_y, scale_z]) # Update bounding box size
                            obb = o3d.geometry.OrientedBoundingBox(obb.center,  obb.R, extents)
                            obbs.append(obb)
                            # print("OBB center:", obb.center)
                            # print("OBB extent:", obb.extent)
                            # print("OBB rotation matrix:", obb.R)
                
                l2w = np.array(lidar_frame["transform_matrix"]) 
                # opencv2waymo = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
                #  extrinsic[:3, :3] = extrinsic[:3, :3] @ opencv2waymo
                l2w[0:3, 1:3] *= -1       
                l2w = l2w[np.array([1, 0, 2, 3]), :]
                l2w[2, :] *= -1                   
                l2w[:3,3] -= T0
                l2w[2, :] *= -1
                l2w = l2w[np.array([1, 0, 2, 3]), :]
                l2w[0:3, 1:3] *= -1
                
                for frame in found_dict:
                    # Read pcd
                    file_path = root_path + lidar_frame["file_path"]
                # points = np.fromfile(file=file_path, dtype=np.float32, count=-1).reshape([-1,5])[:,0:3]
                    pcd_data = o3d.io.read_point_cloud(file_path)
                    points = np.array(pcd_data.points)
                    indices = points[:, 2] > -2
                    points = points[indices]
                    nan_rows = np.isnan(points).any(axis=1)

                    # Use boolean indexing to remove rows containing NaN values
                    points = points[~nan_rows]
                    if points.shape[0] > 10000:
                        # Randomly generate 10000 unique indices
                        indices = np.random.choice(points.shape[0], 10000, replace=False)
                        # Use these indices to select elements from the array
                        points = points[indices]
                    #print(points.shape)
                    # Represent each point as homogeneous coordinates (x, y, z, 1)
                    homogeneous_positions = np.hstack([points , np.ones((points.shape[0], 1))])
                    transformed_positions = np.dot(l2w, homogeneous_positions.T).T[:,:3]
                    
                    if (len(obbs) > 0):
                        pcds = o3d.geometry.PointCloud()
                        pcds.points = o3d.utility.Vector3dVector(transformed_positions)
                        for obb in obbs:
                            inliers_indices = obb.get_point_indices_within_bounding_box(pcds.points)
                            outliers_pcd =  pcds.select_by_index(inliers_indices, invert=True) #select outside points
                            pcds = outliers_pcd
                        transformed_positions = np.array(outliers_pcd.points)
                        
                    rgb=cv2.imread(root_path + frame["file_path"]
                        )
                    c2w = np.array(frame["transform_matrix"])
                    c2w[:3,3] -= T0
                    c2w[2, :] *= -1
                    c2w = c2w[np.array([1, 0, 2, 3]), :]
                    c2w[0:3, 1:3] *= -1

                    w2c = np.linalg.inv(c2w)
                    h=frame['h']
                    w=frame['w']
                    fl_x=frame['fl_x']
                    fl_y=frame['fl_y']
                    cx=frame['cx']
                    cy=frame['cy']
                    ## Undistort
                    if (frame['camera_model']=="OPENCV"):
                        k = np.asarray([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])
                        d = np.asarray([frame['k1'],frame['k2'],frame['p1'],frame['p2']])
                        rgb = undistort_nearest(rgb, k, d,False)
                    elif (frame['camera_model']=="OPENCV_FISHEYE"):
                        k = np.asarray([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])
                        d = np.asarray([frame['k1'],frame['k2'],frame['k3'],frame['k4']])
                        rgb = undistort_nearest(rgb, k, d, True)                       
                    
                    intrinsic_matrix=np.array([[fl_x,0,cx,0],
                                            [0,fl_y,cy,0],
                                            [0,0,1,0],
                                            [0,0,0,1]])  
                            
                    # Extract the first three columns from the result, remove homogeneous coordinates
                    # final_positions = transformed_positions[:, :3]-np.array([l2w[0,3],l2w[1,3],l2w[2,3]])
                    # print("final_max_X:" , np.max(final_positions[:,2]))
                    
                    for m in transformed_positions:
                        if abs(m[0]) >100000:
                            continue
                        m_1= np.array([m[0],m[1],m[2],1])
                        #print(m_1)
                        # Projective division, only convert to int at the last step
                        uv_homogeneous = intrinsic_matrix @ w2c @ m_1
                        u, v = (uv_homogeneous[:2] / uv_homogeneous[2]).astype(int)
                        #print (u,v)
                        # Check if coordinates are within valid image range
                        if 0 <= u < w and 0 < v < h and uv_homogeneous[2]>0:
                            #rgb_point = rgb[v, u]
                            rgb_point = rgb[v, u]
                            error = random.uniform(0,1)
                            # Output point info and some random values

                            j.write(f'{i} {m[0]:.3f} {m[1]:.3f} {m[2]:.3f} {rgb_point[2]} {rgb_point[1]} {rgb_point[0]} {error:.3f} 1 1 2 2 {random.randint(1,300)} {random.randint(1,2000)}\n')
                            i += 1