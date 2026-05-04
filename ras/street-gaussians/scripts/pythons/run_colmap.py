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

import sqlite3
import os
import numpy as np
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import sys
import json
sys.path.append(os.getcwd())
from scipy.spatial.transform import Rotation as R


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [  # type: ignore
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def run_colmap_waymo(args):    
    root = args.root
    os.system(f'{sys.executable} scripts/pythons/transform2colmap.py \
              --input_path {root}')
    meta_path = os.path.join(root, 'transform.json')
    colmap_dir = os.path.join(root, 'colmap')
    os.makedirs(colmap_dir, exist_ok=True)
    print('runing colmap, colmap dir: ', colmap_dir)
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    unique_cams = meta['sensor_params']['camera_order']
    print('cameras: ', unique_cams)
    
    mask_dir = os.path.join(root, 'masks')
    images_dir = os.path.join(root, 'images')
  
    # load extrinsic and image filenames        
    c2w_dict = dict()
    for frame in meta['frames']:
        name = frame['file_path'].replace('images/', '')
        c2w_dict[name] = frame['transform_matrix']

    # https://colmap.github.io/faq.html#mask-image-regions
    os.system(f'colmap feature_extractor \
            --ImageReader.mask_path {mask_dir} \
            --ImageReader.camera_model SIMPLE_PINHOLE  \
            --ImageReader.single_camera_per_folder 1 \
            --database_path {colmap_dir}/database.db \
            --image_path {images_dir} \
            --SiftExtraction.use_gpu 0')

    # load intrinsic
    camera_infos = dict()
    for unique_cam in unique_cams:
        img_h = meta['sensor_params'][unique_cam]['height']
        img_w = meta['sensor_params'][unique_cam]['width']
        ixt = np.array(meta['sensor_params'][unique_cam]['camera_intrinsic'])
        camera_infos[unique_cam] = {
            'ixt': ixt,
            'img_h': img_h,
            'img_w': img_w,
        }

    # load id_names from database
    db = f'{colmap_dir}/database.db'
    conn = sqlite3.connect(db)
    c = conn.cursor()

    c.execute('SELECT * FROM images')
    result = c.fetchall()
    
    id_names = []
    for i in result:
        id_names.append([int(i[0]),i[1]])

    cam_to_id = dict()
    for i in result:
        name = i[1]
        cam = name.split('/')[0]
        cam_id = i[2]
        cam_to_id[cam] = cam_id

    model_dir = f'{colmap_dir}/created/sparse/model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    quaternions_Ts = {}
    with open(f'{colmap_dir}/sparse/origin/images.txt','r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    quaternions_Ts[parts[-1]] = [float(x) for x in parts[1:8]]

    # create images.txt
    with open(f'{model_dir}/images.txt','w') as f_w:
        for i in range(len(id_names)):
            id_ = id_names[i][0]
            name = id_names[i][1]

            cam = cam_to_id[name.split('/')[0]]

            f_w.write(f'{id_} ')
            f_w.write(' '.join([str(a) for a in quaternions_Ts[name]] ) )
            f_w.write(f' {cam} {name}')
            f_w.write('\n\n')

    # create cameras.txt
    cameras_fn = os.path.join(model_dir, 'cameras.txt')
    with open(cameras_fn, 'w') as f:
        for unique_cam in unique_cams:
            cam_id = cam_to_id[unique_cam]
            camera_info = camera_infos[unique_cam]
            ixt = camera_info['ixt']
            img_w = camera_info['img_w']
            img_h = camera_info['img_h']
            fx = ixt[0, 0]
            cx = ixt[0, 2]
            cy = ixt[1, 2]
            f.write(f'{cam_id} SIMPLE_PINHOLE {img_w} {img_h} {fx} {cx} {cy}')
            f.write('\n')

    # update database
    db = f'{colmap_dir}/database.db'
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('SELECT * FROM images')
    result = c.fetchall()
    cam_to_id = dict()
    for i in result:
        name = i[1]
        cam = name.split('/')[0]
        cam_id = i[2]
        cam_to_id[cam] = cam_id
    
    for __, unique_cam in enumerate(unique_cams):
        cam_id = cam_to_id[unique_cam]
        ixt = camera_infos[unique_cam]['ixt']
        fx, __, cx, cy = ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2]
        params = np.array([fx, cx, cy]).astype(np.float64)
        c.execute("UPDATE cameras SET params = ? WHERE camera_id = ?",
                (params.tostring(), cam_id))
    conn.commit()
    conn.close()

    # create points3D.txt
    points3D_fn = os.path.join(model_dir, 'points3D.txt')
    os.system(f'touch {points3D_fn}')
    
    # create rid ba config
    cam_rigid = dict()
    
    ref_camera_id = unique_cams[0]
    cam_rigid["ref_camera_id"] = ref_camera_id
    rigid_cam_list = []

    extrinsics = meta['sensor_params']
    for cam_id in unique_cams:
        rigid_cam = dict()
        rigid_cam["camera_id"] = cam_id

        ref_extrinsic = extrinsics[ref_camera_id]['extrinsic']
        cur_extrinsic = extrinsics[cam_id]['extrinsic']
        rel_extrinsic = np.linalg.inv(cur_extrinsic) @ ref_extrinsic
        print('relative extrinisc')
        print(cam_id, rel_extrinsic)
        r = R.from_matrix(rel_extrinsic[:3, :3])
        qvec = r.as_quat()
        rigid_cam["image_prefix"] = 'cam_{}'.format(cam_id)
        
        rigid_cam['cam_from_rig_rotation'] = [qvec[3], qvec[0], qvec[1], qvec[2]]
        rigid_cam['cam_from_rig_translation'] = [rel_extrinsic[0, 3], rel_extrinsic[1, 3], rel_extrinsic[2, 3]]
        
        rigid_cam_list.append(rigid_cam)

    cam_rigid["cameras"] = rigid_cam_list

    rigid_config_path = os.path.join(colmap_dir, "cam_rigid_config.json")
    with open(rigid_config_path, "w+") as f:
        json.dump([cam_rigid], f, indent=4)   
    
    os.system(f'colmap exhaustive_matcher \
        --database_path {colmap_dir}/database.db \
        --SiftMatching.use_gpu 0')
    os.system(f'colmap model_converter --input_path {model_dir} --output_path {model_dir} --output_type BIN')
    triangulated_dir = os.path.join(colmap_dir, 'sparse/0')
    os.makedirs(triangulated_dir, exist_ok=True)
    print('point triangulator')
    os.system(f'colmap point_triangulator \
            --database_path {colmap_dir}/database.db \
            --image_path {images_dir} \
            --input_path {model_dir} \
            --output_path {triangulated_dir} \
            --Mapper.ba_refine_focal_length 0 \
            --Mapper.ba_refine_principal_point 0 \
            --Mapper.max_extra_param 0 \
            --clear_points 0 \
            --Mapper.ba_global_max_num_iterations 30 \
            --Mapper.filter_max_reproj_error 4 \
            --Mapper.filter_min_tri_angle 0.5 \
            --Mapper.tri_min_angle 0.5 \
            --Mapper.tri_ignore_two_view_tracks 1 \
            --Mapper.tri_complete_max_reproj_error 4 \
            --Mapper.tri_continue_max_angle_error 4')
    os.system(f'colmap model_converter --input_path {triangulated_dir} --output_path {triangulated_dir} --output_type TXT')

if __name__ == '__main__':
    import argparse    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="")
    parser.add_argument("--meta_file", default="transform.json")

    args = parser.parse_args()
    run_colmap_waymo(args)