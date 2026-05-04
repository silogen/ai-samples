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

import collections
import json
import os
import numpy as np

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

        
if __name__ == "__main__":
    import argparse    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="")
    parser.add_argument("--json_file", default="transform.json")
    parser.add_argument("--lower_timestamp", type=float, default=-1)
    parser.add_argument("--higher_timestamp", type=float, default=-1)
    parser.add_argument("--subtract_first_frame", type=bool, default=True)

    args = parser.parse_args()

    sparse_num = 1
    lower_timestamp = args.lower_timestamp
    higher_timestamp = args.higher_timestamp
    input_path = args.input_path
    input_json = input_path + "/" + args.json_file
    point3D_txt = input_path + "/colmap/sparse/origin/points3D.txt"
    cameras_txt = input_path + "/colmap/sparse/origin/cameras.txt"
    images_txt = input_path + "/colmap/sparse/origin/images.txt"
    folder_path = input_path + "/colmap/sparse/origin"
    result_path = input_path + "/colmap/sparse/0"
    
    if not os.path.exists(result_path):  
        os.makedirs(result_path)
        
    if not os.path.exists(folder_path):  
        os.makedirs(folder_path)
    if os.path.exists(point3D_txt):  
        print("文件已存在")  
    else:  
        with open(point3D_txt, 'w') as file:  
            pass
        print("文件已创建")
    with open(input_json, 'r') as file:
        data = json.load(file)

    frames = data['frames']

    cams = list(collections.OrderedDict.fromkeys(frame["camera"] for frame in frames))
    cams_map = {cam: i + 1 for i, cam in enumerate(cams)}

    new_frames = frames[::sparse_num]
    new_data=[]
    for obj in new_frames:  
        if 'camera' in obj:
            if obj["camera"] in cams: 
                if 'timestamp' in obj and lower_timestamp > 0 and higher_timestamp > 0:
                    if float(obj['timestamp']) >= lower_timestamp and float(obj['timestamp']) <= higher_timestamp:  
                        new_data.append(obj)  
                else:
                    new_data.append(obj) 
        else:
            new_data.append(obj) 

    cameras_content={}
    images_content={}
    id= 0
    c2w0 = np.array(new_data[0]["transform_matrix"])

    T0=(c2w0[:3,3])*0.98
    if not args.subtract_first_frame:
        T0=np.array([0,0,0])
    for obj in new_data:  
        id +=1
        c2w = np.array(obj["transform_matrix"])
        #T0=np.array([random.random()*0.05,random.random()*0.05,random.random()*0.05])
        c2w[:3,3]-=T0

        c2w[2, :] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[0:3, 1:3] *= -1

        c2w = np.linalg.inv(c2w)

        quaternion = rotmat2qvec(c2w[:3,:3])

        cam_index = cams_map[obj["camera"]]
        T=(c2w[:3,3]).tolist()
        filename = obj["file_path"]
        
        slash_index = filename.find('/')

        if slash_index != -1:
            result_string = filename[slash_index + 1:]
        else:
            result_string = filename
        
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

        image_paras = [quaternion[0],quaternion[1],quaternion[2],quaternion[3],T[0],T[1],T[2],cam_index,result_string]
        image_paras = [str(i) for i in image_paras]  
        
        images_content[id] = image_paras
        
        camera_model =obj["camera_model"]
        
        w= obj["w"]
        h= obj["h"]
        fx= obj["fl_x"]
        fy= obj["fl_y"]
        cx = obj["cx"]
        cy = obj["cy"]
        k1 = obj["k1"]
        k2 = obj["k2"]

 
        if(cam_index in cameras_content):
            continue
        else:
            paras=[]
            if(camera_model == "OPENCV_FISHEYE"):
                k3 = obj["k3"]
                k4 = obj["k4"]
                paras=[camera_model,w,h,fx,fy,cx,cy,k1,k2,k3,k4]
            elif(camera_model == "OPENCV"):
                p1 = obj["p1"]
                p2 = obj["p2"]   
                paras=[camera_model,w,h,fx,fy,cx,cy,k1,k2,p1,p2]
            else:
                paras=[camera_model,w,h,fx,fy,cx,cy]
            paras =[str(i) for i in paras]  
            cameras_content[cam_index] = paras


    with open(cameras_txt, 'w') as f:  
        for cam_index in cameras_content:
            f.write(str(cam_index)+" ")
            f.write(' '.join(cameras_content[cam_index]) + '\n')
            
    with open(images_txt, 'w') as f:  
        for image_index in images_content:
            f.write(str(image_index)+" ")
            f.write(' '.join(images_content[image_index]) + '\n\n')