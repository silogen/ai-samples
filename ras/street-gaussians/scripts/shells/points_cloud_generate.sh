#!/bin/bash
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

root=$1

mkdir "$root"/colmap/sparse/lidar

python scripts/pythons/pcd2colmap_points3D.py \
    --root_path "$root" \
    --main_lidar_in_transforms lidar_FRONT \

python scripts/pythons/colmap_pts_combine.py \
    --src1 "$root"/colmap/sparse/lidar/points3D.txt \
    --src2 "$root"/colmap/sparse/0/points3D.bin \
    --dst "$root"/colmap/sparse/0/points3D_withlidar.bin