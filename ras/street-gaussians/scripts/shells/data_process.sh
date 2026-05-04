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

sh scripts/shells/segs_generate.sh "$root"

sh scripts/shells/masks_generate.sh "$root"

# sh scripts/shells/run_colmap.sh $root
python scripts/pythons/run_colmap.py --root "$root"

# in case colmap does not work (e.g., 10448102132863604198_472_000_492_000), we just ues the LiDAR data for initialization
if [ -d "$root/colmap/sparse/0" ] && [ -z "$(ls -A "$root/colmap/sparse/0")" ]; then
    # just use the origin model, which has a empty 3Dpoints.txt
    echo "Colmap failed, using LiDAR data for initialization"
    colmap model_converter --input_path "$root"/colmap/created/sparse/model --output_path "$root"/colmap/sparse/0 --output_type BIN
fi

sh scripts/shells/points_cloud_generate.sh "$root"

sh scripts/shells/object_pts_generate.sh "$root"