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

data_root=$1
echo "Training on $data_root"

sgn-train street-gaussians-ns \
    --save-only-latest-checkpoint True \
    --experiment_name street-gaussians \
    --output_dir output/ \
    --vis tensorboard \
    --pipeline.datamanager.masks-on-gpu True \
    --pipeline.datamanager.images-on-gpu True \
    --pipeline.datamanager.cache-images gpu \
    colmap-data-parser-config \
    --data "$data_root"
