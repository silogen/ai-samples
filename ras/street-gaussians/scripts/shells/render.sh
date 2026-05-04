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

config_path=$1
model_names=${2:-full_scene}
edit_config_path=$3

if [ -n "$edit_config_path" ]; then
    sgn-render --load-config "$config_path" --model_names "$model_names" --edits "$edit_config_path"
else
    sgn-render --load-config "$config_path" --model_names "$model_names"
fi

