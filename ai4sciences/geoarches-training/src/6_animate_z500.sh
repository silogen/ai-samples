#!/bin/bash
# Copyright 2025 Advanced Micro Devices, Inc.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

python z500_vs_gt.py \
    --model archesweathergen-s-ft \
    --data-path data/era5_240/full/ \
    --output-dir ./gifs/ \
    --rollout-iterations 10 \
    --n-members 25 \
    --cmap viridis