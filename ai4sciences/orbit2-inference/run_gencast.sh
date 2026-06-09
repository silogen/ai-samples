#!/usr/bin/env bash

# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run ORBIT-2 precipitation downscaling on GenCast GRIB output
# set datapath and outpath
datapath=
outpath_highres=
outpath_lowres=
export PYTHONPATH=/workspace/ORBIT-2/examples:${PYTHONPATH}

python grib_to_npz.py \
      --grib "$datapath/gencast-1.0.grib" \
      --valid-time 2020-01-02 \
      --date 2020-01-01 \
      --out "$outpath_lowres/gencast-1.0_step1.npz"

python grib_to_npz.py \
      --grib "$datapath/gencast-0.25.grib" \
      --valid-time 2020-01-02 \
      --date 2020-01-01 \
      --out "$outpath_highres/gencast-0.25_step1.npz"

bash run_infer.sh
