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

MODEL=archesweathergen-s-ft

# run on cpu only (xarray won't save tensors otherwise) 
export HIP_VISIBLE_DEVICES=""

python -m geoarches.evaluation.eval_multistep  \
    --pred_path evalstore/${MODEL}/ \
    --output_dir evalstore/${MODEL}_metrics/ \
    --groundtruth_path data/era5_240/full/ \
    --multistep 10 --num_workers 4 \
    --metrics era5_rank_histogram_25_members era5_ensemble_metrics era5_power_spectrum era5_power_spectrum_with_ref era5_brier_skill_score hres_brier_skill_score \
    --pred_filename_filter "members=25-"