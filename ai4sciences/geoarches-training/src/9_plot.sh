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

python -m geoarches.evaluation.plot --output_dir plots/rankhist/ \
    --metric_paths evalstore/${MODEL}_metrics/test-multistep=10-era5_rank_histogram_25_members.nc \
    --model_names ArchesWeatherGen \
    --model_colors red \
    --metrics rankhist \
    --vars Z500:geopotential:level:500 T850:temperature:level:850 Q700:specific_humidity:level:700 U850:u_component_of_wind:level:850 V850:v_component_of_wind:level:850 \
    --rankhist_prediction_timedeltas 1 10 \
    --figsize 15 8

python -m geoarches.evaluation.plot --output_dir plots/ensemble/ \
    --metric_paths evalstore/${MODEL}_metrics/test-multistep=10-era5_ensemble_metrics.nc \
    --model_names ArchesWeatherGen \
    --model_colors red \
    --metrics rmse crps fcrps spskr \
    --vars Z500:Z500 T850:T850 Q700:Q700 U850:U850 V850:V850 \
    --figsize 15 8

python -m geoarches.evaluation.plot --output_dir plots/brierskill/ \
    --metric_paths evalstore/${MODEL}_metrics/test-multistep=10-hres_brier_skill_score.nc \
    --model_names ArchesWeatherGen \
    --model_colors red \
    --metrics brierskillscore \
    --vars Z500:Z500 T850:T850 Q700:Q700 U850:U850 V850:V850 \
    --figsize 15 8