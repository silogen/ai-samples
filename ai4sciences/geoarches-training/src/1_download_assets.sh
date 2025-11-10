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

src="https://huggingface.co/gcouairon/ArchesWeather/resolve/main"

mkdir -p geoarches/stats
wget -q --show-progress -O geoarches/stats/era5-quantiles-2016_2022.nc "$src/era5-quantiles-2016_2022.nc"

## Uncomment the following lines if you want to download pretrained model checkpoints and configs
# MODELS=("archesweather-m-seed0" "archesweather-m-seed1" "archesweather-m-skip-seed0" "archesweather-m-skip-seed1" "archesweathergen")
# for MOD in "${MODELS[@]}"; do
#     mkdir -p "modelstore/$MOD/checkpoints"
#     wget -q --show-progress -O "modelstore/$MOD/checkpoints/checkpoint.ckpt" "$src/${MOD}_checkpoint.ckpt"
#     wget -q --show-progress -O "modelstore/$MOD/config.yaml" "$src/${MOD}_config.yaml"
# done

# Download training data
echo "Downloading ERA5 data"
python geoarches/download/dl_era.py

echo "Done"