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

# Configs can be found here:
# model hyperparameters: geoarches/geoarches/configs/module/archesweather.yaml
# cluster parameters: geoarches/geoarches/configs/cluster/local.yaml
# dataloader params: geoarches/geoarches/configs/dataloader/era5.yaml
# You can modify these configs or override with ++ in command below

export HYDRA_FULL_ERROR=1

echo "Pretraining deterministic models..."

for i in {0..3}; do
    python -m geoarches.main_hydra ++log=True \
        dataloader=era5 \
        module=archesweather \
        ++name=archesweather-m-seed$i \
        ++cluster.precision=32-true \
        ++cluster.cpus=4 \
        ++batch_size=4 \
        ++max_steps=250000 \
        ++save_step_frequency=50000 \
        ++dataloader.dataset.path=data/era5_240/full/
done

echo "Done"