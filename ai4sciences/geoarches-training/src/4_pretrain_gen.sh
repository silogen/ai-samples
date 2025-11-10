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

export HYDRA_FULL_ERROR=1

M4ARGS="++dataloader.dataset.pred_path=data/outputs/deterministic/archesweather-m/ \
++module.module.load_deterministic_model=[archesweather-m-seed0-ft,archesweather-m-seed1-ft,archesweather-m-seed2-ft,archesweather-m-seed3-ft]"

python -m geoarches.main_hydra ++log=True \
    module=archesweathergen \
    dataloader=era5pred \
    ++limit_val_batches=10 \
    ++max_steps=200000 \
    ++name=archesweathergen-s \
    $M4ARGS \
    ++seed=0 \
    ++save_step_frequency=50000 \
    ++batch_size=4 \
    ++cluster.cpus=4 \
    ++module.module.weight_decay=0.05 \
    ++dataloader.dataset.path=data/era5_240/full/