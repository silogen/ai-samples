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

multistep=10

python -m geoarches.main_hydra ++mode=test ++name=archesweathergen-s-ft \
    ++limit_test_batches=0.1 \
    ++dataloader.test_args.multistep=$multistep \
    ++module.inference.save_test_outputs=True \
    ++module.inference.rollout_iterations=$multistep \
    ++module.inference.num_steps=25 \
    ++module.inference.num_members=25 \
    ++module.inference.scale_input_noise=1.05 \
    ++dataloader.dataset.path=data/era5_240/full/