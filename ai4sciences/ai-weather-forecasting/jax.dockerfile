### Copyright 2025 Advanced Micro Devices, Inc.  All rights reserved.
### Licensed under the Apache License, Version 2.0 (the "License");
### you may not use this file except in compliance with the License.
### You may obtain a copy of the License at
###      http://www.apache.org/licenses/LICENSE-2.0
### Unless required by applicable law or agreed to in writing, software
### distributed under the License is distributed on an "AS IS" BASIS,
### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
### See the License for the specific language governing permissions and
### limitations under the License.

FROM rocm/jax-community:latest
#### FROM rocm/jax-community@sha256:8bab484be1713655f74da51a191ed824bb9d03db1104fd63530a1ac3c37cf7b1
WORKDIR /workspace

COPY code/set_XLA_params.sh .
COPY code/output_processors/grib_visualizer.py .
COPY code/jax_script.sh .

RUN pip install git+https://github.com/deepmind/graphcast.git
#### RUN pip install git+https://github.com/deepmind/graphcast.git@5e63ba030f6438d0a97521b30360b7c95d28796a


RUN git clone https://github.com/ecmwf-lab/ai-models.git
#### RUN cd ai-models && git reset --hard fabc763 && cd ..
COPY code/rocm.patch ai-models/
RUN cd ai-models && git apply rocm.patch && cd ..
RUN pip install -e ai-models


# install gencast model
RUN pip install git+https://github.com/ecmwf-lab/ai-models-gencast.git
#### RUN pip install git+https://github.com/ecmwf-lab/ai-models-gencast.git@dc5e13b3fc6cb748d7a1c637eaa6ba33baf02882

RUN pip install dm-haiku==0.0.13

# panguweather runs with onnx runtime, it installs cuda version by default, need to reinstall
RUN pip install git+https://github.com/ecmwf-lab/ai-models-panguweather.git
#### RUN pip install git+https://github.com/ecmwf-lab/ai-models-panguweather.git@06b79a4f6a37aba19f9920dde257b157f8419805
RUN pip uninstall -y onnxruntime-gpu
RUN pip3 install onnxruntime-rocm -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.4/
RUN pip install --upgrade ml_dtypes

