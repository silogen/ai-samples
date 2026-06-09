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

FROM rocm/pytorch:rocm6.4_ubuntu22.04_py3.11_pytorch_release_2.5.1

RUN apt-get update && \
    apt-get install -y libmpich-dev libopenmpi-dev \
        libgeos-dev libproj-dev proj-data proj-bin && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/ORBIT-2

RUN pip install --no-cache-dir \
    torch==2.8.0+rocm6.4 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/rocm6.4 && \
    pip install --no-cache-dir -U \
    xformers==0.0.32.post2 \
    --index-url https://download.pytorch.org/whl/rocm6.4 && \
    pip install --no-cache-dir \
    cfgrib gcsfs zarr==3.1.6 scipy==1.17.1 numpy==2.4.3 scikit-image==0.26.0 mpi4py huggingface_hub \
    cartopy plotly cmocean

COPY ORBIT-2/ /workspace/ORBIT-2/
RUN pip install --no-cache-dir -e .
