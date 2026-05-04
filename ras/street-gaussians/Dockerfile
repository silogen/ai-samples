# Copyright 2026 Advanced Micro Devices, Inc.  All rights reserved.
 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
 
#       http://www.apache.org/licenses/LICENSE-2.0
 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Main Dockerfile for training and rendering
# Note that there is a separate Dockerfile for data pre-processing, which has different requirements
FROM rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.5.1

RUN apt-get update && apt-get install -y \
    git \
    python3-dev \
    build-essential \
    cmake \
    libjpeg-dev \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# For rendering, we also need ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app

# Get AMD-enabled street gaussian repository
COPY . /app/street-gaussians-ns


RUN pip install nerfstudio
RUN pip install wheel cython
# Pin numpy to 1.x, since nerfstudio is incompatible with numpy 2 (even though it's supposed to work)
# This only comes up when you try using the exporter, which loads skimage
RUN pip install --upgrade numpy==1.26.4
RUN pip install -e /app/street-gaussians-ns

# Install gsplat from the AMD repo
# It's important to do this after nerfstudio, so that we don't accidentally upgrade gsplat to the main pypi latest version
RUN pip install --no-deps --force-reinstall gsplat --index-url=https://pypi.amd.com/simple

WORKDIR /app/street-gaussians-ns
