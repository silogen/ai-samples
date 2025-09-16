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

FROM rocm/pytorch:latest
#### FROM rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0
WORKDIR /workspace

COPY code/output_processors/grib_visualizer.py .
COPY code/torch_script.sh .
COPY code/multiurl.patch .

RUN pip install xarray=="2024.11.0" # This is needed due to the format in which the data is writtten
RUN pip install cfgrib
RUN git clone https://github.com/ecmwf-lab/ai-models.git

#### RUN cd ai-models && git reset --hard fabc763 && cd ..

RUN git clone https://github.com/ecmwf/multiurl.git

#### RUN cd multiurl && git reset --hard acdbb59009f12faec1e4c6db04932d417aced7dc && cd ..

RUN git clone https://github.com/ecmwf-lab/ai-models-aurora.git

#### RUN cd ai-models-aurora && git reset --hard 9f5022231d03cd92d0f2862caeb3c21e3629cca1 && cd ..
RUN git clone https://github.com/microsoft/aurora.git

#### RUN cd aurora && git reset --hard c7d67843a57bccfee5c809176195a83f790d3f0f && cd ..
RUN cd multiurl && git apply ../multiurl.patch && cd .. # fix tqdm progress bar problem 
RUN pip install -e multiurl
RUN pip install microsoft-aurora && pip install -e ai-models
RUN pip install -e ai-models-aurora
RUN pip install matplotlib && pip install cartopy
#### RUN pip install matplotlib==3.10.5 && pip install cartopy==0.25.0

