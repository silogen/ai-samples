
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
MODEL_NAME=aurora
mkdir -p predictions assets logs
ai-models --download-assets --assets "assets/$MODEL_NAME" --input=cds --date=20240101 --time=0000 --lead-time=24 --path="predictions/$MODEL_NAME.grib" $MODEL_NAME  > logs/$MODEL_NAME.log 2>&1 &

python3 grib_visualizer.py --input predictions/$MODEL_NAME.grib

