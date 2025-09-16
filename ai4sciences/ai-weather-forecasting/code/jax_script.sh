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


MODEL_NAME=panguweather
mkdir -p predictions assets logs
LOGFILE="logs/${MODEL_NAME}.log"
cp "$0" "logs/$(basename "$0" .sh).sh"
echo "starting at `date`" > "${LOGFILE}"
{
ai-models --download-assets --assets "assets/$MODEL_NAME" --input=cds --date=20240101 --time=0000 --lead-time=240 --path="predictions/$MODEL_NAME.grib" $MODEL_NAME 

echo "ending model at `date`"  
python3 grib_visualizer.py --input predictions/$MODEL_NAME.grib
  # Print done
  echo "Done!"
  echo "Log saved to: $LOG_FILE"
} 2>&1 | tee -a "${LOGFILE}"

source set_XLA_params.sh
MODEL_NAME=gencast-0.25
{mkdir -p predictions assets logs
ai-models --download-assets --assets "assets/$MODEL_NAME" --input=cds --date=20240101 --time=0000 --lead-time=240 --path="predictions/$MODEL_NAME.grib"  $MODEL_NAME
python3 grib_visualizer.py --input predictions/$MODEL_NAME.grib

echo "ending model at `date`"  
python3 grib_visualizer.py --input predictions/$MODEL_NAME.grib
  # Print done
  echo "Done!"
  echo "Log saved to: $LOG_FILE"
} 2>&1 | tee -a "${LOGFILE}"

