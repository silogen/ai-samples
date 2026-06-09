# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluate metrics from metrics.py against saved ORBIT-2 predictions.

Loads predictions from --pred-dir and ground truth from OLCF_DATA_PATH,
evaluates each metric in metrics.METRICS per timestep, and writes a
summary table to <out-dir>/metrics.txt.
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import argparse
import os
import datetime
import numpy as np
from metrics import METRICS

parser = argparse.ArgumentParser()
parser.add_argument("--pred-dir", default="./results",
                    help="Directory containing *_preds.npy files (default: ./results)")
parser.add_argument("--out-dir", default=None,
                    help="Output directory for metrics.txt (default: same as pred-dir)")
parser.add_argument("--no-flip", action="store_true",
                    help="Do not flip prediction along N-S axis (default is to flip)")
parser.add_argument("--metrics", nargs="+", default=None,
                    help="Subset of metrics to evaluate (default: all). "
                         f"Available: {list(METRICS)}")
args = parser.parse_args()

OLCF_data_path = os.environ.get("OLCF_DATA_PATH", "/workspace/Data")
high_res_dir   = os.path.join(OLCF_data_path, "ERA5_IMERG_output/test")
pred_dir       = args.pred_dir
out_dir        = args.out_dir or pred_dir
npz_file       = "2021_0.npz"
variable       = "total_precipitation_24hr"
n_timesteps    = 5

metric_names = args.metrics if args.metrics else list(METRICS)
unknown = [m for m in metric_names if m not in METRICS]
if unknown:
    raise ValueError(f"Unknown metrics: {unknown}. Available: {list(METRICS)}")

_hr_npz    = np.load(os.path.join(high_res_dir, npz_file))
hr_all     = _hr_npz[variable][:, 0, :, :]
_year      = int(os.path.splitext(npz_file)[0].split("_")[0])
_doys      = _hr_npz["days_of_year"][:, 0, 0, 0].astype(int)
timestamps = [
    (datetime.date(_year, 1, 1) + datetime.timedelta(days=int(d) - 1)).strftime("%Y-%m-%d")
    for d in _doys
]

stem = os.path.splitext(npz_file)[0]

rows = []
for i in range(n_timesteps):
    hr        = hr_all[i]
    pred_path = os.path.join(pred_dir, f"{stem}_{i}_preds.npy")
    if not os.path.exists(pred_path):
        pred_path = os.path.join(pred_dir, f"{i}_preds.npy")
    pred = np.expm1(np.load(pred_path)) / 1000.0
    if not args.no_flip:
        pred = pred[::-1, :]

    row = {"t": i, "date": timestamps[i]}
    print(f"\nTimestep {i}  ({timestamps[i]})")
    for name in metric_names:
        val = METRICS[name](pred, hr)
        row[name] = val
        print(f"  {name:25s} {val:.6f}")
    rows.append(row)

# Mean across timesteps
print("\n--- Mean across all timesteps ---")
means = {}
for name in metric_names:
    means[name] = float(np.mean([r[name] for r in rows]))
    print(f"  {name:25s} {means[name]:.6f}")

# Write table to file
col_w  = 14
header = f"{'t':>3}  {'date':>12}  " + "  ".join(f"{n:>{col_w}}" for n in metric_names)
sep    = "-" * len(header)
lines  = [header, sep]
for r in rows:
    vals = "  ".join(f"{r[n]:>{col_w}.6f}" for n in metric_names)
    lines.append(f"{r['t']:>3}  {r['date']:>12}  {vals}")
lines.append(sep)
mean_vals = "  ".join(f"{means[n]:>{col_w}.6f}" for n in metric_names)
lines.append(f"{'mean':>3}  {'':>12}  {mean_vals}")

table = "\n".join(lines)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "metrics.txt")
with open(out_path, "w") as f:
    f.write(table + "\n")
print(f"\nSaved {out_path}")
