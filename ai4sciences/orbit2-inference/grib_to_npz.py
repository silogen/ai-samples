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
Convert a GenCast GRIB file to the ORBIT-2 NPZ input format.

Output matches ERA5_IMERG_input/test/2021_0.npz exactly:
  - All spatial variables: float32, shape (T, 1, H, W), S→N latitude
  - days_of_year / time_of_day: int32,  shape (T, 1, W, H)  ← transposed
  - latitude / lattitude:       float64, shape (T, 1, H, W)
  - Scalar metadata:            int64 scalars

Variables sourced from the GRIB file:
  total_precipitation_24hr, 2m_temperature, 2m_temperature_max/min,
  10m_u/v_component_of_wind, temperature/u/v/specific_humidity/geopotential
  at 200/500/850 hPa, land_sea_mask, orography, sea_surface_temperature

Variables downloaded from the Copernicus CDS API (ERA5 reanalysis):
  volumetric_soil_water_layer_1, landcover (soil_type)

CDS credentials:
  Create ~/.cdsapirc with:
    url: https://cds.climate.copernicus.eu/api
    key: <your-personal-access-token>
  Or set env vars CDSAPI_URL and CDSAPI_KEY.

Usage:
  # Date is read automatically from the GRIB file:
  python grib_to_npz.py --grib results/gencast-1.0.grib --out results/gencast_input.npz

  # Override date explicitly if needed:
  python grib_to_npz.py --grib results/gencast-1.0.grib \\
      --date 2021-01-01 --n-steps 5 --out results/gencast_input.npz

  # Skip CDS download (fills missing vars with zeros):
  python grib_to_npz.py --grib results/gencast-1.0.grib --no-cds --out results/gencast_input.npz
"""

import argparse
import datetime
import os
import tempfile
import warnings

import cfgrib
import numpy as np

warnings.filterwarnings("ignore")

# Standard gravity used to convert geopotential (m²/s²) → geometric height (m).
# Value matches ORBIT-2 training pipeline (2026-05-19):
# https://github.com/XiaoWang-Github/ORBIT-2/blob/main/src/climate_learn/data/processing/climatebench.py
# Line 82
GNEWTON = 9.807

# ── Pressure level selection ──────────────────────────────────────────────────
TARGET_PLEVS = [200, 500, 850]   # hPa


def _select_step(da, target_vt: np.datetime64):
    """Select a single valid-time step from a DataArray, returning (plev,H,W) or (H,W).

    Handles:
    - 2-D static (H,W)           — returned unchanged
    - 3-D (T,H,W)                — select matching time index
    - 4-D (T,plev,H,W)           — select matching time index
    - 4-D (member,T,H,W) tp      — select member whose valid_time 2-D array
                                    contains the target date, then select step
    """
    if da is None:
        return da

    target_day = target_vt.astype("datetime64[D]")

    # ── Try valid_time coordinate first ──────────────────────────────────
    if "valid_time" in da.coords:
        vt = da.coords["valid_time"].values
        if vt.ndim == 0:                        # scalar — already the right step
            return da
        if vt.ndim == 1:                        # (T,) — simple time series
            matches = np.where(vt.astype("datetime64[D]") == target_day)[0]
            if len(matches):
                step_dim = da.dims[0]
                return da.isel({step_dim: int(matches[0])})
        if vt.ndim == 2:                        # (member, T) — ensemble tp
            # Find (member, step) where valid_time matches target day at 00Z
            target_ns = target_vt.astype("datetime64[ns]")
            rows, cols = np.where(vt == target_ns)
            if len(rows):
                m, s = int(rows[0]), int(cols[0])
                member_dim, step_dim = da.dims[0], da.dims[1]
                return da.isel({member_dim: m, step_dim: s})
            # Fallback: match by day only
            rows, cols = np.where(vt.astype("datetime64[D]") == target_day)
            if len(rows):
                m, s = int(rows[0]), int(cols[0])
                member_dim, step_dim = da.dims[0], da.dims[1]
                return da.isel({member_dim: m, step_dim: s})

    # ── Fall back to step coordinate + time reference ─────────────────────
    if "step" in da.coords and "time" in da.coords:
        time_ref = da.coords["time"].values
        steps    = da.coords["step"].values
        # Compute valid times as time_ref + step
        if np.ndim(time_ref) == 0:
            vts = (time_ref + steps).astype("datetime64[D]")
            matches = np.where(vts == target_day)[0]
            if len(matches):
                step_dim = da.dims[0]
                return da.isel({step_dim: int(matches[0])})

    raise ValueError(
        f"Could not find valid_time {target_vt} in DataArray with "
        f"coords {list(da.coords)}"
    )


def _find_var(datasets: list, name: str):
    """Return the first DataArray named `name` found across all datasets."""
    for ds in datasets:
        if name in ds.data_vars:
            return ds[name]
    return None


def _to_sn(arr: np.ndarray) -> np.ndarray:
    """Flip lat axis from N→S (GRIB) to S→N (NPZ convention).

    Also drops the North Pole row (+90°) if present: GRIB grids include both
    poles (181 or 721 pts) while the NPZ uses 180/720 pts (-90 to 89/89.75°).
    """
    arr = arr[..., ::-1, :]          # N→S to S→N
    # Drop last row (now +90°) if grid has an odd pole-inclusive count
    h = arr.shape[-2]
    if h in (181, 721):
        arr = arr[..., :-1, :]       # 181→180, 721→720
    return arr


def _ensure_3d(arr: np.ndarray, T: int) -> np.ndarray:
    """Return (T, H, W) array. Broadcasts 2-D static fields across T."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:                          # static (H, W)
        arr = np.broadcast_to(arr[np.newaxis], (T, *arr.shape)).copy()
    elif arr.ndim == 4:                        # (member, T, H, W)
        arr = arr[0]
    return arr.astype(np.float32)


def _select_plev(da, plev: int) -> np.ndarray:
    """Extract a single pressure level from a 3-D or 4-D DataArray.

    Uses xarray coordinate selection where available; falls back to the
    integer indices from visualize_orbit2.py (850→2, 500→5, 200→9).
    """
    fallback = {850: 2, 500: 5, 200: 9}
    if da is None:
        raise ValueError(f"Pressure-level variable not found in GRIB")
    arr = np.asarray(da, dtype=np.float32)

    # Try coordinate-based selection via xarray
    for coord in da.coords:
        if "pressure" in coord.lower() or "isobaric" in coord.lower():
            try:
                return np.asarray(da.sel({coord: plev}), dtype=np.float32)
            except Exception:
                pass

    # Fallback: integer index
    idx = fallback[plev]
    if arr.ndim == 3:           # (plev, H, W)
        return arr[idx]
    elif arr.ndim == 4:         # (T, plev, H, W)
        return arr[:, idx, :, :]
    raise ValueError(f"Cannot select {plev} hPa from array shape {arr.shape}")


# ── CDS download ──────────────────────────────────────────────────────────────

_CDS_SINGLE_VARS = {
    "volumetric_soil_water_layer_1": "volumetric_soil_water_layer_1",
    "landcover":                     "soil_type",
}

_CDS_STATIC_VARS = {
    "land_sea_mask": "land_sea_mask",
    "orography":     "geopotential",       # orography = z/g at surface
}


def _download_cds(date_str: str, n_steps: int, variables: list,
                  static: bool = False) -> dict:
    """Download ERA5 variables from CDS for `n_steps` daily timesteps.

    Returns dict mapping variable_name → (T, H, W) float32 arrays (N→S).
    """
    try:
        import cdsapi
        import xarray as xr
    except ImportError as e:
        print(f"  WARNING: {e} — skipping CDS download for {variables}")
        return {}

    start = datetime.date.fromisoformat(date_str)
    days  = [(start + datetime.timedelta(days=i)) for i in range(n_steps)]

    years  = sorted({d.strftime("%Y") for d in days})
    months = sorted({d.strftime("%m") for d in days})
    ddays  = sorted({d.strftime("%d") for d in days})

    c = cdsapi.Client(quiet=True)
    result = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        nc_path = os.path.join(tmpdir, "era5.nc")
        try:
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable":     variables,
                    "year":  years,
                    "month": months,
                    "day":   ddays,
                    "time":  ["00:00"],
                    "format": "netcdf",
                },
                nc_path,
            )
        except Exception as ex:
            print(f"  CDS download failed: {ex}")
            return {}

        ds = xr.open_dataset(nc_path)
        print(f"  CDS NetCDF variables: {list(ds.data_vars)}")
        # CDS returns GRIB short names; build flexible lookup
        # e.g. volumetric_soil_water_layer_1 → swvl1, soil_type → slt
        _SHORT = {
            "volumetric_soil_water_layer_1": ["swvl1", "volumetric_soil_water_layer_1"],
            "soil_type":                     ["slt",   "soil_type", "soiltype"],
            "land_sea_mask":                 ["lsm",   "land_sea_mask"],
            "geopotential":                  ["z",     "geopotential"],
        }
        for var in variables:
            candidates = _SHORT.get(var, [var]) + [var]
            match = next((k for k in ds.data_vars
                          if any(c.lower() == k.lower() for c in candidates)
                          or var.lower() in k.lower()), None)
            if match is None:
                print(f"  WARNING: {var} not found in CDS response "
                      f"(available: {list(ds.data_vars)})")
                continue
            arr = ds[match].values.astype(np.float32)   # (T, lat, lon)
            if static and arr.ndim == 3 and arr.shape[0] > 1:
                arr = arr[:1]
            result[var] = arr

    return result


def download_missing_from_cds(date_str: str, n_steps: int,
                               need_static: bool, no_cds: bool) -> dict:
    """Coordinate all CDS downloads. Returns dict of (T,H,W) arrays (N→S)."""
    if no_cds:
        return {}

    print("\nDownloading missing variables from CDS ERA5…")
    out = {}

    # Time-varying single-level fields
    tv_vars = list(_CDS_SINGLE_VARS.values())
    tv_data = _download_cds(date_str, n_steps, tv_vars, static=False)
    for npz_key, cds_var in _CDS_SINGLE_VARS.items():
        if cds_var in tv_data:
            out[npz_key] = tv_data[cds_var]

    # Static fields (only if GRIB is missing them)
    if need_static:
        st_vars = list(_CDS_STATIC_VARS.values())
        st_data = _download_cds(date_str, 1, st_vars, static=True)
        for npz_key, cds_var in _CDS_STATIC_VARS.items():
            if cds_var in st_data:
                arr = st_data[cds_var]
                if npz_key == "orography":
                    arr = arr / GNEWTON      # geopotential → geopotential height (m)
                out[npz_key] = arr

    print(f"  Downloaded: {list(out)}")
    return out


def _dates_from_grib(datasets: list) -> list:
    """Extract sorted list of datetime.date objects from GRIB time coordinates."""
    dates = set()
    for ds in datasets:
        for coord in ("valid_time", "time"):
            if coord in ds.coords:
                vals = np.asarray(ds.coords[coord]).ravel()
                for v in vals:
                    # cfgrib returns numpy datetime64 or cftime objects
                    try:
                        ts = np.datetime64(v, "D")
                        d = datetime.date.fromisoformat(str(ts))
                        dates.add(d)
                    except Exception:
                        pass
    return sorted(dates)


# ── Main converter ────────────────────────────────────────────────────────────

def convert(grib_path: str, date_str: str | None, n_steps: int,
            out_path: str, no_cds: bool, cds_date_str: str | None = None,
            valid_time_str: str | None = None):

    print(f"\nLoading GRIB: {grib_path}")
    datasets = cfgrib.open_datasets(grib_path)
    print(f"  {len(datasets)} dataset(s) found:")
    for i, ds in enumerate(datasets):
        print(f"  [{i}] {list(ds.data_vars)}")

    # ── Determine output grid from GRIB ──────────────────────────────────
    ref_da = _find_var(datasets, "tp")
    if ref_da is None:
        ref_da = _find_var(datasets, "t2m")
    if ref_da is None:
        raise RuntimeError("Cannot find tp or t2m in GRIB to determine grid size")
    ref_arr = np.asarray(ref_da)
    if ref_arr.ndim == 2:
        H, W = ref_arr.shape
        T_grib = 1
    elif ref_arr.ndim == 3:
        T_grib, H, W = ref_arr.shape
    else:
        T_grib, H, W = ref_arr.shape[0], ref_arr.shape[-2], ref_arr.shape[-1]
    T = min(T_grib, n_steps)
    # GRIB grids include both poles; NPZ excludes +90°
    H_out = H - 1 if H in (181, 721) else H
    print(f"  Grid: {H}×{W} (GRIB) → {H_out}×{W} (NPZ), using {T} of {T_grib} timestep(s)")

    # ── Resolve GRIB start date (for metadata: days_of_year) ─────────────
    grib_dates = _dates_from_grib(datasets)
    if grib_dates:
        start = grib_dates[0]
        print(f"  GRIB start date: {start}  (all: {[str(d) for d in grib_dates]})")
    elif date_str:
        start = datetime.date.fromisoformat(date_str)
        print(f"  GRIB start date (from --date fallback): {start}")
    else:
        raise RuntimeError("Could not extract dates from GRIB. Use --date YYYY-MM-DD.")

    # ── Resolve CDS download date ─────────────────────────────────────────
    cds_date = datetime.date.fromisoformat(cds_date_str) if cds_date_str else start
    print(f"  CDS download date: {cds_date}"
          + (" (from --date)" if cds_date_str else " (from GRIB)"))

    # ── Optionally filter to a single valid-time step ─────────────────────
    if valid_time_str:
        target_vt = np.datetime64(valid_time_str + "T00:00:00", "ns")
        print(f"  Selecting valid_time: {valid_time_str} 00Z")
        T = 1   # output will be single timestep

        def get(name):
            da = _find_var(datasets, name)
            if da is None:
                return None
            return _select_step(da, target_vt)
    else:
        def get(name):
            return _find_var(datasets, name)

    # ── Helper: shape a (T,H,W) or (H,W) → (T,1,H,W) float32, S→N ──────
    def pack(arr: np.ndarray) -> np.ndarray:
        arr = _ensure_3d(arr, T)[:T]
        arr = _to_sn(arr)
        return arr[:, np.newaxis, :, :].astype(np.float32)

    # ── Helper: fill from CDS cache or zeros ──────────────────────────────
    def from_cds(cds_dict: dict, key: str) -> np.ndarray:
        if key in cds_dict:
            arr = _ensure_3d(cds_dict[key], T)[:T]
            arr = _to_sn(arr)
            # Regrid if CDS resolution doesn't match target (e.g. 0.25°→1°)
            if arr.shape[-2] != H_out or arr.shape[-1] != W:
                sh, sw = arr.shape[-2] // H_out, arr.shape[-1] // W
                if sh > 1 or sw > 1:
                    # Nearest-neighbour downsample (safe for categorical fields)
                    arr = arr[:, ::sh, ::sw][:, :H_out, :W]
                else:
                    arr = arr[:, ::max(1, arr.shape[-2]//H_out),
                                 ::max(1, arr.shape[-1]//W)][:, :H_out, :W]
            return arr[:, np.newaxis, :, :].astype(np.float32)
        print(f"  WARNING: {key} missing — filling with zeros")
        return np.zeros((T, 1, H_out, W), dtype=np.float32)

    # ── Precipitation ─────────────────────────────────────────────────────
    # GenCast tp is a 12-hour interval accumulation per step.
    # Sum two consecutive 12-hour steps to get the 24-hour total.
    # GenCast tp: member 0 steps 1-4 contain valid 12-hour interval accumulations.
    # member 0 step 0 and all member 1 steps are NaN in the GRIB.
    #
    # With initialization at Dec 31 18Z, the step valid times are:
    #   step 1 → Jan 1 06Z,  step 2 → Jan 1 18Z,
    #   step 3 → Jan 2 06Z,  step 4 → Jan 2 18Z
    # ERA5 defines a daily total as 00Z–00Z (midnight to midnight UTC).
    # Steps 2+3 span Jan 1 06Z → Jan 2 06Z, which straddles the Jan 1/2
    # midnight boundary symmetrically (±6h) and is the closest 24h window
    # achievable from this initialization to the ERA5 00Z–00Z definition.
    tp_da = _find_var(datasets, "tp")
    if tp_da is None:
        raise RuntimeError("tp (precipitation) not found in GRIB")
    raw_tp = np.asarray(tp_da, dtype=np.float32)  # (member, step, lat, lon)
    if raw_tp.ndim == 4:
        # member 0, steps 2+3 — 24h window Jan1 06Z → Jan2 06Z, closest to ERA5 00Z day
        tp_arr = np.nan_to_num(raw_tp[0, 2], nan=0.0) + \
                 np.nan_to_num(raw_tp[0, 3], nan=0.0)
        print("  tp: summed member 0 steps [2]+[3] (24h Jan1 06Z → Jan2 06Z, ~ERA5 00Z day)")
    else:
        tp_arr = np.nan_to_num(raw_tp, nan=0.0)
    tp = pack(np.clip(tp_arr, 0, None))

    # ── Surface time-varying ──────────────────────────────────────────────
    t2m = pack(get("t2m"))
    u10 = pack(get("u10"))
    v10 = pack(get("v10"))
    sst_da = get("sst")
    sst = pack(sst_da) if sst_da is not None else np.zeros((T,1,H_out,W), dtype=np.float32)

    # t2m max/min: running cumulative max/min across steps (axis=0)
    t2m_3d = t2m[:, 0, :, :]   # (T, H, W)
    t2m_max = np.maximum.accumulate(t2m_3d, axis=0)[:, np.newaxis]
    t2m_min = np.minimum.accumulate(t2m_3d, axis=0)[:, np.newaxis]

    # ── Static surface fields ─────────────────────────────────────────────
    lsm_da = get("lsm")
    need_static = lsm_da is None

    if lsm_da is not None:
        lsm = pack(lsm_da)
    else:
        lsm = None   # filled from CDS below

    # Orography: surface geopotential z (pick the 2-D one, not pressure levels)
    orog_da = None
    for ds in datasets:
        if "z" in ds.data_vars:
            z_arr = np.asarray(ds["z"])
            if z_arr.ndim == 2:              # clearly static surface
                orog_da = ds["z"]
                break
            elif z_arr.ndim == 3 and z_arr.shape[0] <= 30:
                orog_da = ds["z"]

    # ── Pressure-level fields ─────────────────────────────────────────────
    t_pl = get("t")
    u_pl = get("u")
    v_pl = get("v")
    q_pl = get("q")
    z_pl = get("z")

    pl_fields = {}
    for plev in TARGET_PLEVS:
        pl_fields[f"temperature_{plev}"]          = pack(_select_plev(t_pl, plev))
        pl_fields[f"u_component_of_wind_{plev}"]  = pack(_select_plev(u_pl, plev))
        pl_fields[f"v_component_of_wind_{plev}"]  = pack(_select_plev(v_pl, plev))
        pl_fields[f"specific_humidity_{plev}"]    = pack(_select_plev(q_pl, plev))
        pl_fields[f"geopotential_{plev}"]         = pack(_select_plev(z_pl, plev))

    # ── CDS downloads ─────────────────────────────────────────────────────
    cds = download_missing_from_cds(str(cds_date), T, need_static, no_cds)

    if lsm is None:
        lsm = from_cds(cds, "land_sea_mask")

    if orog_da is not None:
        orog_arr = np.asarray(orog_da, dtype=np.float32)
        if orog_arr.ndim == 3:
            orog_arr = orog_arr[-1]          # (H, W)
        elif orog_arr.ndim == 4:
            orog_arr = orog_arr[0, -1]
        # GRIB z is geopotential (m²/s²); NPZ stores geometric height (m)
        orog_arr = orog_arr / GNEWTON
        orog = pack(orog_arr)
    else:
        orog = from_cds(cds, "orography")

    landcover = from_cds(cds, "landcover")
    swl1      = from_cds(cds, "volumetric_soil_water_layer_1")

    # ── Coordinate / metadata arrays ──────────────────────────────────────
    # Match exact NPZ lat grids: 180-pt→1° spacing, 720-pt→0.25° spacing
    step = 1.0 if H_out == 180 else 0.25
    lat_vals = np.arange(-90.0, -90.0 + H_out * step, step)[:H_out]   # S→N
    LAT = np.broadcast_to(
        lat_vals[np.newaxis, np.newaxis, :, np.newaxis], (T, 1, H_out, W)
    ).copy().astype(np.float64)

    doys = np.array(
        [(start + datetime.timedelta(days=i)).timetuple().tm_yday
         for i in range(T)], dtype=np.int32
    )
    # days_of_year stored as (T, 1, W, H) — transposed spatial dims
    DOYS = np.broadcast_to(
        doys[:, np.newaxis, np.newaxis, np.newaxis], (T, 1, W, H_out)
    ).copy().astype(np.int32)
    TOD  = np.zeros((T, 1, W, H_out), dtype=np.int32)

    # ── Save NPZ ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    save_dict = {
        "total_precipitation_24hr":         tp,
        "2m_temperature":                   t2m,
        "2m_temperature_max":               t2m_max.astype(np.float32),
        "2m_temperature_min":               t2m_min.astype(np.float32),
        "10m_u_component_of_wind":          u10,
        "10m_v_component_of_wind":          v10,
        "sea_surface_temperature":          sst,
        "land_sea_mask":                    lsm,
        "orography":                        orog,
        "landcover":                        landcover,
        "volumetric_soil_water_layer_1":    swl1,
        "lattitude":                        LAT,     # intentional typo matching NPZ
        "latitude":                         LAT,
        "days_of_year":                     DOYS,
        "time_of_day":                      TOD,
        "hrs_each_step":                    np.int64(24),
        "num_steps_per_shard":              np.int64(T),
        "extra_steps":                      np.int64(0),
    }
    save_dict.update(pl_fields)   # temperature/u/v/q/geopotential at 200/500/850

    np.savez_compressed(out_path, **save_dict)

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\nSaved: {out_path}")
    print(f"  Grid: {H}×{W}  |  Timesteps: {T}  |  Variables: {len(save_dict)}")
    print(f"  {'Variable':40s} {'Shape':25s} dtype")
    print(f"  {'-'*75}")
    for k, v in sorted(save_dict.items()):
        arr = np.asarray(v)
        print(f"  {k:40s} {str(arr.shape):25s} {arr.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert GenCast GRIB → ORBIT-2 NPZ (with CDS gap-fill)"
    )
    parser.add_argument("--grib",    required=True,
                        help="Input GRIB file")
    parser.add_argument("--date",    default=None,
                        help="Date for CDS download (YYYY-MM-DD). "
                             "If omitted, uses the date read from the GRIB file.")
    parser.add_argument("--n-steps", type=int, default=5,
                        help="Number of daily timesteps to extract (default: 5)")
    parser.add_argument("--out",     default="gencast_out.npz",
                        help="Output NPZ file (default: gencast_out.npz)")
    parser.add_argument("--valid-time", default=None, metavar="YYYY-MM-DD",
                        help="Extract only the GRIB step with this valid_time (00Z). "
                             "If omitted, all steps are extracted.")
    parser.add_argument("--no-cds",  action="store_true",
                        help="Skip CDS download; fill missing vars with zeros")
    args = parser.parse_args()

    convert(args.grib, None, args.n_steps, args.out, args.no_cds,
            cds_date_str=args.date, valid_time_str=args.valid_time)
