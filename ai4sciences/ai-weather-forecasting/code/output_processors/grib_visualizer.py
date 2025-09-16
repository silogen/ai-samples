"""
GRIB File Visualizer

This script loads a GRIB file and creates animated GIFs
for each variable over time, optionally across pressure levels.

Usage:
    python grib_visualizer.py --input <path_to_grib_file>

Functions:
    open_grib_datasets: Opens GRIB datasets with proper time decoding.
    describe_datasets: Describes datasets and extracts model information.
    select_data: Selects and subsets data based on time dimensions and pressure level.
    compute_scale: Computes color scale normalization and colormap.
    format_units_latex: Formats units with proper LaTeX notation.
    save_variable_gif: Saves animated GIF for a variable.
    fix_longitude: Fixes longitude coordinates to range from -180 to 180.
    get_units: Extracts units from xarray DataArray attributes.
    process_dataset: Processes a single dataset to create GIF animations.
    main: Main function to process command line arguments and run visualization.
"""

import os
import re
import sys
from typing import List, Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cf
import cfgrib
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

def get_official_model_name(model_name:str) -> str:
    """
    Get the official model name corresponding to the string used in code.

    Returns
    -------
    str
        Official model name.
    """
    model_names = {
        'gencast': 'GenCast',
        'panguweather': 'Pangu-Weather',
        'aurora': 'Aurora',
    }
    return model_names.get(model_name, 'Other')
    
def open_grib_datasets(path: str) -> List[xr.Dataset]:
    """
    Open GRIB datasets with proper time decoding.

    Parameters
    ----------
    path : str
        Path to the GRIB file.

    Returns
    -------
    List[xr.Dataset]
        List of xarray Datasets loaded from the GRIB file.
    """
    return cfgrib.open_datasets(path, decode_timedelta=xr.coding.times.CFTimedeltaCoder)


def describe_datasets(datasets: List[xr.Dataset], path: str) -> List[dict]:
    """
    Describe datasets and extract model information such as model name, index, and dimension presence.

    Parameters
    ----------
    datasets : List[xr.Dataset]
        List of xarray Datasets.
    path : str
        Path to the GRIB file.

    Returns
    -------
    List[dict]
        List of dictionaries describing each dataset.
    """
    descriptions = []
    for ds_idx, ds in enumerate(datasets):
        if 'gencast' in path:
            model_name = 'gencast'
        elif 'pangu' in path:
            model_name = 'panguweather'
        elif 'aurora' in path:
            model_name = 'aurora'
        else:
            model_name = 'other'

        descriptions.append(
            {
                'model_name': model_name,
                'dataset_idx': ds_idx,
                'dataset': ds,
                'has_levels': 'isobaricInhPa' in ds,
                'has_time': 'time' in ds.dims,
                'has_step': 'step' in ds.dims,
            }
        )
    return descriptions


def select_data(data: xr.DataArray, has_step: bool, has_time: bool, max_steps: int, level: Optional[int] = None) -> xr.DataArray:
    """
    Select and subset data based on time dimensions and pressure level.

    Parameters
    ----------
    data : xr.DataArray
        The data array to subset.
    has_step : bool
        Whether the data has a 'step' dimension.
    has_time : bool
        Whether the data has a 'time' dimension.
    max_steps : int
        Maximum number of steps to select.
    level : Optional[int]
        Pressure level to select, if applicable.

    Returns
    -------
    xr.DataArray
        The subsetted and sorted data array.
    """
    if has_step and data.sizes.get('step', 0) > max_steps:
        data = data.isel(step=range(max_steps))
    if has_time and data.sizes.get('time', 0) > max_steps:
        data = data.isel(time=range(max_steps))
    if level is not None and 'isobaricInhPa' in data.coords:
        data = data.sel(isobaricInhPa=level)
    return data.sortby('latitude').sortby('longitude')


def compute_scale(
    data: xr.DataArray,
    center: Optional[float] = None,
    robust: bool = False,
    global_min: Optional[float] = None,
    global_max: Optional[float] = None,
) -> Tuple[matplotlib.colors.Normalize, str]:
    """
    Compute color scale normalization and colormap for visualization.

    Parameters
    ----------
    data : xr.DataArray
        Data array to compute scale for.
    center : Optional[float]
        Center value for diverging colormap.
    robust : bool
        Whether to use robust percentiles.
    global_min : Optional[float]
        Global minimum value.
    global_max : Optional[float]
        Global maximum value.

    Returns
    -------
    Tuple[matplotlib.colors.Normalize, str]
        Normalization and colormap name.
    """
    if global_min is not None and global_max is not None:
        vmin = global_min
        vmax = global_max
    else:
        vmin = np.nanpercentile(data, 2 if robust else 0)
        vmax = np.nanpercentile(data, 98 if robust else 100)

    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff

    cmap = 'RdBu_r' if center is not None else 'viridis'
    return matplotlib.colors.Normalize(vmin, vmax), cmap


def format_units_latex(units: Optional[str]) -> str:
    """
    Format units with proper LaTeX notation for display in plots.

    Parameters
    ----------
    units : Optional[str]
        Units string to format.

    Returns
    -------
    str
        LaTeX formatted units string.
    """
    if not units:
        return ''

    # Common unit conversions to LaTeX
    unit_conversions = {
        'm**2 s**-2': r'm^2\,s^{-2}',
        'm s**-1': r'm\,s^{-1}',
        'K': r'K',
        'Pa': r'Pa',
        'kg m**-2 s**-1': r'kg\,m^{-2}\,s^{-1}',
        'kg kg**-1': r'kg\,kg^{-1}',
        's**-1': r's^{-1}',
        'J kg**-1': r'J\,kg^{-1}',
        'W m**-2': r'W\,m^{-2}',
        'kg m**-2': r'kg\,m^{-2}',
        'dimensionless': r'',
        '1': r'',
        'm': r'm',
        'degC': r'°C',
        'deg C': r'°C',
        'C': r'°C',
    }

    # Check for exact matches first
    if units in unit_conversions:
        return unit_conversions[units]

    # General formatting for common patterns
    formatted = units
    # Replace ** with ^ for exponents
    formatted = formatted.replace('**', '^')
    # Add spacing before units
    formatted = formatted.replace(' ', r'\,')
    # Handle negative exponents
    formatted = formatted.replace('^-', '^{-')
    if '^{-' in formatted and not formatted.endswith('}'):
        # Find the exponent and wrap it properly
        formatted = re.sub(r'\^{-(\d+)', r'^{-\1}', formatted)

    return formatted


def save_variable_gif(
    data: xr.DataArray,
    time_dim: str,
    cmap: str,
    norm: matplotlib.colors.Normalize,
    var: str,
    frames: int,
    out_path: str,
    official_model_name: str,
    nanoseconds_per_step: Optional[int],
    prefix: str = '',
    projection: ccrs.Projection = ccrs.PlateCarree(),
    units: Optional[str] = None,
) -> None:
    """
    Save animated GIF for a variable over time or steps.

    Parameters
    ----------
    data : xr.DataArray
        Data array to visualize.
    time_dim : str
        Name of the time dimension ('step' or 'time').
    cmap : str
        Colormap name.
    norm : matplotlib.colors.Normalize
        Color normalization.
    var : str
        Variable name.
    frames : int
        Number of frames in the animation.
    out_path : str
        Output path for the GIF file.
    official_model_name : str
        Name of the model.
    nanoseconds_per_step : Optional[int]
        Nanoseconds per step for time calculation.
    prefix : str
        Prefix for the plot title.
    projection : ccrs.Projection
        Cartopy projection for the plot.
    units : Optional[str]
        Units for the variable.

    Returns
    -------
    None
    """
    # Compute global min/max across all frames for consistent colorbar
    global_min = float(np.nanmin(data.values))
    global_max = float(np.nanmax(data.values))

    # Recompute normalization with global values
    norm, cmap = compute_scale(data, global_min=global_min, global_max=global_max)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Create figure with minimal margins
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': projection})

    # Remove margins and padding
    fig.subplots_adjust(left=0.05, bottom=0, right=0.85, top=0.95, wspace=0, hspace=0)

    ax.add_feature(cf.COASTLINE.with_scale('50m'), lw=0.5)
    ax.add_feature(cf.BORDERS.with_scale('50m'), lw=0.3)
    ax.gridlines(draw_labels=False, linewidth=0.5, color='black', alpha=0.5, linestyle='--')

    # Set global extent for better visualization
    ax.set_global()

    # Remove axis spines and ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['geo'].set_visible(False)

    if isinstance(projection, (ccrs.Mollweide, ccrs.Robinson)):
        # Create coordinate meshgrid
        lonmesh, latmesh = np.meshgrid(data.longitude.values, data.latitude.values)
        im = ax.pcolormesh(
            lonmesh,
            latmesh,
            data.isel({time_dim: 0}, missing_dims='ignore'),
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
            shading='auto',
        )
    else:
        im = ax.imshow(
            data.isel({time_dim: 0}, missing_dims='ignore'),
            cmap=cmap,
            norm=norm,
            origin='upper',
            transform=projection,
            extent=[0, 360, 90, -90],
        )

    ax.set_title(f'{official_model_name}' f'{var} {prefix} ({time_dim})', pad=10)

    # Create smaller, more proportional colorbar
    latex_units = format_units_latex(units)
    if latex_units:
        cbar_label = f'{var} $[{latex_units}]$'
    else:
        cbar_label = var

    # Smaller colorbar with better proportions
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.6, aspect=30, pad=0.02, label=cbar_label)
    cbar.ax.tick_params(labelsize=9)

    def update(frame, prefix=prefix):
        """Update function for animation."""
        im.set_data(data.isel({time_dim: frame}, missing_dims='ignore'))
        #cbar = plt.colorbar(mappable, ax=ax, orientation='vertical', shrink=0.6, aspect=30, pad=0.02, label=cbar_label)
        #cbar.ax.tick_params(labelsize=9)
        if nanoseconds_per_step:
            time_hours = frame * nanoseconds_per_step / (60 * 60 * 1e9)
        else:
            time_hours = 0
        ax.set_title(f'{official_model_name} {var} {prefix} {time_dim}={frame}, time={time_hours:.1f} hours', pad=10)

    def step_to_fps(nanoseconds: int) -> float:
        """Convert step size to frames per second."""
        hours = nanoseconds / (60 * 60 * 1e9)
        return 2 * 24 / hours if hours > 0 else 1.0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Create animation with tight layout
    ani = animation.FuncAnimation(fig, update, frames=frames)
    fps = step_to_fps(nanoseconds_per_step) if nanoseconds_per_step else 5

    # Save with minimal whitespace
    ani.save(out_path, writer='pillow', fps=fps, savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0.1, 'facecolor': 'white'})
    plt.close(fig)


def fix_longitude(data: xr.DataArray) -> xr.DataArray:
    """
    Fix longitude coordinates to range from -180 to 180 for proper visualization.

    Parameters
    ----------
    data : xr.DataArray
        Data array with longitude coordinates.

    Returns
    -------
    xr.DataArray
        Data array with fixed longitude coordinates.
    """
    if 'longitude' in data.coords:
        lon_name = 'longitude'
        data['lon'] = xr.where(data[lon_name] > 180, data[lon_name] - 360, data[lon_name])
        data = data.swap_dims({'longitude': 'lon'}).sel(lon=sorted(data.lon)).drop_vars('longitude')
        data = data.rename({'lon': 'longitude'})
    return data


def get_units(data: xr.DataArray) -> Optional[str]:
    """
    Extract units from xarray DataArray attributes.

    Parameters
    ----------
    data : xr.DataArray
        Data array to extract units from.

    Returns
    -------
    Optional[str]
        Units string if available, else None.
    """
    return data.attrs.get('units', data.attrs.get('GRIB_units', None))


def process_dataset(
    ds: xr.Dataset,
    dataset_idx: int,
    output_dir: str,
    model_name: str,
    has_levels: bool,
    has_time: bool,
    has_step: bool,
    max_steps: int,
    projection: ccrs.Projection = ccrs.PlateCarree(),
) -> None:
    """
    Process a single dataset to create GIF animations for each variable and level.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to process.
    dataset_idx : int
        Index of the dataset.
    output_dir : str
        Output directory for GIFs.
    model_name : str
        Name of the model.
    has_levels : bool
        Whether the dataset has pressure levels.
    has_time : bool
        Whether the dataset has time dimension.
    has_step : bool
        Whether the dataset has step dimension.
    max_steps : int
        Maximum number of steps to process.
    projection : ccrs.Projection
        Cartopy projection for plots.

    Returns
    -------
    None
    """
    step_to_nanoseconds = int(ds.get('step')[1]) if has_step and 'step' in ds else None
    model_output_dir = os.path.join(output_dir, model_name)

    for var in ds.data_vars:
        variable_data = ds[var]
        units = get_units(variable_data)

        if has_levels and 'isobaricInhPa' in variable_data.coords:
            levels = [int(level) for level in variable_data.coords['isobaricInhPa'].values]
        else:
            levels = [None]

        level_type = 'surface_level' if levels == [None] else 'pressure levels'
        print(f'Processing: {var} on {level_type}')

        for level in levels:
            data = select_data(variable_data, has_step, has_time, max_steps, level)
            data = fix_longitude(data)
            if has_time and has_step:
                data = data.isel(time=0)

            time_dim = 'step' if has_step else ('time' if has_time else None)
            if not time_dim:
                print(f'Skipping {var}, no time dimension. Dataset #{dataset_idx} likely static map.')
                continue

            frames = data.sizes[time_dim]
            prefix = f'level_{level}' if level is not None else ''
            gif_name = f'{var}.gif'
            out_path = os.path.join(model_output_dir, prefix, gif_name)

            official_model_name = get_official_model_name(model_name)

            save_variable_gif(
                data, time_dim, None, None, var, frames, out_path, official_model_name, step_to_nanoseconds, prefix, projection=projection, units=units
            )


def main():
    """
    Main function to process command line arguments and run visualization.

    Parses command line arguments, loads GRIB datasets, and processes each dataset to generate GIFs.

    Returns
    -------
    None
    """
    if len(sys.argv) < 3 or '--input' not in sys.argv:
        print('Usage: python grib_visualizer.py --input <path_to_grib_file>')
        sys.exit(1)

    input_file = sys.argv[sys.argv.index('--input') + 1]
    datasets = open_grib_datasets(input_file)
    descriptions = describe_datasets(datasets, input_file)

    for desc in descriptions:
        print('Processing dataset #', desc['dataset_idx'])
        process_dataset(
            ds=desc['dataset'],
            dataset_idx=desc['dataset_idx'],
            output_dir='outputs/',
            model_name=desc['model_name'],
            has_levels=desc['has_levels'],
            has_time=desc['has_time'],
            has_step=desc['has_step'],
            max_steps=desc['dataset'].sizes.get('step', 40),
            projection=ccrs.PlateCarree(),
        )


if __name__ == '__main__':
    main()
