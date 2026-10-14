"""Extract one ICON-2I issue and write its AEM3D MET forcing files."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_FORECAST_HOURS = 72

_GRIB_VARIABLES = {
    "temperature": ("t", False, False),
    "cloud_cover": ("clct", False, False),
    "relative_humidity_1000hpa": ("r", False, False),
    "shortwave_radiation": ("avg_snswrf", True, False),
    "precipitation": ("tp", False, True),
}

_MET_OUTPUTS = {
    "MET_AirTemp.dat": ("AIR_TEMP", "temperature", 1.0, None),
    "MET_CCF.dat": ("CLOUDS", "cloud_cover", 0.01, (0.0, 1.0)),
    "MET_RelHum.dat": (
        "REL_HUM",
        "relative_humidity_1000hpa",
        0.01,
        (0.0, 1.0),
    ),
    "MET_SolRad.dat": ("SOLAR_RAD", "shortwave_radiation", 1.0, (0.0, None)),
    "MET_WindDir.dat": ("WIND_DIR", "wind_direction", 1.0, (0.0, 360.0)),
    "MET_WindSpd.dat": ("WIND_SPEED", "wind_speed", 1.0, (0.0, None)),
}


def _utc_naive(value: datetime | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("issue_time must be timezone-aware")
    return timestamp.tz_convert("UTC").tz_localize(None)


def _process_averaged(da: xr.DataArray) -> xr.DataArray:
    """Convert forecast-step averages into interval values."""
    runs = []
    for time_value in da.time.values:
        run = da.sel(time=time_value)
        step_hours = run.step.values / np.timedelta64(1, "h")
        step_hours = step_hours.astype(float)
        step_coord = xr.DataArray(step_hours, coords={"step": run.step}, dims="step")

        energy = run * step_coord
        energy = xr.DataArray(
            np.maximum.accumulate(energy.values, axis=0),
            coords=energy.coords,
            dims=energy.dims,
        )
        delta_hours = np.diff(step_hours)
        delta_hours[delta_hours == 0] = np.nan
        interval = energy.diff("step") / xr.DataArray(
            delta_hours,
            coords={"step": run.step.values[1:]},
            dims="step",
        )

        first_hours = step_hours[0] if step_hours[0] > 0 else 1.0
        first = (run.isel(step=0) * step_hours[0] / first_hours).expand_dims(
            {"step": [run.step.values[0]]}
        )
        runs.append(xr.concat([first, interval], dim="step").expand_dims(time=[time_value]))

    return xr.concat(runs, dim="time")


def _process_accumulated(da: xr.DataArray) -> xr.DataArray:
    """Convert cumulative precipitation depth to hourly interval rates."""
    runs = []
    for time_value in da.time.values:
        run = da.sel(time=time_value)
        step_hours = np.asarray(
            run.step.values / np.timedelta64(1, "h"), dtype=float
        )
        cumulative = xr.DataArray(
            np.maximum.accumulate(run.values, axis=0),
            coords=run.coords,
            dims=run.dims,
            attrs=run.attrs,
        )
        delta_hours = np.diff(step_hours)
        if np.any(delta_hours <= 0):
            raise ValueError("precipitation forecast steps must be strictly increasing")
        interval = cumulative.diff("step") / xr.DataArray(
            delta_hours,
            coords={"step": run.step.values[1:]},
            dims="step",
        )
        first = xr.zeros_like(run.isel(step=0)).expand_dims(
            {"step": [run.step.values[0]]}
        )
        rate = xr.concat([first, interval], dim="step")
        rate.attrs = run.attrs.copy()
        rate.attrs["units"] = "kg m-2 h-1"
        runs.append(rate.expand_dims(time=[time_value]))
    return xr.concat(runs, dim="time")


def reference_time_series(
    da: xr.DataArray,
    issue_time: datetime | pd.Timestamp,
    *,
    averaged: bool = False,
    accumulated: bool = False,
) -> xr.DataArray:
    """Select exactly one forecast reference time and expose its valid-time axis."""
    if "time" not in da.dims or "step" not in da.dims:
        raise ValueError(f"Expected time and step dimensions, got {da.dims}")

    target = _utc_naive(issue_time)
    available = pd.DatetimeIndex(da.time.values)
    if target not in available:
        raise ValueError(
            f"Reference time {target} not found; available range is "
            f"{available.min()} through {available.max()}"
        )

    issue = da.sel(time=[target.to_datetime64()])
    if averaged and accumulated:
        raise ValueError("a variable cannot be both averaged and accumulated")
    if averaged:
        issue = _process_averaged(issue)
    elif accumulated:
        issue = _process_accumulated(issue)
    issue = issue.isel(time=0, drop=True).load()

    spatial_dims = [dimension for dimension in issue.dims if dimension != "step"]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected two spatial dimensions, got {issue.dims}")

    valid_times = target.to_datetime64() + issue.step.values
    issue = issue.assign_coords(valid_time=("step", valid_times))
    issue = issue.swap_dims({"step": "valid_time"})
    issue = issue.transpose("valid_time", *spatial_dims)
    return issue.assign_coords(forecast_reference_time=target.to_datetime64())


def _convert_units(da: xr.DataArray, variable: str) -> xr.DataArray:
    units = str(da.attrs.get("units", ""))
    if variable == "temperature":
        if units.lower() == "k":
            converted = da - 273.15
            converted.attrs = da.attrs.copy()
            converted.attrs["units"] = "degC"
            return converted
        if units.lower() not in {"degc", "c", "degree_celsius", "degrees_celsius"}:
            raise ValueError(f"temperature has unsupported units: {units!r}")
    return da


def _require_units(da: xr.DataArray, variable: str) -> None:
    units = str(da.attrs.get("units", "")).strip().lower().replace("**", "^")
    accepted = {
        "temperature": {"degc", "c", "degree_celsius", "degrees_celsius"},
        "cloud_cover": {"%", "percent"},
        "relative_humidity_1000hpa": {"%", "percent"},
        "shortwave_radiation": {"w m-2", "w/m^2", "w m^-2"},
        "wind_u": {"m s-1", "m/s", "m s^-1"},
        "wind_v": {"m s-1", "m/s", "m s^-1"},
        "precipitation": {"kg m-2", "kg/m^2", "kg m^-2"},
    }
    if units not in accepted[variable]:
        raise ValueError(f"{variable} has unsupported units: {units!r}")


def _read_grib_variable(
    path: Path,
    short_name: str,
    output_name: str,
    issue_time: datetime,
    *,
    averaged: bool = False,
    accumulated: bool = False,
) -> xr.DataArray:
    with xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    ) as dataset:
        if short_name not in dataset:
            raise KeyError(f"{path} does not contain GRIB variable {short_name!r}")
        source = dataset[short_name]




        if "time" not in source.dims and "time" in source.coords:
            reference_time = np.asarray(source.coords["time"].values).reshape(-1)[0]
            source = source.expand_dims(time=np.asarray([reference_time]))
        squeeze = [
            name
            for name, size in source.sizes.items()
            if size == 1
            and name not in {"time", "step", "latitude", "longitude", "x", "y"}
        ]
        if squeeze:
            source = source.squeeze(squeeze, drop=True)
        source = _convert_units(source, output_name)
        _require_units(source, output_name)
        result = reference_time_series(
            source,
            issue_time,
            averaged=averaged,
            accumulated=accumulated,
        )
    result.name = output_name
    return result


def _expected_times(issue_time: datetime, forecast_hours: int) -> pd.DatetimeIndex:
    if forecast_hours < 1:
        raise ValueError("forecast_hours must be positive")
    return pd.date_range(
        _utc_naive(issue_time),
        periods=forecast_hours + 1,
        freq="h",
    )


def _validate_common_axis(
    series: Mapping[str, xr.DataArray],
    issue_time: datetime,
    forecast_hours: int,
) -> pd.DatetimeIndex:
    expected = _expected_times(issue_time, forecast_hours)
    for name, data in series.items():
        actual = pd.DatetimeIndex(data.valid_time.values)
        if not actual.equals(expected):
            missing = expected.difference(actual)
            extra = actual.difference(expected)
            raise ValueError(
                f"{name} has an unexpected time axis; "
                f"missing={list(missing[:5])}, extra={list(extra[:5])}"
            )
        if not np.isfinite(data.values).all():
            raise ValueError(f"{name} contains missing or non-finite values")
    return expected


def _select_horizon(data: xr.DataArray, issue_time: datetime, forecast_hours: int) -> xr.DataArray:
    expected = _expected_times(issue_time, forecast_hours)
    actual = pd.DatetimeIndex(data.valid_time.values)
    missing = expected.difference(actual)
    if len(missing):
        raise ValueError(f"{data.name} is missing required horizon times: {list(missing[:5])}")
    return data.sel(valid_time=expected.to_numpy(dtype="datetime64[ns]"))


def _save_netcdf(data: xr.DataArray, output_dir: Path) -> None:
    dataset = data.to_dataset()
    dataset.attrs.update({"Conventions": "CF-1.6", "source": "ICON-2I"})
    dataset.to_netcdf(output_dir / f"{data.name}.nc")
    dataset.close()


def extract_issue_grib(
    grib_files: Mapping[str, Path],
    output_dir: Path,
    issue_time: datetime,
    *,
    forecast_hours: int = DEFAULT_FORECAST_HOURS,
    grib_reference_time: datetime | None = None,
) -> Path:
    """Extract one valid-time horizon from one explicit GRIB reference cycle."""
    required = set(_GRIB_VARIABLES) | {"wind_u", "wind_v"}
    missing = sorted(required.difference(grib_files))
    if missing:
        raise ValueError("missing GRIB variables: " + ", ".join(missing))

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    reference_time = grib_reference_time or issue_time
    series = {}
    for name, (short_name, averaged, accumulated) in _GRIB_VARIABLES.items():
        series[name] = _select_horizon(_read_grib_variable(
            Path(grib_files[name]),
            short_name,
            name,
            reference_time,
            averaged=averaged,
            accumulated=accumulated,
        ), issue_time, forecast_hours)

    u = _select_horizon(_read_grib_variable(Path(grib_files["wind_u"]), "u", "wind_u", reference_time), issue_time, forecast_hours)
    v = _select_horizon(_read_grib_variable(Path(grib_files["wind_v"]), "v", "wind_v", reference_time), issue_time, forecast_hours)
    speed = np.hypot(u, v)
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    speed.name = "wind_speed"
    direction.name = "wind_direction"
    series["wind_speed"] = speed
    series["wind_direction"] = direction

    _validate_common_axis(series, issue_time, forecast_hours)
    for data in series.values():
        _save_netcdf(data, output_dir)
    return output_dir


def _load_zone_mapping(npz_path: Path) -> tuple[np.ndarray, np.ndarray, int, str]:
    with np.load(npz_path) as mapping:
        required = {"matrice_corrispondenze", "bati", "X_U10"}
        missing = sorted(required.difference(mapping.files))
        if missing:
            raise ValueError("zone mapping is missing arrays: " + ", ".join(missing))
        correspondences = mapping["matrice_corrispondenze"]
        bathymetry = mapping["bati"].ravel(order="F")
        met_shape = mapping["X_U10"].T.shape

    valid_rows = []
    for row_number in range(correspondences.shape[0]):
        row = correspondences[row_number, 1:]
        indices = row[~np.isnan(row)].astype(int) - 1
        if indices.size and np.any(bathymetry[indices] != 999.0):
            valid_rows.append(row_number)

    if not valid_rows:
        raise ValueError("zone mapping contains no valid meteorological zones")
    met_rows, met_cols = np.unravel_index(valid_rows, met_shape, order="F")
    zone_ids = "0 " + " ".join(str(301 + index) for index in range(len(valid_rows)))
    return met_rows, met_cols, len(valid_rows), zone_ids


def _aem3d_time_index(times: pd.DatetimeIndex) -> np.ndarray:
    fraction = (times.hour + (times.minute + times.second / 60) / 60) / 24
    return (times.year * 1000 + times.day_of_year + fraction).to_numpy(dtype=float)


def _load_netcdf_series(nc_dir: Path, name: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    path = nc_dir / f"{name}.nc"
    with xr.open_dataset(path) as dataset:
        if name not in dataset:
            raise KeyError(f"{path} does not contain {name!r}")
        times = pd.DatetimeIndex(dataset.valid_time.values)
        values = dataset[name].values.astype(float)
    return times, values


def write_met_files(nc_dir: Path, zone_mapping: Path, output_dir: Path) -> None:
    """Write the complete seven-file AEM3D meteorological forcing set."""
    nc_dir = Path(nc_dir)
    output_dir = Path(output_dir)
    met_rows, met_cols, zone_count, zone_ids = _load_zone_mapping(Path(zone_mapping))

    loaded = {}
    common_times = None
    for _, name, _, _ in _MET_OUTPUTS.values():
        times, values = _load_netcdf_series(nc_dir, name)
        if common_times is None:
            common_times = times
        elif not times.equals(common_times):
            raise ValueError(f"{name} does not share the common MET time axis")
        loaded[name] = values

    rain_times, rain_values = _load_netcdf_series(nc_dir, "precipitation")
    if common_times is None:
        common_times = rain_times
    elif not rain_times.equals(common_times):
        raise ValueError("precipitation does not share the common MET time axis")
    loaded["precipitation"] = rain_values

    assert common_times is not None
    if not common_times.is_monotonic_increasing or common_times.has_duplicates:
        raise ValueError("MET time axis must be strictly increasing and unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    aem3d_times = [f"{value:.6f}" for value in _aem3d_time_index(common_times)]
    with ExitStack() as stack:
        outputs = {}
        for filename, (label, name, scale, limits) in _MET_OUTPUTS.items():
            values = loaded[name][:, met_rows, met_cols] * scale
            if limits is not None:
                lower, upper = limits
                if lower is not None:
                    values = np.maximum(values, lower)
                if upper is not None:
                    values = np.minimum(values, upper)
            if values.shape != (len(common_times), zone_count):
                raise ValueError(f"{filename} has unexpected shape {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"{filename} contains missing or non-finite values")

            handle = stack.enter_context((output_dir / filename).open("w", newline=""))
            handle.write(f"{zone_count} data sets\n")
            handle.write("0 seconds between data\n")
            handle.write(zone_ids + "\n")
            handle.write("DAY " + " ".join([label] * zone_count) + "\n")
            outputs[filename] = (handle, values)

        rain = loaded["precipitation"][:, met_rows, met_cols].mean(axis=1) * 0.024
        if rain.shape != (len(common_times),) or not np.isfinite(rain).all():
            raise ValueError("MET_Rain.dat contains missing or non-finite values")
        if np.any(rain < 0):
            raise ValueError("MET_Rain.dat contains negative rainfall")
        rain_handle = stack.enter_context(
            (output_dir / "MET_Rain.dat").open("w", newline="")
        )
        rain_handle.write("1 data sets\n")
        rain_handle.write("0 seconds between data\n")
        rain_handle.write("0 0\n")
        rain_handle.write("DAY RAIN\n")

        for row_number, timestamp in enumerate(aem3d_times):
            for handle, values in outputs.values():
                row = " ".join(f"{value:.2f}" for value in values[row_number])
                handle.write(f"{timestamp} {row}\n")
            rain_handle.write(f"{timestamp} {rain[row_number]:.6f}\n")
