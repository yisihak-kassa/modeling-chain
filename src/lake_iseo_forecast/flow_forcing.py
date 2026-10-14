"""Focused AEM3D FLOW forcing generation for one forecast issue."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class FlowConfiguration:
    oglio_split_x: float
    oglio_split_y: float
    industrial_canal_q: float
    industrial_split_x: float
    industrial_split_y: float
    oglio_station_x: int
    oglio_station_y: int
    industrial_station_x: int
    industrial_station_y: int
    offtake_station: int
    point_lat: float
    point_lon: float


DischargeLoader = Callable[[datetime, datetime], pd.Series]


def _aem3d_times(index: pd.DatetimeIndex) -> list[str]:
    fraction = (index.hour + index.minute / 60 + index.second / 3600) / 24
    values = index.year * 1000 + index.day_of_year + fraction
    return [f"{value:.6f}" for value in values]


def _water_temperature(air: np.ndarray, month: np.ndarray, day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    season = np.sin(2 * np.pi * ((month * 30.5 + day) - 172) / 365.25)
    oglio = np.clip(0.72 * air + 3.2 + 1.2 * season, 0.0, 30.0)
    canal = np.clip(0.64 * air + 4.1 + 1.0 * season, 0.0, 30.0)
    return oglio, canal


def _write(path: Path, count: int, stations: str, labels: str, times: list[str], rows: np.ndarray) -> None:
    if rows.shape != (len(times), count):
        raise ValueError(f"{path.name} has unexpected shape {rows.shape}")
    if not np.isfinite(rows).all():
        raise ValueError(f"{path.name} contains missing or non-finite values")
    with path.open("w", newline="") as handle:
        handle.write(f"{count} data sets\n0 seconds between data\n{stations}\n{labels}\n")
        for timestamp, row in zip(times, rows, strict=True):
            handle.write(timestamp + " " + " ".join(f"{value:.3f}" for value in row) + "\n")


def write_flow_files(
    temperature_nc: Path,
    output_dir: Path,
    issue_time: datetime,
    *,
    forecast_hours: int,
    discharge_loader: DischargeLoader,
    configuration: FlowConfiguration,
) -> None:
    """Write five FLOW files including the horizon endpoint look-ahead row."""
    if issue_time.tzinfo is None:
        raise ValueError("issue_time must be timezone-aware")
    expected = pd.date_range(issue_time, periods=forecast_hours + 1, freq="h").tz_convert(None)
    discharge = discharge_loader(issue_time, issue_time + timedelta(hours=forecast_hours))
    discharge.index = pd.DatetimeIndex(discharge.index).tz_localize(None) if discharge.index.tz is not None else pd.DatetimeIndex(discharge.index)
    q = discharge.reindex(expected)
    if q.isna().any() or not np.isfinite(q.to_numpy(dtype=float)).all():
        raise ValueError("discharge does not provide complete finite hourly coverage including look-ahead")
    with xr.open_dataset(temperature_nc) as dataset:
        if "temperature" not in dataset:
            raise KeyError(f"{temperature_nc} does not contain 'temperature'")
        units = str(dataset["temperature"].attrs.get("units", "")).lower()
        if units not in {"degc", "c", "degree_celsius", "degrees_celsius"}:
            raise ValueError(f"temperature has unsupported units: {units!r}")
        selected = dataset["temperature"].sel(latitude=configuration.point_lat, longitude=configuration.point_lon, method="nearest")
        air = pd.Series(selected.values.astype(float), index=pd.DatetimeIndex(selected.valid_time.values)).reindex(expected)
    if air.isna().any() or not np.isfinite(air.to_numpy()).all():
        raise ValueError("temperature does not provide complete finite hourly coverage including look-ahead")
    output_dir.mkdir(parents=True, exist_ok=True)
    times = _aem3d_times(expected)
    tw_oglio, tw_canal = _water_temperature(air.to_numpy(), expected.month.to_numpy(), expected.day.to_numpy())
    tracer = np.ones(len(expected))
    cfg = configuration
    _write(output_dir / "FLOW_Oglio_River_x.dat", 3, f"0 {cfg.oglio_station_x} {cfg.oglio_station_x} {cfg.oglio_station_x}", "TIME INFLOW WTR_TEMP TRACER_1", times, np.column_stack((q * cfg.oglio_split_x, tw_oglio, tracer)))
    _write(output_dir / "FLOW_Oglio_River_y.dat", 3, f"0 {cfg.oglio_station_y} {cfg.oglio_station_y} {cfg.oglio_station_y}", "TIME INFLOW WTR_TEMP TRACER_1", times, np.column_stack((q * cfg.oglio_split_y, tw_oglio, tracer)))
    _write(output_dir / "FLOW_Industrial_Canal_x.dat", 3, f"0 {cfg.industrial_station_x} {cfg.industrial_station_x} {cfg.industrial_station_x}", "TIME INFLOW WTR_TEMP TRACER_1", times, np.column_stack((np.full(len(expected), cfg.industrial_canal_q * cfg.industrial_split_x), tw_canal, tracer)))
    _write(output_dir / "FLOW_Industrial_Canal_y.dat", 3, f"0 {cfg.industrial_station_y} {cfg.industrial_station_y} {cfg.industrial_station_y}", "TIME INFLOW WTR_TEMP TRACER_1", times, np.column_stack((np.full(len(expected), cfg.industrial_canal_q * cfg.industrial_split_y), tw_canal, tracer)))
    _write(output_dir / "FLOW_Offtake.dat", 2, f"0 {cfg.offtake_station} {cfg.offtake_station}", "TIME OUTFLOW WTR_TEMP", times, np.column_stack((q + cfg.industrial_canal_q, tw_oglio)))
