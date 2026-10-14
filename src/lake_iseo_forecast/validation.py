"""Validation shared by forcing preparation and bounded AEM3D execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import re

import numpy as np
import xarray as xr

TIME_COMPONENTS = ("Year", "Month", "Day", "Hour", "Minute", "Second")
DRYING_MIN_CELLS = 2
DRYING_MAX_FRACTION = 0.01
RESTART_TIME_RE = re.compile(r"sim_current\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")


def parse_dat_axis(path: Path) -> np.ndarray:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"forcing file is missing or empty: {path}")
    rows = [line.split() for line in path.read_text(errors="replace").splitlines()[4:] if line.strip()]
    if not rows:
        raise ValueError(f"forcing file contains no data rows: {path}")
    try:
        values = np.asarray([[float(value) for value in row] for row in rows])
    except ValueError as exc:
        raise ValueError(f"forcing file contains non-numeric data: {path}") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"forcing file contains missing or non-finite values: {path}")
    if np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError(f"forcing time axis is not strictly increasing: {path}")
    return values[:, 0]


def restart_time(path: Path) -> datetime:
    metadata = path.with_suffix(".txt")
    if not path.is_file() or path.stat().st_size == 0 or not metadata.is_file() or metadata.stat().st_size == 0:
        raise FileNotFoundError(f"restart pair is missing or empty: {path}")
    match = RESTART_TIME_RE.search(metadata.read_text(errors="replace"))
    if not match:
        raise ValueError(f"could not parse sim_current from {metadata}")
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def netcdf_times(path: Path) -> np.ndarray:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"AEM3D output is missing or empty: {path}")
    with xr.open_dataset(path, decode_times=False) as dataset:
        missing = [name for name in TIME_COMPONENTS if name not in dataset]
        if missing or dataset.sizes.get("T", 0) == 0:
            raise ValueError(f"{path} has no complete AEM3D time axis")
        result = []
        for index in range(dataset.sizes["T"]):
            parts = [int(round(float(dataset[name].values[index]))) for name in TIME_COMPONENTS]
            result.append(np.datetime64(datetime(*parts[:3]) + timedelta(hours=parts[3], minutes=parts[4], seconds=parts[5]), "ns"))
        times = np.asarray(result)
        if len(times) > 1 and np.any(times[1:] <= times[:-1]):
            raise ValueError(f"{path} time axis is not strictly increasing")
        for name, array in dataset.data_vars.items():
            if "T" not in array.dims or not np.issubdtype(array.dtype, np.number):
                continue
            values = np.moveaxis(array.values, array.get_axis_num("T"), 0)
            if np.isinf(values).any():
                raise ValueError(f"{path}:{name} contains infinite values")
            finite = np.isfinite(values)
            if finite.all():
                continue
            _validate_wet_dry(path, name, finite)
    return times


def _edge_cells(finite_frame: np.ndarray) -> np.ndarray:
    """Cells with a non-finite neighbour along some axis; the array border counts as finite."""
    frame = np.asarray(finite_frame, dtype=bool)
    edge = np.zeros(frame.shape, dtype=bool)
    for axis in range(frame.ndim):
        pad = [(1, 1) if other == axis else (0, 0) for other in range(frame.ndim)]
        padded = np.pad(frame, pad, constant_values=True)
        size = frame.shape[axis]
        edge |= ~np.take(padded, range(0, size), axis=axis) | ~np.take(padded, range(2, size + 2), axis=axis)
    return edge


def _validate_wet_dry(path: Path, name: str, finite: np.ndarray) -> None:
    """Accept wetting and drying at the edge of the finite region; reject anything else.

    Non-finite cells mark inactive water. Every frame keeps at least one finite
    cell. A finite cell may turn non-finite only while it borders a non-finite
    cell, so an interior hole is rejected, and a single step may drop at most
    max(DRYING_MIN_CELLS, DRYING_MAX_FRACTION of the previous finite cells).
    """
    if not finite[0].any():
        raise ValueError(f"{path}:{name} has a wholly non-finite first frame")
    for index in range(1, len(finite)):
        if not finite[index].any():
            raise ValueError(f"{path}:{name} has a wholly non-finite frame at time index {index}")
    for index in range(len(finite) - 1):
        lost = finite[index] & ~finite[index + 1]
        count = int(lost.sum())
        if not count:
            continue
        limit = max(DRYING_MIN_CELLS, math.ceil(DRYING_MAX_FRACTION * int(finite[index].sum())))
        if count > limit:
            raise ValueError(f"{path}:{name} has {count} finite cells that become non-finite at once (limit {limit}) after time index {index}")
        if np.any(lost & ~_edge_cells(finite[index])):
            raise ValueError(f"{path}:{name} has a finite cell that becomes non-finite away from the edge of the finite region after time index {index}")


@dataclass(frozen=True)
class OutputExpectation:
    rows: int
    first: datetime
    last: datetime


def validate_output(path: Path, expected: OutputExpectation) -> np.ndarray:
    times = netcdf_times(path)
    wanted_first = np.datetime64(expected.first.astimezone(timezone.utc).replace(tzinfo=None), "ns")
    wanted_last = np.datetime64(expected.last.astimezone(timezone.utc).replace(tzinfo=None), "ns")
    if len(times) != expected.rows or times[0] != wanted_first or times[-1] != wanted_last:
        raise ValueError(f"{path} axis is rows={len(times)}, first={times[0]}, last={times[-1]}; expected rows={expected.rows}, first={wanted_first}, last={wanted_last}")
    return times
