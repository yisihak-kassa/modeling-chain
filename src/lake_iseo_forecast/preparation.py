from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Mapping

import numpy as np

from .flow_forcing import write_flow_files
from .issue import IssueInputs
from .met_forcing import extract_issue_grib, write_met_files
from .validation import parse_dat_axis


REQUIRED_MET_FILES = frozenset(
    {
        "MET_AirTemp.dat",
        "MET_CCF.dat",
        "MET_Rain.dat",
        "MET_RelHum.dat",
        "MET_SolRad.dat",
        "MET_WindDir.dat",
        "MET_WindSpd.dat",
    }
)
REQUIRED_FLOW_FILES = frozenset(
    {
        "FLOW_Offtake.dat",
        "FLOW_Oglio_River_x.dat",
        "FLOW_Oglio_River_y.dat",
        "FLOW_Industrial_Canal_x.dat",
        "FLOW_Industrial_Canal_y.dat",
    }
)
REQUIRED_AEM3D_FILES = REQUIRED_MET_FILES | REQUIRED_FLOW_FILES


@dataclass(frozen=True)
class PreparedIssue:
    """Published inputs and their issue time after successful preparation."""

    issue_time: datetime
    output_dir: Path
    files: tuple[Path, ...]
    forcing_available_time: datetime | None = None


Extractor = Callable[[Mapping[str, Path], Path, object], Path | None]
MetWriter = Callable[[Path, Path, Path], None]
FlowWriter = Callable[[object, Path, Path, Path], None]


def _missing_outputs(directory: Path) -> tuple[str, ...]:
    return tuple(sorted(name for name in REQUIRED_AEM3D_FILES if not (directory / name).is_file()))


def _share_axis(axes: list[np.ndarray]) -> bool:
    return all(np.array_equal(axes[0], axis) for axis in axes[1:])


def validate_prepared_inputs(directory: Path) -> tuple[str, ...]:
    """Return structural and numerical forcing errors without changing files."""

    directory = Path(directory)
    if not directory.is_dir():
        return (f"prepared input directory does not exist: {directory}",)
    missing = _missing_outputs(directory)
    errors = [f"missing AEM3D input: {name}" for name in missing]
    if missing:
        return tuple(errors)
    axes = {}
    for name in REQUIRED_AEM3D_FILES:
        try:
            axes[name] = parse_dat_axis(directory / name)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
    met_axes = [axes[name] for name in REQUIRED_MET_FILES if name in axes]
    flow_axes = [axes[name] for name in REQUIRED_FLOW_FILES if name in axes]
    if not _share_axis(met_axes):
        errors.append("MET files do not share a common forcing time axis")
    if not _share_axis(flow_axes):
        errors.append("FLOW files do not share a common forcing time axis")
    if met_axes and flow_axes and not np.array_equal(met_axes[0], flow_axes[0]):
        errors.append("MET and FLOW files do not share the complete horizon endpoint axis")
    return tuple(errors)


def prepare_issue(
    issue: IssueInputs,
    *,
    extractor: Extractor,
    met_writer: MetWriter,
    flow_writer: FlowWriter,
) -> PreparedIssue:
    """Create a complete AEM3D input set and publish it atomically.

    ``extractor`` writes NetCDF files into its supplied working directory and
    returns that directory (or ``None`` on failure). ``met_writer`` receives
    the NetCDF directory, zone mapping, and output directory. ``flow_writer``
    receives the issue, NetCDF directory, zone mapping, and output directory. Writers should
    raise on failure; partial output is discarded automatically.
    """

    issue.require_valid()
    forcing_available_time = datetime.fromtimestamp(
        max(Path(path).stat().st_mtime for path in issue.grib_files.values()),
        tz=timezone.utc,
    )
    output_dir = Path(issue.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output_dir must not exist before preparation: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temp_name:
        temp_dir = Path(temp_name)
        nc_dir = extractor(issue.grib_files, temp_dir / "nc", issue.issue_time_utc)
        if nc_dir is None:
            raise RuntimeError("GRIB extraction failed")
        nc_dir = Path(nc_dir)
        if not nc_dir.is_dir():
            raise RuntimeError(f"GRIB extractor returned no NetCDF directory: {nc_dir}")

        inputs_dir = temp_dir / "inputs"
        inputs_dir.mkdir()
        met_writer(nc_dir, issue.zone_mapping, inputs_dir)
        flow_writer(issue, nc_dir, issue.zone_mapping, inputs_dir)

        errors = validate_prepared_inputs(inputs_dir)
        if errors:
            raise RuntimeError("Incomplete AEM3D input set:\n- " + "\n- ".join(errors))

        output_dir.mkdir()
        for source in inputs_dir.iterdir():
            source.replace(output_dir / source.name)

    files = tuple(sorted(output_dir / name for name in REQUIRED_AEM3D_FILES))
    return PreparedIssue(
        issue.issue_time_utc,
        output_dir,
        files,
        forcing_available_time,
    )


def prepare_production_issue(
    issue: IssueInputs,
    *,
    discharge_loader,
    flow_configuration,
    forecast_hours: int = 72,
    grib_reference_time: datetime | None = None,
) -> PreparedIssue:
    def flow_writer(current_issue, nc_dir, _zone_mapping, output_dir):
        write_flow_files(
            Path(nc_dir) / "temperature.nc",
            output_dir,
            current_issue.issue_time_utc,
            forecast_hours=forecast_hours,
            discharge_loader=discharge_loader,
            configuration=flow_configuration,
        )
    return prepare_issue(
        issue,
        extractor=partial(
            extract_issue_grib,
            forecast_hours=forecast_hours,
            grib_reference_time=grib_reference_time,
        ),
        met_writer=write_met_files,
        flow_writer=flow_writer,
    )
