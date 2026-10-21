"""AEM3D forecast segment execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import re
import subprocess
import shutil
import time
import numpy as np
from typing import Callable, Mapping
from uuid import uuid4

from .validation import OutputExpectation, restart_time, validate_output

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentRequest:
    issue_time: datetime
    name: str
    run_root: Path
    expected_restart_time: datetime
    expected_outputs: Mapping[str, OutputExpectation]
    restart_input: Path | None = None
    restart_run_path: Path | None = None
    attempt_id: str = ""
    start_time: datetime | None = None
    iterations: int | None = None


@dataclass(frozen=True)
class SegmentResult:
    request: SegmentRequest
    nc_dir: Path
    unf_dir: Path
    txt_dir: Path
    log_file: Path
    restart_file: Path
    simulation_started: datetime | None = None
    simulation_finished: datetime | None = None


RunFileWriter = Callable[[SegmentRequest, Path, Path, Path], Path]
ProcessExecutor = Callable[[Path, Path], int]


def make_run_file_writer(template: Path) -> RunFileWriter:
    """Create an isolated run-file writer from a validated deployment template."""
    template = Path(template)
    if not template.is_file() or template.stat().st_size == 0:
        raise FileNotFoundError(f"run-file template is missing or empty: {template}")

    def writer(request: SegmentRequest, nc_dir: Path, unf_dir: Path, txt_dir: Path) -> Path:
        if request.start_time is None or request.iterations is None or request.iterations < 1:
            raise ValueError("production segment requires start_time and positive iterations")
        lines = template.read_text().splitlines(keepends=True)
        values: dict[str, str] = {
            "start_date_cwr": _cwr_time(request.start_time),
            "del_t": "60.0",
            "iter_max": str(request.iterations),
            "iheat_input": "1",
            "irain": "1",
            "iflow": "1",
            "itemperature": "1",
            "idensity": "1",
            "time_zone": "0.0",
            "irestart": "1" if request.restart_input else "0",
            "iter_out_save": "60",
            "iter_out_restart": str(request.iterations),
            "restart_save_file": "save",
            "restart_out_file": "restart_out",
            "outfile_nc_dir": repr(nc_dir.as_posix()),
            "outfile_unf_dir": repr(unf_dir.as_posix()),
            "outfile_txt_dir": repr(txt_dir.as_posix()),
        }
        for label, value in values.items():
            pattern = re.compile(rf"^(?!\s*!).*\s{re.escape(label)}\s*$", re.I)
            matches = [index for index, line in enumerate(lines) if pattern.match(line.rstrip("\r\n"))]
            if len(matches) != 1:
                raise ValueError(f"run-file template must contain exactly one {label!r} value")
            newline = "\r\n" if lines[matches[0]].endswith("\r\n") else "\n"
            lines[matches[0]] = f"  {value} {label}{newline}"
        restart_pattern = re.compile(r"^(?!\s*!).*\srestart_in_file\s*$", re.I)
        lines[:] = [line for line in lines if not restart_pattern.match(line.rstrip("\r\n"))]
        if request.restart_input:
            run_path = request.restart_run_path or request.restart_input
            restart_line = f"  {Path(run_path).as_posix()} restart_in_file\n"
            restart_index = next(index for index, line in enumerate(lines) if re.search(r"\sirestart\s*$", line, re.I) and not line.lstrip().startswith("!"))
            lines.insert(restart_index + 1, restart_line)
        rain_pattern = re.compile(
            r"^(?!\s*!).*\bMET_Rain\.dat\s+boundary_condition_file\s*$",
            re.I,
        )
        rain_matches = [
            index
            for index, line in enumerate(lines)
            if rain_pattern.match(line.rstrip("\r\n"))
        ]
        if len(rain_matches) > 1:
            raise ValueError("run-file template contains duplicate MET_Rain.dat entries")
        if not rain_matches:
            met_index = next(
                index
                for index, line in enumerate(lines)
                if re.match(
                    r"^(?!\s*!).*\bMET_AirTemp\.dat\s+boundary_condition_file\s*$",
                    line.rstrip("\r\n"),
                    re.I,
                )
            )
            lines.insert(
                met_index,
                "  MET_Rain.dat                          boundary_condition_file\n",
            )
        target = nc_dir.parent / "run_aem3d.dat"
        target.write_text("".join(lines), newline="")
        return target
    return writer


def _cwr_time(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("start_time must be timezone-aware")
    utc = value.astimezone(timezone.utc)
    fraction = (utc.hour + utc.minute / 60 + utc.second / 3600) / 24
    return f"{utc.year * 1000 + utc.timetuple().tm_yday + fraction:.6f}"


DEFAULT_PROCESS_TIMEOUT_SECONDS = 6 * 3600
DEFAULT_HEARTBEAT_SECONDS = 120
STEP_LINE = re.compile(r"Step:\s+(\d+)/(\d+)\s+SimDate:\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")


def _hms(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def _latest_step(log_file: Path) -> tuple[int, int, str] | None:
    """Return the newest ``Step: k/n SimDate: ...`` line of a running AEM3D log."""
    try:
        with Path(log_file).open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(handle.tell() - 4096, 0))
            tail = handle.read().decode(errors="replace")
    except OSError:
        return None
    matches = STEP_LINE.findall(tail)
    if not matches:
        return None
    step, total, model_time = matches[-1]
    return int(step), int(total), model_time


def _wait_with_heartbeat(
    process,
    log_file: Path,
    *,
    timeout_seconds: float | None,
    heartbeat_seconds: float,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Wait for ``process`` and log its AEM3D step progress every ``heartbeat_seconds``.

    Raises ``subprocess.TimeoutExpired`` once ``timeout_seconds`` have elapsed; the
    caller kills the process.
    """
    started = clock()
    while True:
        elapsed = clock() - started
        if timeout_seconds is not None and elapsed >= timeout_seconds:
            raise subprocess.TimeoutExpired(process.args, timeout_seconds)
        wait = heartbeat_seconds if timeout_seconds is None else min(heartbeat_seconds, timeout_seconds - elapsed)
        try:
            return process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            elapsed = clock() - started
            progress = _latest_step(log_file)
            if progress is None:
                LOGGER.info("AEM3D running for %s, no step reported yet", _hms(elapsed), extra={"kind": "heartbeat"})
                continue
            step, total, model_time = progress
            remaining = f", about {_hms(elapsed * (total - step) / step)} left" if step else ""
            LOGGER.info("AEM3D running at step %d of %d, model time %s, %s elapsed%s", step, total, model_time, _hms(elapsed), remaining, extra={"kind": "heartbeat"})


def make_process_executor(
    executable: Path,
    working_directory: Path,
    *,
    timeout_seconds: float | None = DEFAULT_PROCESS_TIMEOUT_SECONDS,
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
) -> ProcessExecutor:
    """Create the bounded subprocess adapter; stdout and stderr remain in the issue log.

    The process is killed and ``subprocess.TimeoutExpired`` is raised when it runs
    longer than ``timeout_seconds``. While it runs, its step progress is logged every
    ``heartbeat_seconds``. The runtime directory is owned by the single issue
    coordinator, so a run file left there by a killed run is replaced.
    """
    executable, working_directory = Path(executable), Path(working_directory)
    if not executable.is_file():
        raise FileNotFoundError(f"AEM3D executable does not exist: {executable}")
    def execute(run_file: Path, log_file: Path) -> int:
        installed = working_directory / "run_aem3d.dat"
        installed.unlink(missing_ok=True)
        shutil.copy2(run_file, installed)
        try:
            with log_file.open("w") as log:
                with subprocess.Popen([str(executable)], cwd=working_directory, stdout=log, stderr=subprocess.STDOUT) as process:
                    try:
                        returncode = _wait_with_heartbeat(process, log_file, timeout_seconds=timeout_seconds, heartbeat_seconds=heartbeat_seconds)
                    except BaseException:
                        process.kill()
                        raise
            for name in ("aem3d_echo.txt", "aem3d_debug.txt", "aem3d_debug_timestep.txt", "outfile.txt", "scratch.txt"):
                source = working_directory / name
                if source.is_file():
                    shutil.copy2(source, log_file.parent / name)
            return returncode
        finally:
            installed.unlink(missing_ok=True)
    return execute


class BoundedAem3dRunner:
    def __init__(self, run_file_writer: RunFileWriter, process_executor: ProcessExecutor):
        self._write_run_file = run_file_writer
        self._execute = process_executor

    def run(self, request: SegmentRequest) -> SegmentResult:
        if request.issue_time.tzinfo is None or request.expected_restart_time.tzinfo is None:
            raise ValueError("segment timestamps must be timezone-aware")
        if request.name not in {"handoff", "product"}:
            raise ValueError("segment name must be 'handoff' or 'product'")
        if request.restart_input is not None:
            if request.start_time is None:
                raise ValueError("restart segment requires an explicit start_time")
            expected_input = request.start_time.astimezone(timezone.utc) - timedelta(minutes=1)
            actual_input = restart_time(Path(request.restart_input))
            if actual_input != expected_input:
                raise ValueError(f"restart input is {actual_input.isoformat()}; segment start requires {expected_input.isoformat()}")
        issue_id = request.issue_time.astimezone(timezone.utc).strftime("issue_%Y%m%dT%H%MZ")
        attempt = request.attempt_id or f"attempt_{uuid4().hex}"
        root = Path(request.run_root) / issue_id / attempt / request.name
        if root.exists():
            raise FileExistsError(f"segment output directory already exists: {root}")
        nc_dir, unf_dir, txt_dir = root / "nc", root / "unf", root / "txt"
        for directory in (nc_dir, unf_dir, txt_dir):
            directory.mkdir(parents=True)
        run_file = self._write_run_file(request, nc_dir, unf_dir, txt_dir)
        log_file = root / "aem3d.log"
        source = f"restarting from restart file {Path(request.restart_input).name}" if request.restart_input is not None else "starting from a cold start"
        LOGGER.info(
            "AEM3D %s segment %s with %s iterations, simulating up to model time %s UTC",
            request.name, source, request.iterations, request.expected_restart_time.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            extra={"kind": "segment"},
        )
        simulation_started = datetime.now(timezone.utc)
        code = self._execute(run_file, log_file)
        simulation_finished = datetime.now(timezone.utc)
        if code != 0:
            raise RuntimeError(f"AEM3D exited {code}; evidence retained at {root}")
        if not log_file.is_file() or "FINISHED AEM3D RUN" not in log_file.read_text(errors="replace"):
            raise RuntimeError(f"AEM3D completion marker is missing: {log_file}")
        restart = unf_dir / "restart_out.unf"
        actual = restart_time(restart)
        expected = request.expected_restart_time.astimezone(timezone.utc)
        if actual != expected:
            raise ValueError(f"restart model time is {actual.isoformat()}; expected {expected.isoformat()}")
        LOGGER.info(
            "AEM3D %s segment finished in %s, restart reached model time %s UTC as expected",
            request.name, _hms((simulation_finished - simulation_started).total_seconds()), actual.strftime("%Y-%m-%d %H:%M"),
            extra={"kind": "segment"},
        )
        for name, expectation in request.expected_outputs.items():
            times = validate_output(nc_dir / name, expectation)
            LOGGER.info("AEM3D %s segment output %s validated, %d rows from %s to %s", request.name, name, len(times), str(times[0])[:16].replace("T", " "), str(times[-1])[:16].replace("T", " "), extra={"kind": "detail"})
        return SegmentResult(
            request,
            nc_dir,
            unf_dir,
            txt_dir,
            log_file,
            restart,
            simulation_started,
            simulation_finished,
        )


def validate_segment_overlap(handoff: SegmentResult, product: SegmentResult) -> None:
    """Validate an exact overlap or the next declared output-sampling interval."""
    common = set(handoff.request.expected_outputs).intersection(product.request.expected_outputs)
    if not common:
        raise ValueError("segments have no common required outputs")
    from .validation import netcdf_times
    for name in common:
        first = netcdf_times(handoff.nc_dir / name)
        second = netcdf_times(product.nc_dir / name)
        gap = second[0] - first[-1]
        if gap < np.timedelta64(0, "ns"):
            raise ValueError(f"{name} product output starts before the handoff output ends")
        intervals = []
        if len(first) > 1: intervals.append(first[-1] - first[-2])
        if len(second) > 1: intervals.append(second[1] - second[0])
        if gap and (not intervals or gap > max(intervals)):
            raise ValueError(f"{name} segment boundary gap exceeds its output sampling interval")
