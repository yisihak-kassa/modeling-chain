"""Concrete production composition for isolated bounded AEM3D issues."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import shutil
import time
from typing import Callable, Mapping

from .chain import IssueRun
from .preparation import PreparedIssue
from .runner import BoundedAem3dRunner, SegmentRequest
from .validation import OutputExpectation


LOGGER = logging.getLogger(__name__)
STATIC_INPUTS = ("INIT_Temperature.dat", "INIT_Tracer_1.dat", "UPDATE_Tracer_1.dat", "datablock_new_base.xml", "sparsedata.unf", "usedata.unf")
AEM3D_EXECUTABLE = "aem3d.exe"


def stage_isolated_runtime(source: Path, target: Path) -> Path:
    """Copy only the executable runtime and static model inputs into a new directory."""
    source, target = Path(source), Path(target)
    if target.exists():
        raise FileExistsError(f"isolated runtime already exists: {target}")
    target.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and (path.suffix.lower() in {".exe", ".dll"} or path.name in {"layer_tops.txt", "surf_map.txt"}):
            shutil.copy2(path, target / path.name)
    infiles = target / "infiles"
    infiles.mkdir()
    for name in STATIC_INPUTS:
        source_file = source / "infiles" / name
        if source_file.is_file():
            shutil.copy2(source_file, infiles / name)
    required = (
        infiles / "datablock_new_base.xml",
        infiles / "sparsedata.unf",
        infiles / "usedata.unf",
    )
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("isolated runtime is incomplete: " + ", ".join(missing))
    return target


@dataclass(frozen=True)
class WorkflowConfiguration:
    run_root: Path
    runtime_dir: Path
    product_hours: int = 16
    output_names: tuple[str, ...] = ("LDS.nc", "Sarnico.nc", "top0.nc", "top01.nc")


Preparation = Callable[[datetime], PreparedIssue]


class ProductionIssueWorker:
    """Prepare forcing and execute handoff then product in an isolated runtime."""
    def __init__(self, prepare: Preparation, runner: BoundedAem3dRunner, configuration: WorkflowConfiguration):
        self.prepare, self.runner, self.configuration = prepare, runner, configuration

    def __call__(self, issue: datetime, current_handoff: Path | None) -> IssueRun:
        preparation_started = time.monotonic()
        LOGGER.info("preparing forcing for issue %s", f"{issue:%Y-%m-%dT%H:%MZ}", extra={"kind": "phase"})
        prepared = self.prepare(issue)
        infiles = self.configuration.runtime_dir / "infiles"
        for path in prepared.files:
            shutil.copy2(path, infiles / path.name)
        attempt = prepared.output_dir.name
        LOGGER.info(
            "forcing prepared for issue %s in %.0f s from %d input files, available since %s, attempt %s",
            f"{issue:%Y-%m-%dT%H:%MZ}", time.monotonic() - preparation_started, len(prepared.files),
            prepared.forcing_available_time, attempt,
            extra={"kind": "phase"},
        )
        boundary = issue + timedelta(hours=12)
        current_run_path = (
            self._stage_restart(current_handoff, issue, attempt, "current")
            if current_handoff
            else None
        )
        handoff = self.runner.run(SegmentRequest(
            issue, "handoff", self.configuration.run_root, boundary - timedelta(minutes=1),
            self._expectations(issue, boundary, cold=current_handoff is None),
            restart_input=current_handoff, restart_run_path=current_run_path,
            attempt_id=attempt, start_time=issue,
            iterations=720 if current_handoff is None else 721,
        ))
        shutil.copy2(handoff.restart_file.with_suffix(".txt"), handoff.log_file.parent / "validated_restart.txt")
        end = issue + timedelta(hours=self.configuration.product_hours)
        handoff_run_path = self._stage_restart(
            handoff.restart_file, issue, attempt, "handoff"
        )
        product = self.runner.run(SegmentRequest(
            issue, "product", self.configuration.run_root, end - timedelta(minutes=1),
            self._expectations(boundary, end, cold=False, product=True),
            restart_input=handoff.restart_file, restart_run_path=handoff_run_path,
            attempt_id=attempt, start_time=boundary,
            iterations=int((end - boundary).total_seconds() // 60) + 1,
        ))
        shutil.copy2(product.restart_file.with_suffix(".txt"), product.log_file.parent / "validated_restart.txt")
        self._discard_restart_copies(product.restart_file, (infiles / handoff_run_path).parent)
        return IssueRun(
            handoff,
            product,
            boundary - timedelta(minutes=1),
            forcing_available_time=prepared.forcing_available_time,
            simulation_started=handoff.simulation_started,
            simulation_finished=product.simulation_finished,
        )

    @staticmethod
    def _discard_restart_copies(product_restart: Path, staging_dir: Path) -> None:
        """Remove restart binaries that are not chain state once both segments validate.

        The chain handoff store holds the only retained restarts. The product
        segment's end-of-run restart and the restarts staged for AEM3D are
        dropped. Cleanup never fails a validated issue.
        """
        with suppress(OSError):
            product_restart.unlink(missing_ok=True)
        shutil.rmtree(staging_dir, ignore_errors=True)
        leftovers = [str(path) for path in (product_restart, staging_dir) if path.exists()]
        if leftovers:
            LOGGER.warning("restart cleanup incomplete, could not remove: %s", ", ".join(leftovers))
        else:
            LOGGER.info("restart cleanup removed the product segment restart and the staged restart copies", extra={"kind": "housekeeping"})

    def _stage_restart(
        self, source: Path, issue: datetime, attempt: str, role: str
    ) -> Path:
        relative = (
            Path("restart_attempts")
            / f"{issue:%Y%m%dT%H%MZ}_{attempt}"
            / f"{role}.unf"
        )
        target = self.configuration.runtime_dir / "infiles" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        shutil.copy2(source.with_suffix(".txt"), target.with_suffix(".txt"))
        return relative

    def _expectations(self, start: datetime, end: datetime, *, cold: bool, product: bool = False) -> Mapping[str, OutputExpectation]:
        offset = timedelta(minutes=59 if cold else 58)
        result = {}
        for name in self.configuration.output_names:
            cadence = timedelta(hours=4 if name.startswith("top") else 1)
            first = start + cadence - timedelta(minutes=1 if cold else 2)
            rows = int((end - first) // cadence) + 1
            last = first + cadence * (rows - 1)
            result[name] = OutputExpectation(rows, first, last)
        return result
