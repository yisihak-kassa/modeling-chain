"""Command-line entry point for the production Lake Iseo forecast chain."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import time
from typing import Callable
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xarray as xr

from icon2i.api.auth import get_headers, login
from icon2i.api.config import DOWNLOAD_DIR
from icon2i.live_acquisition import (
    acquire_forecast_issue,
    delete_published_issue_requests,
    published_issue_files,
)

from .chain import (
    ChainStateStore,
    bootstrap_chain,
    process_available_issues,
    process_expected_issue,
    seed_chain_state,
)
from .console_log import ConsoleFormatter, supports_color
from .darfo_update import update_q_csv
from .discharge_forecast import Chronos2DischargeProvider, require_measured_discharge
from .flow_forcing import FlowConfiguration
from .issue import IssueInputs, REQUIRED_GRIB_VARIABLES
from .preparation import prepare_production_issue
from .runner import BoundedAem3dRunner, make_process_executor, make_run_file_writer
from .validation import restart_time
from .workflow import (
    AEM3D_EXECUTABLE,
    ProductionIssueWorker,
    WorkflowConfiguration,
    stage_isolated_runtime,
)

UTC = timezone.utc
EUROPE_ROME = ZoneInfo("Europe/Rome")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOGGER = logging.getLogger("lake_iseo_forecast.cli")


def _configure_logging(workspace: Path) -> None:
    """Send package progress to the terminal (stderr) and to ``<workspace>/chain.log``."""
    workspace.mkdir(parents=True, exist_ok=True)
    package = logging.getLogger("lake_iseo_forecast")
    package.setLevel(logging.INFO)
    for handler in list(package.handlers):
        if getattr(handler, "_lake_iseo_cli", False):
            package.removeHandler(handler)
            handler.close()
    file_formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(ConsoleFormatter(supports_color(console.stream)))
    log_file = logging.FileHandler(workspace / "chain.log", encoding="utf-8")
    log_file.setFormatter(file_formatter)
    for handler in (console, log_file):
        handler._lake_iseo_cli = True
        package.addHandler(handler)


def _label(issue: datetime) -> str:
    return f"{issue.astimezone(UTC):%Y-%m-%dT%H:%MZ}"


def _hms(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


class IssueProgress:
    """Log run progress around the chain loop.

    ``process_available_issues`` calls the availability check once before every issue,
    so each call marks the commit of the previous issue and the start of the next one;
    ``finish`` or ``failed`` closes the run.
    """

    def __init__(
        self,
        is_available: Callable[[datetime], bool],
        max_issues: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        free_gb: Callable[[], float | None] = lambda: None,
    ):
        self._is_available, self.max_issues = is_available, max_issues
        self._clock, self._free_gb = clock, free_gb
        self._started = self._committed = 0
        self._durations: list[float] = []
        self._current: datetime | None = None
        self._current_since = 0.0
        self._run_started: float | None = None
        self._stop_reason: str | None = None

    def _commit_current(self, now: float) -> None:
        if self._current is None:
            return
        duration = now - self._current_since
        self._committed += 1
        self._durations.append(duration)
        free = self._free_gb()
        LOGGER.info(
            "committed issue %s in %s (%d of %d done%s)",
            _label(self._current), _hms(duration), self._committed, self.max_issues,
            "" if free is None else f", {free:.1f} GB free",
            extra={"kind": "success"},
        )
        self._current = None

    def __call__(self, issue: datetime) -> bool:
        now = self._clock()
        if self._run_started is None:
            self._run_started = now
        self._commit_current(now)
        LOGGER.info("checking GRIB availability for issue %s", _label(issue), extra={"kind": "phase"})
        if not self._is_available(issue):
            self._stop_reason = f"issue {_label(issue)} not available in the GRIB inputs"
            LOGGER.warning(
                "issue %s is not available in the GRIB inputs; stopping after %d of %d issues",
                _label(issue), self._committed, self.max_issues,
            )
            return False
        self._started += 1
        self._current, self._current_since = issue, now
        eta = ""
        if self._durations:
            estimate = sum(self._durations) / len(self._durations) * (self.max_issues - self._committed)
            finish = datetime.now().astimezone() + timedelta(seconds=estimate)
            eta = f", ETA {_hms(estimate)} (about {finish:%Y-%m-%d %H:%M} local time)"
        LOGGER.info("[%d/%d] starting issue %s%s", self._started, self.max_issues, _label(issue), eta, extra={"kind": "banner"})
        return True

    def finish(self) -> str:
        now = self._clock()
        self._commit_current(now)
        reason = self._stop_reason or "max-issues reached"
        LOGGER.info(
            "run finished with %d of %d issues committed in %s, stopped because %s",
            self._committed, self.max_issues, _hms(now - (self._run_started if self._run_started is not None else now)), reason,
            extra={"kind": "success"},
        )
        return reason

    def failed(self, error: BaseException) -> None:
        now = self._clock()
        if self._current is not None:
            where = f"issue {_label(self._current)} after {_hms(now - self._current_since)}"
            note = "the chain state was not advanced for it"
        else:
            where = "while checking the next issue"
            note = "the chain state is unchanged"
        LOGGER.error(
            "FAILED %s: %s: %s; %d issues committed this run; %s; rerun with --max-issues %d to resume",
            where, type(error).__name__, error, self._committed, note, self.max_issues - self._committed,
        )


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("issue time must include a timezone")
    return parsed.astimezone(UTC)


def discover_grib_files(downloads: Path, issue: datetime) -> dict[str, Path]:
    """Select separate per-variable GRIB archives containing one issue.

    Production downloads may either use the current ``variable__.grib`` names
    or historical ``variable_YYYYMMDD_YYYYMMDD.grib`` bundles. Dated archives
    that contain the requested issue win over a co-located ``variable__.grib``
    so stale live files cannot shadow the intended historical bundle.
    Overlapping dated archives are ambiguous; combined wind archives never
    match the exact per-variable pattern.
    """
    downloads = Path(downloads)
    issue_date = issue.astimezone(UTC).date()
    cycle_tag = issue.astimezone(UTC).strftime("%Y%m%dT%H%MZ")
    cycle_dir = downloads / "issues" / f"issue_{cycle_tag}"
    if cycle_dir.exists():
        selected = {
            variable: cycle_dir / f"{variable}_{cycle_tag}.grib"
            for variable in REQUIRED_GRIB_VARIABLES
        }
        missing = [
            str(path)
            for path in selected.values()
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            raise FileNotFoundError(
                "published cycle-specific GRIB batch is incomplete: "
                + ", ".join(missing)
            )
        return selected
    selected: dict[str, Path] = {}
    for variable in REQUIRED_GRIB_VARIABLES:
        pattern = re.compile(
            rf"^{re.escape(variable)}_(\d{{8}})_(\d{{8}})\.grib$"
        )
        matches = []
        for candidate in downloads.glob(f"{variable}_*.grib"):
            match = pattern.fullmatch(candidate.name)
            if not match or not candidate.stat().st_size:
                continue
            first = datetime.strptime(match.group(1), "%Y%m%d").date()
            last = datetime.strptime(match.group(2), "%Y%m%d").date()
            if first > last:
                raise ValueError(f"GRIB archive has an inverted date range: {candidate}")
            if first <= issue_date <= last:
                matches.append(candidate)
        if len(matches) > 1:
            names = ", ".join(sorted(path.name for path in matches))
            raise ValueError(
                f"multiple {variable} GRIB archives contain {issue_date}: {names}"
            )
        if len(matches) == 1:
            selected[variable] = matches[0]
            continue

        current = downloads / f"{variable}__.grib"
        if current.is_file() and current.stat().st_size:
            selected[variable] = current
            continue
        raise FileNotFoundError(
            f"no {variable} GRIB archive in {downloads} contains {issue_date}"
        )
    return selected


def available_issue_times(grib_files: dict[str, Path]) -> frozenset[datetime]:
    """Return the common 00Z/12Z reference times in all configured GRIB files."""
    common: set[np.datetime64] | None = None
    for name, path in grib_files.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"GRIB input is missing or empty: {path}")
        with xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as dataset:
            if "time" not in dataset.coords:
                raise ValueError(f"{path} has no forecast reference-time coordinate")
            values = set(np.atleast_1d(dataset.time.values).astype("datetime64[s]"))
        common = values if common is None else common.intersection(values)
    result = set()
    for value in common or ():
        timestamp = pd.Timestamp(value).to_pydatetime().replace(tzinfo=UTC)
        if timestamp.hour in {0, 12} and not timestamp.minute and not timestamp.second:
            result.add(timestamp)
    return frozenset(result)


def resolve_grib_source(
    downloads: Path,
    issue: datetime,
    *,
    fallback: str,
    forecast_hours: int,
) -> tuple[dict[str, Path], datetime]:
    """Resolve the exact cycle or an explicitly selected preceding cycle."""
    candidates = [issue]
    if fallback == "previous-cycle":
        candidates.append(issue - timedelta(hours=12))
    errors = []
    for reference in candidates:
        try:
            files = discover_grib_files(downloads, reference)
            if reference not in available_issue_times(files):
                raise ValueError(f"reference cycle {_label(reference)} is absent")
            lead_end = int((issue - reference).total_seconds() // 3600) + forecast_hours
            if lead_end > 72:
                raise ValueError(
                    f"reference cycle {_label(reference)} ends at lead 72, "
                    f"but issue {_label(issue)} requires lead {lead_end}"
                )
            if reference != issue:
                LOGGER.warning(
                    "weather fallback: issue %s uses preceding reference cycle %s",
                    _label(issue),
                    _label(reference),
                )
            return files, reference
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{_label(reference)}: {exc}")
    raise FileNotFoundError("; ".join(errors))


def load_discharge_csv(path: Path) -> pd.Series:
    """Load ARPA discharge recorded in Italian civil time onto a UTC-naive index."""
    discharge = pd.read_csv(path, parse_dates=["tempo"], index_col="tempo")["Q"].sort_index()
    local_index = pd.DatetimeIndex(discharge.index)
    if local_index.tz is None:
        local_index = local_index.tz_localize(
            EUROPE_ROME,



            ambiguous=False,
            nonexistent="shift_forward",
        )
    discharge.index = local_index.tz_convert(UTC).tz_localize(None)
    if discharge.index.has_duplicates:
        discharge = discharge.groupby(level=0).mean()
    return discharge.sort_index()


def _build_worker(args) -> tuple[ProductionIssueWorker, Callable[[datetime], bool]]:
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = workspace / "runtime"
    if not runtime.exists():
        stage_isolated_runtime(args.aem3d_dir.resolve(), runtime)
    downloads = args.downloads.resolve()
    discharge = load_discharge_csv(args.discharge_csv)
    LOGGER.info(
        "Darfo measurements run up to %s UTC", discharge.index.max(),
    )

    if args.discharge_policy == "measured-only":
        def load_discharge(start: datetime, end: datetime) -> pd.Series:
            return require_measured_discharge(discharge, start, end)
    else:
        load_discharge = Chronos2DischargeProvider(
            discharge,
            workspace / "discharge_forecasts",
            model_id=args.discharge_model,
            context_hours=args.discharge_context_hours,
            maximum_age_hours=args.discharge_max_age_hours,
        )
        LOGGER.info(
            "Chronos-2 will forecast discharge past the latest measurement, using %d hours of context and accepting measurements up to %.1f hours old",
            args.discharge_context_hours,
            args.discharge_max_age_hours,
        )

    flow = FlowConfiguration(.328, .672, 27.0, .5, .5, 208, 209, 198, 199, 40, 45.7, 10.05)

    def prepare(issue: datetime):
        gribs, reference_time = resolve_grib_source(
            downloads,
            issue,
            fallback=args.weather_fallback,
            forecast_hours=args.forecast_hours,
        )
        attempt = f"issue_{issue:%Y%m%dT%H%MZ}_attempt_{uuid4().hex}"
        request = IssueInputs(issue, gribs, args.zone_mapping.resolve(), workspace / "forcing" / attempt)
        return prepare_production_issue(
            request,
            discharge_loader=load_discharge,
            flow_configuration=flow,
            forecast_hours=args.forecast_hours,
            grib_reference_time=reference_time,
        )

    runner = BoundedAem3dRunner(
        make_run_file_writer(args.aem3d_dir.resolve() / "run_aem3d.dat"),
        make_process_executor(runtime / AEM3D_EXECUTABLE, runtime),
    )
    worker = ProductionIssueWorker(
        prepare,
        runner,
        WorkflowConfiguration(workspace / "runs", runtime, product_hours=args.forecast_hours),
    )

    def is_available(issue: datetime) -> bool:
        try:
            resolve_grib_source(
                downloads,
                issue,
                fallback=args.weather_fallback,
                forecast_hours=args.forecast_hours,
            )
            return True
        except (FileNotFoundError, ValueError):
            return False

    return worker, is_available


def _log_run_plan(args, workspace: Path, store: ChainStateStore, state_path: Path, checkpoints: Path) -> None:
    """State what the run will use before the first issue starts."""
    state = store.load()
    following = state.next_expected_issue
    LOGGER.info("workspace %s", workspace)
    LOGGER.info(
        "chain state file %s%s",
        state_path, " (preferred over chain-state.json because it exists)" if state_path.name == "chain-state-continued.json" else "",
    )
    LOGGER.info("handoff store %s, archive %s", checkpoints, state.archive_dir)
    LOGGER.info("last completed issue %s, next expected issue %s", _label(state.last_completed_issue), _label(following))
    LOGGER.info(
        "active handoff %s at model time %s",
        state.active_handoff.name, restart_time(state.active_handoff).strftime("%Y-%m-%d %H:%M:%S"),
    )
    last_planned = following + timedelta(hours=12 * (args.max_issues - 1))
    LOGGER.info(
        "plan covers up to %d issues from %s to %s with %d-hour forecasts, %.1f GB free",
        args.max_issues, _label(following), _label(last_planned), args.forecast_hours,
        shutil.disk_usage(workspace).free / 1e9,
    )
    try:
        gribs = discover_grib_files(args.downloads.resolve(), following)
        later = sorted(time_ for time_ in available_issue_times(gribs) if time_ >= following)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.warning("GRIB inputs for the next issue could not be read (%s)", exc)
        return
    LOGGER.info(
        "GRIB archives for the next issue are %s, holding %d reference times from the next issue on, the last being %s",
        ", ".join(sorted({path.name for path in gribs.values()})), len(later), _label(later[-1]) if later else "none",
    )


def _state_payload(store: ChainStateStore) -> dict[str, str]:
    state = store.load()
    return {
        "last_completed_issue": state.last_completed_issue.isoformat(),
        "next_expected_issue": state.next_expected_issue.isoformat(),
        "active_handoff": str(state.active_handoff),
        "handoff_model_time": state.handoff_model_time.isoformat(),
        "archive_dir": str(state.archive_dir),
    }


def _load_permanently_retain_top_issues(workspace: Path) -> tuple[datetime, ...]:
    path = Path(workspace) / "retention.json"
    if not path.is_file():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("permanently_retain_top_issues", [])
    return tuple(_datetime(value) for value in values)


def _acquire_cli_issue(args, issue: datetime) -> dict[str, Path]:
    """Reuse a validated local issue or authenticate and acquire it from MeteoHub."""
    downloads = args.downloads.resolve()
    existing = published_issue_files(
        downloads,
        issue,
        forecast_hours=args.forecast_hours,
    )
    if existing is not None:
        LOGGER.info("reusing validated MeteoHub issue %s", _label(issue))
        return existing
    return acquire_forecast_issue(
        issue,
        get_headers(login()),
        download_dir=downloads,
        forecast_hours=args.forecast_hours,
        retry_failed=args.retry_failed,
    )


def _refresh_live_discharge(args, issue: datetime) -> None:
    """Refresh measured Darfo data through the live issue cutoff."""
    local_end = issue.astimezone(EUROPE_ROME).replace(tzinfo=None)
    last, added = update_q_csv(args.discharge_csv, local_end, PROJECT_ROOT)
    LOGGER.info(
        "Darfo data refreshed up to the issue time %s UTC, %d new rows, newest measurement %s UTC",
        issue.astimezone(UTC).strftime("%Y-%m-%d %H:%M"),
        added,
        pd.Timestamp(last).tz_localize(EUROPE_ROME, ambiguous=True).tz_convert("UTC").strftime("%Y-%m-%d %H:%M"),
    )


def _cleanup_remote_issue(args, issue: datetime) -> None:
    deleted = delete_published_issue_requests(
        issue,
        get_headers(login()),
        download_dir=args.downloads.resolve(),
        forecast_hours=args.forecast_hours,
    )
    LOGGER.info(
        "MeteoHub cleanup verified for %s, %d individual variable requests deleted",
        _label(issue),
        len(deleted),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lake-iseo-forecast")
    subparsers = parser.add_subparsers(dest="command", required=True)
    state = subparsers.add_parser("state", help="validate and print committed chain state")
    state.add_argument("--workspace", type=Path, required=True)
    seed = subparsers.add_parser(
        "seed-state",
        help="copy a validated handoff into a clean chain workspace",
    )
    seed.add_argument("--workspace", type=Path, required=True)
    seed.add_argument("--source-state", type=Path, required=True)
    fetch = subparsers.add_parser(
        "fetch",
        help="acquire and validate one MeteoHub issue without starting AEM3D",
    )
    fetch.add_argument("--workspace", type=Path, required=True)
    fetch.add_argument(
        "--issue",
        type=_datetime,
        help="exact issue to fetch; defaults to the chain's next expected issue",
    )
    fetch.add_argument("--downloads", type=Path, default=Path(DOWNLOAD_DIR))
    fetch.add_argument("--forecast-hours", type=int, default=72)
    fetch.add_argument(
        "--retry-failed",
        action="store_true",
        help="start a new attempt after retaining a failed attempt for diagnosis",
    )
    fetch.add_argument(
        "--cleanup-remote-requests",
        action="store_true",
        help="delete each provider variable request after local publication validates",
    )
    for name in ("bootstrap", "bootstrap-live", "run-one", "run-available", "run-live"):
        command = subparsers.add_parser(name)
        command.add_argument("--workspace", type=Path, required=True)
        command.add_argument("--downloads", type=Path, default=Path(DOWNLOAD_DIR))
        command.add_argument("--zone-mapping", type=Path, default=Path("config/output_fileA.npz"))
        command.add_argument("--discharge-csv", type=Path, required=True)
        command.add_argument("--aem3d-dir", type=Path, default=Path("aem3d"))
        command.add_argument("--forecast-hours", type=int, default=72)
        command.add_argument(
            "--weather-fallback",
            choices=("none", "previous-cycle"),
            default="none",
            help="fail on a missing cycle or explicitly use the preceding cycle's valid-time forecast",
        )
        command.add_argument(
            "--discharge-policy",
            choices=("measured-only", "chronos2"),
            default="measured-only",
            help="reject uncovered discharge or explicitly forecast it with Chronos-2",
        )
        command.add_argument("--discharge-model", default="amazon/chronos-2")
        command.add_argument("--discharge-context-hours", type=int, default=2048)
        command.add_argument("--discharge-max-age-hours", type=float, default=2.0)
    subparsers.choices["bootstrap"].add_argument("--issue", type=_datetime, required=True)
    subparsers.choices["bootstrap-live"].add_argument("--issue", type=_datetime, required=True)
    subparsers.choices["run-available"].add_argument("--max-issues", type=int, required=True)
    subparsers.choices["run-live"].add_argument(
        "--retry-failed",
        action="store_true",
        help="start a new acquisition attempt after retaining a failed attempt for diagnosis",
    )
    subparsers.choices["bootstrap-live"].add_argument(
        "--retry-failed",
        action="store_true",
        help="start a new acquisition attempt after retaining a failed attempt for diagnosis",
    )
    for name in ("bootstrap-live", "run-live"):
        subparsers.choices[name].add_argument(
            "--cleanup-remote-requests",
            action="store_true",
            help="delete each provider variable request after local publication validates",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace.resolve()
    continued_state = workspace / "chain-state-continued.json"
    state_path = (
        continued_state
        if continued_state.is_file()
        else workspace / "chain-state.json"
    )
    store = ChainStateStore(state_path)
    if args.command == "state":
        print(json.dumps(_state_payload(store), indent=2))
        return 0
    if args.command == "seed-state":
        source_path = args.source_state.resolve()
        source_store = ChainStateStore(source_path)
        source = source_store.load()
        seeded = seed_chain_state(
            source_store,
            store,
            workspace / "handoff",
            workspace / "archive",
        )
        receipt = {
            "schema_version": 1,
            "source_state": str(source_path),
            "source_handoff": str(source.active_handoff.resolve()),
            "source_handoff_sha256": _sha256(source.active_handoff),
            "target_handoff": str(seeded.active_handoff.resolve()),
            "target_handoff_sha256": _sha256(seeded.active_handoff),
            "handoff_model_time": seeded.handoff_model_time.astimezone(UTC).isoformat(),
            "next_expected_issue": seeded.next_expected_issue.astimezone(UTC).isoformat(),
        }
        if receipt["source_handoff_sha256"] != receipt["target_handoff_sha256"]:
            raise RuntimeError("seeded handoff checksum does not match its source")
        (workspace / "state-seed.json").write_text(
            json.dumps(receipt, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(_state_payload(store), indent=2))
        return 0
    if args.command == "fetch":
        if args.forecast_hours < 1:
            raise ValueError("forecast-hours must be positive")
        _configure_logging(workspace)
        issue = args.issue or store.load().next_expected_issue
        LOGGER.info("fetching MeteoHub issue %s without starting AEM3D", _label(issue))
        files = _acquire_cli_issue(args, issue)
        if args.cleanup_remote_requests:
            _cleanup_remote_issue(args, issue)
        print(json.dumps({name: str(path) for name, path in files.items()}, indent=2))
        return 0
    if args.forecast_hours < 16:
        raise ValueError("forecast-hours must be at least 16 to exercise all required outputs")
    _configure_logging(workspace)
    live_issue = None
    if args.command in {"bootstrap-live", "run-live"}:
        live_issue = (
            args.issue
            if args.command == "bootstrap-live"
            else store.load().next_expected_issue
        )
        LOGGER.info("acquiring next MeteoHub issue %s before forcing preparation", _label(live_issue))
        _acquire_cli_issue(args, live_issue)
        if args.cleanup_remote_requests:
            _cleanup_remote_issue(args, live_issue)
        _refresh_live_discharge(args, live_issue)
    worker, is_available = _build_worker(args)
    continued_checkpoints = workspace / "handoff_continued"
    checkpoints = (
        continued_checkpoints
        if continued_checkpoints.is_dir()
        else workspace / "handoff"
    )
    archive = workspace / "archive"
    permanent_tops = _load_permanently_retain_top_issues(workspace)
    if args.command in {"bootstrap", "bootstrap-live"}:
        if not is_available(args.issue):
            raise FileNotFoundError(f"issue is not complete in authoritative GRIB inputs: {args.issue.isoformat()}")
        bootstrap_chain(
            args.issue,
            store,
            worker,
            checkpoints,
            archive,
            permanently_retain_top_issues=permanent_tops,
        )
    elif args.command == "run-one":
        issue = store.load().next_expected_issue
        if not is_available(issue):
            raise FileNotFoundError(f"next issue is not complete in authoritative GRIB inputs: {issue.isoformat()}")
        process_expected_issue(
            issue,
            store,
            worker,
            checkpoints,
            permanently_retain_top_issues=permanent_tops,
        )
    elif args.command == "run-live":
        assert live_issue is not None
        if not is_available(live_issue):
            raise ValueError(
                f"acquired issue failed authoritative GRIB discovery: {live_issue.isoformat()}"
            )
        process_expected_issue(
            live_issue,
            store,
            worker,
            checkpoints,
            permanently_retain_top_issues=permanent_tops,
        )
    else:
        _log_run_plan(args, workspace, store, state_path, checkpoints)
        progress = IssueProgress(
            is_available,
            args.max_issues,
            free_gb=lambda: shutil.disk_usage(workspace).free / 1e9,
        )
        try:
            process_available_issues(
                store,
                progress,
                worker,
                checkpoints,
                max_issues=args.max_issues,
                permanently_retain_top_issues=permanent_tops,
            )
        except BaseException as exc:
            progress.failed(exc)
            raise
        progress.finish()
        final = store.load()
        LOGGER.info(
            "final state has last completed issue %s, next expected issue %s, handoff model time %s",
            _label(final.last_completed_issue), _label(final.next_expected_issue), final.handoff_model_time.astimezone(UTC).strftime("%Y-%m-%d %H:%M"),
        )
    print(json.dumps(_state_payload(store), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
