"""Immutable per-issue forecast products and their lightweight catalog."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import gc
import json
import logging
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

import numpy as np
import xarray as xr
from netCDF4 import Dataset as NetcdfDataset

from .grib_layout import (
    _as_grib_archive_issue,
    _merge_segments,
    _write_grib_archive_candidate,
)


LOGGER = logging.getLogger(__name__)

ARCHIVE_OUTPUTS: dict[str, bool] = {
    "LDS.nc": True,
    "Sarnico.nc": True,
    "Curt_est.nc": False,
    "Curt_long.nc": False,
    "top0.nc": True,
    "top01.nc": True,
}

ISSUE_INDEX_FILENAME = "issue_index.json"
ISSUE_INDEX_SCHEMA_VERSION = 1
_TIMESTAMP_FIELDS = (
    "issue_time",
    "forcing_available_time",
    "simulation_started",
    "simulation_finished",
    "products_ready_time",
    "first_valid_time",
    "last_valid_time",
)


@dataclass(frozen=True)
class ArchivedIssue:
    """Receipt for committed immutable issue files and their catalog entries."""

    issue_time: datetime
    archive_files: tuple[Path, ...]
    removed_sources: tuple[Path, ...]
    index_file: Path | None = None
    records: tuple[IssueIndexRecord, ...] = ()


@dataclass(frozen=True)
class IssueIndexRecord:
    """One station/product entry in the lightweight issue catalog."""

    station: str
    product: str
    issue_time: datetime
    forcing_available_time: datetime | None
    simulation_started: datetime | None
    simulation_finished: datetime | None
    products_ready_time: datetime
    first_valid_time: datetime
    last_valid_time: datetime
    path: Path
    status: str = "complete"


def enabled_archive_outputs(selection: Mapping[str, bool] | None = None) -> tuple[str, ...]:
    """Return enabled output filenames in configuration order."""

    configured = ARCHIVE_OUTPUTS if selection is None else selection
    return tuple(name for name, enabled in configured.items() if enabled)


def _as_utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _optional_utc(value: datetime | None, label: str) -> datetime | None:
    return None if value is None else _as_utc(value, label)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _as_utc(value, "timestamp").isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object, label: str, *, optional: bool = False) -> datetime | None:
    if value in (None, "") and optional:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO timestamp") from exc
    return _as_utc(parsed, label)


def _station_name(output_name: str) -> str:
    station = Path(output_name).stem.upper()
    if not station or any(not (character.isalnum() or character == "_") for character in station):
        raise ValueError(f"output name cannot be used as an archive station: {output_name}")
    return station


def _target_path(archive_dir: Path, output_name: str, issue_time: datetime) -> Path:
    station = _station_name(output_name)
    stamp = _as_utc(issue_time, "issue_time").strftime("%Y%m%dT%H%MZ")
    return archive_dir / station / f"{station}_{stamp}.nc"


def _relative_path(path: Path, archive_dir: Path) -> Path:
    try:
        return path.resolve().relative_to(archive_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"issue path is outside the archive: {path}") from exc


def _record_payload(record: IssueIndexRecord) -> dict[str, object]:
    payload = asdict(record)
    for name in _TIMESTAMP_FIELDS:
        payload[name] = _iso(payload[name])
    payload["path"] = record.path.as_posix()
    return payload


def _record_from_payload(payload: object, archive_dir: Path) -> IssueIndexRecord:
    if not isinstance(payload, dict):
        raise ValueError("issue index record must be an object")
    missing = {
        "station",
        "product",
        "issue_time",
        "forcing_available_time",
        "simulation_started",
        "simulation_finished",
        "products_ready_time",
        "first_valid_time",
        "last_valid_time",
        "path",
        "status",
    }.difference(payload)
    if missing:
        raise ValueError("issue index record is missing: " + ", ".join(sorted(missing)))
    relative = Path(str(payload["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"issue index path must be archive-relative: {relative}")
    absolute = archive_dir / relative
    _relative_path(absolute, archive_dir)
    record = IssueIndexRecord(
        station=str(payload["station"]),
        product=str(payload["product"]),
        issue_time=_parse_timestamp(payload["issue_time"], "issue_time"),
        forcing_available_time=_parse_timestamp(
            payload["forcing_available_time"], "forcing_available_time", optional=True
        ),
        simulation_started=_parse_timestamp(
            payload["simulation_started"], "simulation_started", optional=True
        ),
        simulation_finished=_parse_timestamp(
            payload["simulation_finished"], "simulation_finished", optional=True
        ),
        products_ready_time=_parse_timestamp(
            payload["products_ready_time"], "products_ready_time"
        ),
        first_valid_time=_parse_timestamp(payload["first_valid_time"], "first_valid_time"),
        last_valid_time=_parse_timestamp(payload["last_valid_time"], "last_valid_time"),
        path=relative,
        status=str(payload["status"]),
    )
    _validate_record(record)
    return record


def _validate_record(record: IssueIndexRecord) -> None:
    issue = _as_utc(record.issue_time, "issue_time")
    _optional_utc(record.forcing_available_time, "forcing_available_time")
    started = _optional_utc(record.simulation_started, "simulation_started")
    finished = _optional_utc(record.simulation_finished, "simulation_finished")
    ready = _as_utc(record.products_ready_time, "products_ready_time")
    first = _as_utc(record.first_valid_time, "first_valid_time")
    last = _as_utc(record.last_valid_time, "last_valid_time")
    if record.status != "complete":
        raise ValueError("committed issue index records must have status 'complete'")
    expected_station = _station_name(record.product)
    expected_path = Path(expected_station) / (
        f"{expected_station}_{issue.strftime('%Y%m%dT%H%MZ')}.nc"
    )
    if record.station != expected_station or record.path != expected_path:
        raise ValueError("issue index station, product, and path do not agree")
    if first > last:
        raise ValueError("first_valid_time must not be after last_valid_time")
    if started is not None and finished is not None and started > finished:
        raise ValueError("simulation_started must not be after simulation_finished")
    if finished is not None and finished > ready:
        raise ValueError("simulation_finished must not be after products_ready_time")


def read_issue_index(
    archive_dir: Path,
    *,
    require_files: bool = True,
) -> tuple[IssueIndexRecord, ...]:
    """Load and validate the small catalog without opening forecast NetCDF files."""

    archive_dir = Path(archive_dir)
    index_path = archive_dir / ISSUE_INDEX_FILENAME
    if not index_path.exists():
        return ()
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid issue index at {index_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != ISSUE_INDEX_SCHEMA_VERSION:
        raise ValueError(f"unsupported issue index schema at {index_path}")
    records_payload = payload.get("records")
    if not isinstance(records_payload, list):
        raise ValueError(f"issue index records must be a list: {index_path}")
    records = tuple(_record_from_payload(item, archive_dir) for item in records_payload)
    keys = [(record.station, record.product, record.issue_time) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError(f"issue index contains duplicate product issue rows: {index_path}")
    if require_files:
        missing = [record.path for record in records if not (archive_dir / record.path).is_file()]
        if missing:
            raise ValueError(
                "issue index references missing files: "
                + ", ".join(path.as_posix() for path in missing)
            )
    return records


def _write_issue_index(archive_dir: Path, records: Sequence[IssueIndexRecord]) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    index_path = archive_dir / ISSUE_INDEX_FILENAME
    temporary = archive_dir / f".{ISSUE_INDEX_FILENAME}.{uuid4().hex}.tmp"
    ordered = sorted(records, key=lambda item: (item.issue_time, item.station, item.product))
    payload = {
        "schema_version": ISSUE_INDEX_SCHEMA_VERSION,
        "records": [_record_payload(record) for record in ordered],
    }
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, index_path)
    finally:
        temporary.unlink(missing_ok=True)
    return index_path


def find_issue_records(
    archive_dir: Path,
    *,
    station: str | None = None,
    valid_time: datetime | None = None,
    available_at: datetime | None = None,
) -> tuple[IssueIndexRecord, ...]:
    """Return relevant issues newest-first for dashboard overlap resolution."""

    requested_station = station.upper() if station is not None else None
    valid = _optional_utc(valid_time, "valid_time")
    available = _optional_utc(available_at, "available_at")
    records = []
    for record in read_issue_index(archive_dir):
        if requested_station is not None and record.station != requested_station:
            continue
        if valid is not None and not (record.first_valid_time <= valid <= record.last_valid_time):
            continue
        if available is not None and record.products_ready_time > available:
            continue
        records.append(record)
    return tuple(sorted(records, key=lambda item: item.issue_time, reverse=True))


def _metadata_attrs(record: IssueIndexRecord) -> dict[str, str]:
    return {
        "issue_time": _iso(record.issue_time) or "",
        "forcing_available_time": _iso(record.forcing_available_time) or "",
        "simulation_started": _iso(record.simulation_started) or "",
        "simulation_finished": _iso(record.simulation_finished) or "",
        "products_ready_time": _iso(record.products_ready_time) or "",
        "first_valid_time": _iso(record.first_valid_time) or "",
        "last_valid_time": _iso(record.last_valid_time) or "",
        "status": record.status,
        "station": record.station,
        "product": record.product,
        "archive_path": record.path.as_posix(),
        "archive_layout": "immutable-per-issue-v1",
    }


def _stamp_metadata(path: Path, record: IssueIndexRecord) -> None:
    with NetcdfDataset(path, "a") as dataset:
        for name, value in _metadata_attrs(record).items():
            dataset.setncattr(name, value)


def _record_from_issue_file(path: Path, archive_dir: Path) -> IssueIndexRecord:
    with NetcdfDataset(path) as dataset:
        attrs = {name: getattr(dataset, name, None) for name in (
            "station",
            "product",
            "issue_time",
            "forcing_available_time",
            "simulation_started",
            "simulation_finished",
            "products_ready_time",
            "first_valid_time",
            "last_valid_time",
            "status",
        )}
    attrs["path"] = _relative_path(path, archive_dir).as_posix()
    return _record_from_payload(attrs, archive_dir)


def rebuild_issue_index(archive_dir: Path) -> Path:
    """Rebuild the disposable catalog from authoritative per-issue files."""

    archive_dir = Path(archive_dir)
    records = [
        _record_from_issue_file(path, archive_dir)
        for path in sorted(archive_dir.glob("*/*.nc"))
    ]
    return _write_issue_index(archive_dir, records)


def _scientifically_equal(first: Path, second: Path) -> bool:
    with xr.open_dataset(
        first, decode_times=False, decode_coords=False, mask_and_scale=False
    ) as first_opened, xr.open_dataset(
        second, decode_times=False, decode_coords=False, mask_and_scale=False
    ) as second_opened:
        return first_opened.equals(second_opened)


def _verify_issue_file(path: Path, record: IssueIndexRecord) -> None:
    with NetcdfDataset(path) as dataset:
        if len(dataset.dimensions["time"]) != 1:
            raise RuntimeError(f"per-issue file must contain exactly one issue: {path}")
        for name, expected in _metadata_attrs(record).items():
            if getattr(dataset, name, None) != expected:
                raise RuntimeError(f"per-issue metadata is invalid: {path}:{name}")


def _update_index(archive_dir: Path, records: Sequence[IssueIndexRecord]) -> Path:
    existing = list(read_issue_index(archive_dir))
    replacements = {
        (record.station, record.product, record.issue_time): record for record in records
    }
    retained = [
        record
        for record in existing
        if (record.station, record.product, record.issue_time) not in replacements
    ]
    return _write_issue_index(archive_dir, (*retained, *records))


def _remove_committed_source(path: Path, *, attempts: int = 3) -> bool:
    """Remove a committed source without making cleanup a commit gate."""
    for attempt in range(attempts):
        try:
            path.unlink()
            return True
        except PermissionError:
            if attempt + 1 == attempts:
                LOGGER.warning("could not remove %s (permission denied); the file is left in place", path)
                return False
            gc.collect()
            time.sleep(0.1 * (attempt + 1))
    return False


def archive_issue_outputs(
    issue_time: datetime,
    segment_dirs: Sequence[Path],
    archive_dir: Path,
    *,
    output_selection: Mapping[str, bool] | None = None,
    remove_sources: bool = True,
    forcing_available_time: datetime | None = None,
    simulation_started: datetime | None = None,
    simulation_finished: datetime | None = None,
    products_ready_time: datetime | None = None,
    repair: bool = False,
) -> ArchivedIssue:
    """Commit one immutable file per enabled product for one forecast issue.

    Existing issue files are reused only when their scientific datasets equal
    the newly validated candidates. Differing content requires ``repair=True``.
    The JSON index is replaced atomically after every issue file is durable; it
    can be rebuilt from the metadata embedded in those authoritative files.
    """

    issue = _as_utc(issue_time, "issue_time")
    if issue.hour not in {0, 12} or issue.minute or issue.second or issue.microsecond:
        raise ValueError("issue_time must be a 00:00 or 12:00 UTC issue")
    directories = tuple(Path(path) for path in segment_dirs)
    if not directories:
        raise ValueError("At least one AEM3D segment directory is required")
    configured = dict(ARCHIVE_OUTPUTS if output_selection is None else output_selection)
    enabled = enabled_archive_outputs(configured)
    if not enabled:
        raise ValueError("At least one archive output must be enabled")
    sources = {
        name: tuple(directory / name for directory in directories) for name in enabled
    }
    for paths in sources.values():
        for path in paths:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Required AEM3D output is missing or empty: {path}")

    forcing_available = _optional_utc(forcing_available_time, "forcing_available_time")
    started = _optional_utc(simulation_started, "simulation_started")
    finished = _optional_utc(simulation_finished, "simulation_finished")
    if started is not None and finished is not None and started > finished:
        raise ValueError("simulation_started must not be after simulation_finished")
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    targets = {name: _target_path(archive_dir, name, issue) for name in enabled}
    for target in targets.values():
        target.parent.mkdir(parents=True, exist_ok=True)

    staging_root = archive_dir.parent / "archive_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".issue-archive-", dir=staging_root) as temporary:
        staging = Path(temporary)
        candidates: dict[str, Path] = {}
        valid_ranges: dict[str, tuple[str, str]] = {}
        for name in enabled:
            merged, valid_times = _merge_segments(sources[name])
            try:
                issue_dataset = _as_grib_archive_issue(merged, valid_times, issue, name)
                try:
                    candidate = staging / targets[name].name
                    _write_grib_archive_candidate(issue_dataset, candidate)
                    candidates[name] = candidate
                    valid_ranges[name] = (
                        np.datetime_as_string(valid_times[0], unit="us"),
                        np.datetime_as_string(valid_times[-1], unit="us"),
                    )
                finally:
                    issue_dataset.close()
            finally:
                merged.close()

        ready = _optional_utc(products_ready_time, "products_ready_time") or datetime.now(timezone.utc)
        if finished is not None and finished > ready:
            raise ValueError("simulation_finished must not be after products_ready_time")
        records: dict[str, IssueIndexRecord] = {}
        for name in enabled:
            first_text, last_text = valid_ranges[name]
            record = IssueIndexRecord(
                station=_station_name(name),
                product=name,
                issue_time=issue,
                forcing_available_time=forcing_available,
                simulation_started=started,
                simulation_finished=finished,
                products_ready_time=ready,
                first_valid_time=datetime.fromisoformat(first_text).replace(tzinfo=timezone.utc),
                last_valid_time=datetime.fromisoformat(last_text).replace(tzinfo=timezone.utc),
                path=_relative_path(targets[name], archive_dir),
            )
            _validate_record(record)
            _stamp_metadata(candidates[name], record)
            _verify_issue_file(candidates[name], record)
            records[name] = record

        for name in enabled:
            target = targets[name]
            if not target.exists():
                continue
            if repair:
                continue
            try:
                existing_record = _record_from_issue_file(target, archive_dir)
                _verify_issue_file(target, existing_record)
            except Exception as exc:
                raise FileExistsError(
                    f"existing issue file is invalid; explicit repair is required: {target}"
                ) from exc
            if not _scientifically_equal(target, candidates[name]):
                raise FileExistsError(
                    f"immutable issue file already exists with different scientific data: {target}"
                )
            records[name] = existing_record

        rollback = staging / "rollback"
        rollback.mkdir()
        committed: list[str] = []
        previous: dict[str, Path | None] = {}
        try:
            for name in enabled:
                target = targets[name]
                if target.exists() and not repair:
                    continue
                backup = rollback / target.name
                if target.exists():
                    shutil.copy2(target, backup)
                    previous[name] = backup
                else:
                    previous[name] = None
                candidates[name].replace(target)
                committed.append(name)
            for name in enabled:
                _verify_issue_file(targets[name], records[name])
        except Exception:
            for name in reversed(committed):
                target = targets[name]
                backup = previous[name]
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    backup.replace(target)
            raise

        committed_records = tuple(records[name] for name in enabled)
        index_file = _update_index(archive_dir, committed_records)
        LOGGER.info(
            "issue %s archived (%s) and catalog updated",
            f"{issue:%Y-%m-%dT%H:%MZ}",
            ", ".join(f"{name} {'written' if name in committed else 'reused (identical to the committed file)'}" for name in enabled),
            extra={"kind": "housekeeping"},
        )

    removed: list[Path] = []
    if remove_sources:
        for name in configured:
            for directory in directories:
                source = directory / name
                if source.is_file() and _remove_committed_source(source):
                    removed.append(source)
        for directory in directories:
            try:
                directory.rmdir()
            except OSError:
                pass
        LOGGER.info("removed %d archived segment output files", len(removed), extra={"kind": "housekeeping"})

    return ArchivedIssue(
        issue_time=issue,
        archive_files=tuple(targets[name] for name in enabled),
        removed_sources=tuple(removed),
        index_file=index_file,
        records=committed_records,
    )


TRANSIENT_TOP_PRODUCTS = frozenset({"top0.nc", "top01.nc"})


def prune_transient_top_products(
    archive_dir: Path,
    *,
    current_issue: datetime,
    permanently_retained_issues: Sequence[datetime] = (),
) -> tuple[Path, ...]:
    """Keep LDS/Sarnico forever; prune top products outside the retention window.

    Permanently retained issues plus the current and preceding issue keep their
    top products. All other top0/top01 issue files and catalog rows are removed.
    """

    archive_dir = Path(archive_dir)
    current = _as_utc(current_issue, "current_issue")
    keep = {
        current,
        current - timedelta(hours=12),
        *(_as_utc(value, "permanently_retained_issues") for value in permanently_retained_issues),
    }
    records = list(read_issue_index(archive_dir))
    retained: list[IssueIndexRecord] = []
    pruned: list[Path] = []
    for record in records:
        if record.product not in TRANSIENT_TOP_PRODUCTS or record.issue_time in keep:
            retained.append(record)
        else:
            pruned.append(archive_dir / record.path)
    if not pruned:
        return ()

    _write_issue_index(archive_dir, retained)
    removed = tuple(path for path in pruned if path.is_file() and _remove_committed_source(path))
    LOGGER.info("top products outside the retention window pruned (%s)", ", ".join(sorted(path.name for path in removed)) or "none", extra={"kind": "housekeeping"})
    if len(removed) != len(pruned):
        LOGGER.warning(
            "%d pruned top product files were not removed and are no longer in the catalog (%s)",
            len(pruned) - len(removed),
            ", ".join(sorted(path.name for path in pruned if path not in removed)),
        )
    return removed
