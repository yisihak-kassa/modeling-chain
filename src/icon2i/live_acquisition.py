"""Cycle-safe MeteoHub acquisition for one Lake Iseo forecast issue.

The production contract is deliberately narrow: request the seven GRIB inputs
used by AEM3D for one exact 00Z/12Z reference time, stage every response under
one attempt directory, validate the complete batch, and atomically publish it.
No AEM3D process is started here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
import requests as http
import xarray as xr

from icon2i.api.config import (
    BASE_URL,
    DATA_ENDPOINT,
    DATASETS,
    DEFAULT_CROP_BOUNDS,
    DELETE_ENDPOINT,
    DOWNLOAD_DIR,
    DOWNLOAD_ENDPOINT,
    ICON_POLL_INTERVAL_S,
    ICON_POLL_TIMEOUT_S,
    REQUESTS_ENDPOINT,
    get_group_variables,
)
from icon2i.api.request import build_data_request
from icon2i.utils.request_utils import load_registry, save_registry

logger = logging.getLogger(__name__)

REQUIRED_VARIABLES = tuple(get_group_variables("aem3d"))
_SHORT_NAMES = {
    "temperature": "t",
    "cloud_cover": "clct",
    "relative_humidity_1000hpa": "r",
    "wind_u": "u",
    "wind_v": "v",
    "shortwave_radiation": "avg_snswrf",
    "precipitation": "tp",
}
_ACCEPTED_UNITS = {
    "temperature": {"k", "degc", "c", "degree_celsius", "degrees_celsius"},
    "cloud_cover": {"%", "percent"},
    "relative_humidity_1000hpa": {"%", "percent"},
    "wind_u": {"m s-1", "m/s", "m s^-1"},
    "wind_v": {"m s-1", "m/s", "m s^-1"},
    "shortwave_radiation": {"w m-2", "w/m^2", "w m^-2"},
    "precipitation": {"kg m-2", "kg/m^2", "kg m^-2"},
}


def _issue_utc(issue_time: datetime) -> datetime:
    if issue_time.tzinfo is None:
        raise ValueError("issue_time must be timezone-aware")
    issue = issue_time.astimezone(timezone.utc)
    if issue.hour not in {0, 12} or issue.minute or issue.second or issue.microsecond:
        raise ValueError("issue_time must be exactly 00:00 or 12:00 UTC")
    return issue


def issue_tag(issue_time: datetime) -> str:
    return _issue_utc(issue_time).strftime("%Y%m%dT%H%MZ")


def issue_download_dir(download_dir: Path, issue_time: datetime) -> Path:
    return Path(download_dir) / "issues" / f"issue_{issue_tag(issue_time)}"


def issue_grib_name(variable: str, issue_time: datetime) -> str:
    if variable not in REQUIRED_VARIABLES:
        raise ValueError(f"unsupported production GRIB variable: {variable}")
    return f"{variable}_{issue_tag(issue_time)}.grib"


def forecast_dataset_config(run_time: datetime) -> dict:
    """Return an ICON-2I config narrowed to one exact model cycle."""
    issue = _issue_utc(run_time)
    dataset_config = copy.deepcopy(DATASETS["icon2i"])
    dataset_config["runs"] = [issue.hour * 60]
    return dataset_config


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:



            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_manifest(path: Path, manifest: dict) -> None:
    manifest["updated_at"] = _now()
    _atomic_json(path, manifest)


def _new_attempt(download_dir: Path, issue: datetime) -> tuple[Path, dict]:
    root = Path(download_dir) / ".staging" / f"issue_{issue_tag(issue)}"
    attempt_id = uuid4().hex
    attempt_dir = root / f"attempt_{attempt_id}"
    batch_dir = attempt_dir / "batch"
    batch_dir.mkdir(parents=True, exist_ok=False)
    variables = {}
    for variable in REQUIRED_VARIABLES:
        request_name = f"lake_iseo_{issue_tag(issue)}_{variable}_{attempt_id[:8]}"
        variables[variable] = {
            "request_name": request_name,
            "task_id": None,
            "status": "intent",
            "api_filename": None,
            "filesize_bytes": None,
            "local_file": str(batch_dir / issue_grib_name(variable, issue)),
            "downloaded": False,
        }
    manifest = {
        "schema_version": 1,
        "issue_time": issue.isoformat(),
        "attempt_id": attempt_id,
        "created_at": _now(),
        "updated_at": _now(),
        "status": "submitting",
        "variables": variables,
        "last_error": None,
    }
    manifest_path = attempt_dir / "attempt.json"
    _save_manifest(manifest_path, manifest)
    _atomic_json(root / "active.json", {"attempt": str(manifest_path.relative_to(root))})
    return manifest_path, manifest


def _load_attempt(
    download_dir: Path,
    issue: datetime,
    *,
    retry_failed: bool,
) -> tuple[Path, dict]:
    root = Path(download_dir) / ".staging" / f"issue_{issue_tag(issue)}"
    active_path = root / "active.json"
    if not active_path.is_file():
        return _new_attempt(download_dir, issue)
    manifest_path = root / _read_json(active_path)["attempt"]
    manifest = _read_json(manifest_path)
    if manifest.get("issue_time") != issue.isoformat():
        raise ValueError(f"active MeteoHub attempt has the wrong issue: {manifest_path}")
    if manifest.get("status") in {"failed", "validation_failed"}:
        if retry_failed:
            return _new_attempt(download_dir, issue)
        raise RuntimeError(
            f"MeteoHub attempt {manifest['attempt_id']} is {manifest['status']}; "
            "inspect its evidence or explicitly request a retry"
        )
    return manifest_path, manifest


def _api_requests(headers: dict, session=http) -> dict[str, dict]:
    response = session.get(BASE_URL + REQUESTS_ENDPOINT, headers=headers, timeout=30)
    response.raise_for_status()
    grouped: dict[str, list[dict]] = {}
    for item in response.json():
        grouped.setdefault(item.get("name", ""), []).append(item)
    duplicates = sorted(name for name, items in grouped.items() if name and len(items) != 1)
    if duplicates:
        raise RuntimeError("ambiguous MeteoHub request names: " + ", ".join(duplicates))
    return {name: items[0] for name, items in grouped.items() if name}


def _registry_entry(issue: datetime, attempt_id: str, variable: str, entry: dict) -> dict:
    return {
        "request_name": entry["request_name"],
        "task_id": entry.get("task_id") or "",
        "variables": [variable],
        "issue_time": issue.isoformat(),
        "attempt_id": attempt_id,
        "from": issue.isoformat(),
        "to": issue.isoformat(),
        "submitted_at": entry.get("submitted_at"),
        "status": entry.get("status", "pending"),
        "api_filename": entry.get("api_filename"),
        "filesize_bytes": entry.get("filesize_bytes"),
        "downloaded": bool(entry.get("downloaded")),
        "downloaded_at": entry.get("downloaded_at"),
        "local_file": entry.get("local_file"),
    }


def _flush_registry(download_dir: Path, issue: datetime, manifest: dict) -> None:
    registry_path = Path(download_dir) / "registry.json"
    existing = load_registry(registry_path)
    by_name = {entry["request_name"]: entry for entry in existing}
    for variable, entry in manifest["variables"].items():
        by_name[entry["request_name"]] = _registry_entry(
            issue, manifest["attempt_id"], variable, entry
        )
    save_registry(list(by_name.values()), registry_path)


def _submit_missing(
    download_dir: Path,
    issue: datetime,
    headers: dict,
    manifest_path: Path,
    manifest: dict,
    *,
    session=http,
    submit_delay_s: float = 1,
) -> None:
    dataset_config = forecast_dataset_config(issue)
    api_map = _api_requests(headers, session)
    for variable, entry in manifest["variables"].items():
        request_name = entry["request_name"]
        existing = api_map.get(request_name)
        if existing is not None:
            entry["task_id"] = existing.get("task_id") or entry.get("task_id")
            entry["status"] = "submitted"
            entry.setdefault("submitted_at", existing.get("submission_date") or _now())
            _save_manifest(manifest_path, manifest)
            continue
        if entry.get("task_id"):
            continue




        request = build_data_request(
            request_name=request_name,
            from_dt=issue,
            to_dt=issue,
            variable_names=[variable],
            dataset_config=dataset_config,
            crop_bounds=DEFAULT_CROP_BOUNDS,
        )
        response = session.post(
            BASE_URL + DATA_ENDPOINT,
            headers=headers,
            json=request,
            timeout=30,
        )
        response.raise_for_status()
        task_id = response.json().get("task_id")
        if not task_id:
            raise RuntimeError(f"MeteoHub returned no task_id for {request_name}")
        entry.update(task_id=task_id, status="submitted", submitted_at=_now())
        _save_manifest(manifest_path, manifest)
        _flush_registry(download_dir, issue, manifest)
        logger.info("submitted %s for %s", variable, issue.isoformat())
        if submit_delay_s:
            time.sleep(submit_delay_s)
    manifest["status"] = "polling"
    _save_manifest(manifest_path, manifest)
    _flush_registry(download_dir, issue, manifest)


def _download(entry: dict, headers: dict, *, session=http) -> None:
    api_filename = entry.get("api_filename")
    if not api_filename:
        raise RuntimeError(f"ready request has no output filename: {entry['request_name']}")
    destination = Path(entry["local_file"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    response = session.get(
        BASE_URL + DOWNLOAD_ENDPOINT.format(api_filename),
        headers=headers,
        stream=True,
        timeout=120,
    )
    response.raise_for_status()
    expected = int(response.headers.get("content-length", 0))
    written = 0
    with partial.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
                written += len(chunk)
    if not written:
        raise ValueError(f"MeteoHub download is empty: {entry['request_name']}")
    if expected and written != expected:
        raise ValueError(
            f"MeteoHub download size mismatch for {entry['request_name']}: "
            f"expected {expected}, got {written}"
        )
    os.replace(partial, destination)
    entry.update(downloaded=True, downloaded_at=_now(), downloaded_bytes=written)


def _poll_and_download(
    download_dir: Path,
    issue: datetime,
    headers: dict,
    manifest_path: Path,
    manifest: dict,
    *,
    session=http,
    poll_interval_s: float,
    poll_timeout_s: float,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> None:
    deadline = monotonic() + poll_timeout_s
    while True:
        try:
            api_map = _api_requests(headers, session)
        except Exception as exc:
            manifest["last_error"] = f"status poll failed: {type(exc).__name__}: {exc}"
            _save_manifest(manifest_path, manifest)
            if monotonic() >= deadline:
                manifest["status"] = "timed_out"
                _save_manifest(manifest_path, manifest)
                raise TimeoutError(manifest["last_error"]) from exc
            sleep(poll_interval_s)
            continue

        pending = []
        for variable, entry in manifest["variables"].items():
            local_file = Path(entry["local_file"])
            if entry.get("downloaded") and local_file.is_file() and local_file.stat().st_size:
                continue
            remote = api_map.get(entry["request_name"])
            if remote is None:
                pending.append(variable)
                continue
            raw_status = str(remote.get("status", ""))
            status = raw_status.upper()
            if status == "SUCCESS":
                entry.update(
                    status="ready",
                    api_filename=remote.get("fileoutput"),
                    filesize_bytes=remote.get("filesize"),
                )
                _download(entry, headers, session=session)
                entry["status"] = "downloaded"
                _save_manifest(manifest_path, manifest)
                _flush_registry(download_dir, issue, manifest)
            elif status in {"FAILED", "FAILURE", "ERROR"}:
                entry["status"] = "failed"
                manifest["status"] = "failed"
                manifest["last_error"] = f"MeteoHub request failed for {variable}: {raw_status}"
                _save_manifest(manifest_path, manifest)
                _flush_registry(download_dir, issue, manifest)
                raise RuntimeError(manifest["last_error"])
            elif status in {"PENDING", "PROCESSING", ""}:
                entry["status"] = "pending"
                pending.append(variable)
            else:
                entry["status"] = "ambiguous"
                manifest["status"] = "failed"
                manifest["last_error"] = (
                    f"unrecognised MeteoHub status for {variable}: {raw_status!r}"
                )
                _save_manifest(manifest_path, manifest)
                _flush_registry(download_dir, issue, manifest)
                raise RuntimeError(manifest["last_error"])

        _save_manifest(manifest_path, manifest)
        _flush_registry(download_dir, issue, manifest)
        if not pending:
            return
        if monotonic() >= deadline:
            manifest["status"] = "timed_out"
            manifest["last_error"] = "timed out waiting for: " + ", ".join(pending)
            _save_manifest(manifest_path, manifest)
            raise TimeoutError(manifest["last_error"])
        logger.info("waiting for MeteoHub variables: %s", ", ".join(pending))
        sleep(poll_interval_s)


def validate_forecast_grib(
    path: Path,
    variable: str,
    issue_time: datetime,
    *,
    forecast_hours: int = 72,
) -> None:
    """Validate one staged GRIB against the exact production issue contract."""
    path = Path(path)
    issue = _issue_utc(issue_time)
    if variable not in _SHORT_NAMES:
        raise ValueError(f"unsupported production GRIB variable: {variable}")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"GRIB is missing or empty: {path}")
    with xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as dataset:
        short_name = _SHORT_NAMES[variable]
        if short_name not in dataset:
            raise ValueError(f"{path} does not contain {variable} ({short_name})")
        if "time" not in dataset.coords:
            raise ValueError(f"{path} has no forecast reference-time coordinate")
        reference_times = pd.DatetimeIndex(np.atleast_1d(dataset.coords["time"].values))
        expected_time = pd.Timestamp(issue.replace(tzinfo=None))
        if len(reference_times) != 1 or reference_times[0] != expected_time:
            raise ValueError(
                f"{path} reference time is not exactly {expected_time}: {list(reference_times)}"
            )
        if "step" not in dataset.coords:
            raise ValueError(f"{path} has no forecast step coordinate")
        steps = np.atleast_1d(dataset.coords["step"].values)
        if not np.issubdtype(steps.dtype, np.timedelta64):
            raise ValueError(f"{path} forecast steps are not timedeltas")
        hours = np.asarray(steps / np.timedelta64(1, "h"), dtype=float)
        expected_hours = np.arange(forecast_hours + 1, dtype=float)
        if len(hours) < len(expected_hours) or not np.array_equal(
            hours[: len(expected_hours)], expected_hours
        ):
            raise ValueError(f"{path} does not contain every lead hour 0..{forecast_hours}")
        data = dataset[short_name]
        units = str(data.attrs.get("units", "")).strip().lower().replace("**", "^")
        if units not in _ACCEPTED_UNITS[variable]:
            raise ValueError(f"{path} has unsupported {variable} units: {units!r}")
        if not np.isfinite(data.values).all():
            raise ValueError(f"{path} contains missing or non-finite {variable} data")


def validate_forecast_batch(
    files: Mapping[str, Path],
    issue_time: datetime,
    *,
    forecast_hours: int = 72,
    validator: Callable[..., None] = validate_forecast_grib,
) -> None:
    missing = sorted(set(REQUIRED_VARIABLES).difference(files))
    extra = sorted(set(files).difference(REQUIRED_VARIABLES))
    if missing or extra:
        raise ValueError(f"invalid GRIB batch variables; missing={missing}, extra={extra}")
    for variable in REQUIRED_VARIABLES:
        validator(Path(files[variable]), variable, issue_time, forecast_hours=forecast_hours)


def published_issue_files(
    download_dir: Path,
    issue_time: datetime,
    *,
    forecast_hours: int = 72,
    validator: Callable[..., None] = validate_forecast_grib,
) -> dict[str, Path] | None:
    directory = issue_download_dir(download_dir, issue_time)
    if not directory.exists():
        return None
    if not directory.is_dir():
        raise ValueError(f"published issue path is not a directory: {directory}")
    files = {
        variable: directory / issue_grib_name(variable, issue_time)
        for variable in REQUIRED_VARIABLES
    }
    validate_forecast_batch(
        files, issue_time, forecast_hours=forecast_hours, validator=validator
    )
    return files


def delete_published_issue_requests(
    issue_time: datetime,
    headers: dict,
    *,
    download_dir: Path = Path(DOWNLOAD_DIR),
    forecast_hours: int = 72,
    session=http,
    validator: Callable[..., None] = validate_forecast_grib,
) -> tuple[str, ...]:
    """Delete each remote variable request after its local issue is verified."""
    issue = _issue_utc(issue_time)
    files = published_issue_files(
        download_dir,
        issue,
        forecast_hours=forecast_hours,
        validator=validator,
    )
    if files is None:
        raise FileNotFoundError(f"cannot clean an unpublished issue: {issue.isoformat()}")

    directory = issue_download_dir(download_dir, issue)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"published issue has no manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("issue_time") != issue.isoformat():
        raise ValueError(f"published manifest has the wrong issue: {manifest_path}")
    evidence = manifest.get("files", {})
    if set(evidence) != set(REQUIRED_VARIABLES):
        raise ValueError(f"published manifest has the wrong variables: {manifest_path}")

    requests = []
    for variable in REQUIRED_VARIABLES:
        path = files[variable]
        entry = evidence[variable]
        if entry.get("name") != path.name:
            raise ValueError(f"published manifest filename mismatch for {variable}")
        if entry.get("bytes") != path.stat().st_size:
            raise ValueError(f"published manifest size mismatch for {variable}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if entry.get("sha256") != digest:
            raise ValueError(f"published manifest checksum mismatch for {variable}")
        request_name = entry.get("request_name")
        if not request_name:
            raise ValueError(f"published manifest has no request name for {variable}")
        requests.append((variable, request_name, entry.get("task_id")))

    api_map = _api_requests(headers, session)
    deleted = []
    already_absent = []
    for variable, request_name, _task_id in requests:
        remote = api_map.get(request_name)
        if remote is None:
            already_absent.append(request_name)
            continue
        status = str(remote.get("status", "")).upper()
        if status != "SUCCESS":
            raise RuntimeError(
                f"refusing to delete {variable} request {request_name} with status {status!r}"
            )
        request_id = remote.get("id")
        if request_id is None:
            raise RuntimeError(f"MeteoHub returned no request id for {request_name}")
        response = session.delete(
            BASE_URL + DELETE_ENDPOINT.format(request_id),
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        deleted.append(request_name)

    remaining = _api_requests(headers, session)
    not_deleted = [name for _variable, name, _task_id in requests if name in remaining]
    if not_deleted:
        raise RuntimeError(
            "MeteoHub requests remain after cleanup: " + ", ".join(not_deleted)
        )
    _atomic_json(
        directory / "remote-cleanup.json",
        {
            "schema_version": 1,
            "issue_time": issue.isoformat(),
            "cleaned_at": _now(),
            "deleted_requests": deleted,
            "already_absent_requests": already_absent,
            "requests": [
                {"variable": variable, "request_name": name, "task_id": task_id}
                for variable, name, task_id in requests
            ],
        },
    )
    return tuple(deleted)


def _publish(
    download_dir: Path,
    issue: datetime,
    manifest_path: Path,
    manifest: dict,
    *,
    forecast_hours: int,
    validator: Callable[..., None],
) -> dict[str, Path]:
    files = {
        variable: Path(entry["local_file"])
        for variable, entry in manifest["variables"].items()
    }
    manifest["status"] = "validating"
    _save_manifest(manifest_path, manifest)
    try:
        validate_forecast_batch(
            files, issue, forecast_hours=forecast_hours, validator=validator
        )
    except Exception as exc:
        manifest["status"] = "validation_failed"
        manifest["last_error"] = f"{type(exc).__name__}: {exc}"
        _save_manifest(manifest_path, manifest)
        raise

    batch_dir = next(iter(files.values())).parent
    evidence = {
        "schema_version": 1,
        "issue_time": issue.isoformat(),
        "attempt_id": manifest["attempt_id"],
        "validated_at": _now(),
        "forecast_hours": forecast_hours,
        "files": {
            variable: {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "request_name": manifest["variables"][variable]["request_name"],
                "task_id": manifest["variables"][variable]["task_id"],
            }
            for variable, path in files.items()
        },
    }
    _atomic_json(batch_dir / "manifest.json", evidence)
    destination = issue_download_dir(download_dir, issue)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = published_issue_files(
            download_dir,
            issue,
            forecast_hours=forecast_hours,
            validator=validator,
        )
        assert existing is not None
        manifest["status"] = "reused_existing_publish"
        manifest["published_dir"] = str(destination)
        _save_manifest(manifest_path, manifest)
        return existing
    os.replace(batch_dir, destination)
    manifest["status"] = "published"
    manifest["published_dir"] = str(destination)
    _save_manifest(manifest_path, manifest)
    return {
        variable: destination / issue_grib_name(variable, issue)
        for variable in REQUIRED_VARIABLES
    }


def acquire_forecast_issue(
    issue_time: datetime,
    headers: dict,
    *,
    download_dir: Path = Path(DOWNLOAD_DIR),
    forecast_hours: int = 72,
    retry_failed: bool = False,
    session=http,
    poll_interval_s: float = ICON_POLL_INTERVAL_S,
    poll_timeout_s: float = ICON_POLL_TIMEOUT_S,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    validator: Callable[..., None] = validate_forecast_grib,
    submit_delay_s: float = 1,
) -> dict[str, Path]:
    """Acquire, validate, and atomically publish one exact ICON-2I cycle."""
    issue = _issue_utc(issue_time)
    published = published_issue_files(
        download_dir,
        issue,
        forecast_hours=forecast_hours,
        validator=validator,
    )
    if published is not None:
        logger.info("reusing validated MeteoHub issue %s", issue.isoformat())
        return published

    manifest_path, manifest = _load_attempt(download_dir, issue, retry_failed=retry_failed)
    try:
        _submit_missing(
            download_dir,
            issue,
            headers,
            manifest_path,
            manifest,
            session=session,
            submit_delay_s=submit_delay_s,
        )
        _poll_and_download(
            download_dir,
            issue,
            headers,
            manifest_path,
            manifest,
            session=session,
            poll_interval_s=poll_interval_s,
            poll_timeout_s=poll_timeout_s,
            sleep=sleep,
            monotonic=monotonic,
        )
        return _publish(
            download_dir,
            issue,
            manifest_path,
            manifest,
            forecast_hours=forecast_hours,
            validator=validator,
        )
    except Exception as exc:
        manifest["last_error"] = f"{type(exc).__name__}: {exc}"
        _save_manifest(manifest_path, manifest)
        raise
