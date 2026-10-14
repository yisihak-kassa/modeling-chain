"""Archive completed AEM3D forecast issues without losing issue identity."""

from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory



from .grib_layout import (
    _apply_grib_archive_metadata,
    _as_grib_archive_issue,
    _merge_segments,
    _numeric_grib_archive,
    _utc_naive,
    _write_grib_archive_candidate,
)
from .issue_archive import (
    ARCHIVE_OUTPUTS,
    ArchivedIssue,
    IssueIndexRecord,
    _remove_committed_source,
    archive_issue_outputs,
    enabled_archive_outputs,
)
from .validation import RESTART_TIME_RE as _RESTART_TIME_RE


@dataclass(frozen=True)
class HandoffCheckpoint:
    """Receipt for the one retained issue-to-issue restart checkpoint."""

    model_time: datetime
    restart_file: Path
    metadata_file: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _restart_time(metadata_file: Path) -> datetime:
    if not metadata_file.is_file() or metadata_file.stat().st_size == 0:
        raise FileNotFoundError(f"Restart metadata is missing or empty: {metadata_file}")
    match = _RESTART_TIME_RE.search(metadata_file.read_text(errors="replace"))
    if not match:
        raise ValueError(f"Could not parse sim_current from {metadata_file}")
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")


def _restore_replaced_file(previous: Path, target: Path) -> None:
    if previous.exists():
        previous.replace(target)
    else:
        target.unlink(missing_ok=True)


def _verify_handoff_files(
    restart: Path,
    metadata: Path,
    source_restart: Path,
    expected: datetime,
    *,
    stage: str,
) -> None:
    if _sha256(restart) != _sha256(source_restart):
        raise RuntimeError(f"{stage} handoff restart checksum does not match its source")
    if _restart_time(metadata) != expected:
        raise RuntimeError(f"{stage} handoff restart metadata has the wrong model time")


def promote_handoff_checkpoint(
    source_restart: Path,
    checkpoint_dir: Path,
    *,
    expected_model_time: datetime,
    remove_source: bool = True,
) -> HandoffCheckpoint:
    """Save one validated handoff while retaining the preceding checkpoint.

    The caller supplies the authoritative AEM3D model time because AEM3D's
    restart timestamp convention may represent the final completed timestep
    immediately before the nominal 12-hour issue boundary. Older handoffs
    beyond the current and preceding pair are removed only after promotion.
    """

    expected = _utc_naive(expected_model_time, "expected_model_time")
    source_restart = Path(source_restart)
    source_metadata = source_restart.with_suffix(".txt")
    if not source_restart.is_file() or source_restart.stat().st_size == 0:
        raise FileNotFoundError(f"Restart checkpoint is missing or empty: {source_restart}")
    actual = _restart_time(source_metadata)
    if actual != expected:
        raise ValueError(f"Restart model time is {actual}; expected {expected}")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"handoff_{expected.strftime('%Y%m%dT%H%M%SZ')}"
    target_restart = checkpoint_dir / f"{stem}.unf"
    target_metadata = checkpoint_dir / f"{stem}.txt"
    old_handoffs = tuple(
        path for path in checkpoint_dir.glob("handoff_*.unf") if path != target_restart
    )

    with TemporaryDirectory(prefix=".handoff-", dir=checkpoint_dir) as temporary:
        staging = Path(temporary)
        staged_restart = staging / target_restart.name
        staged_metadata = staging / target_metadata.name
        previous_restart = staging / "previous.unf"
        previous_metadata = staging / "previous.txt"
        shutil.copy2(source_restart, staged_restart)
        shutil.copy2(source_metadata, staged_metadata)
        _verify_handoff_files(
            staged_restart, staged_metadata, source_restart, expected, stage="Copied"
        )
        if target_restart.exists():
            shutil.copy2(target_restart, previous_restart)
        if target_metadata.exists():
            shutil.copy2(target_metadata, previous_metadata)
        try:
            staged_metadata.replace(target_metadata)
            staged_restart.replace(target_restart)
            _verify_handoff_files(
                target_restart, target_metadata, source_restart, expected, stage="Promoted"
            )
        except Exception:
            _restore_replaced_file(previous_metadata, target_metadata)
            _restore_replaced_file(previous_restart, target_restart)
            raise


    retained = {target_restart.resolve()}
    if old_handoffs:
        preceding = max(
            old_handoffs,
            key=lambda path: _restart_time(path.with_suffix(".txt")),
        )
        retained.add(preceding.resolve())
    dropped: list[str] = []
    for old_restart in old_handoffs:
        if old_restart.resolve() in retained:
            continue
        old_restart.unlink()
        old_restart.with_suffix(".txt").unlink(missing_ok=True)
        dropped.append(old_restart.name)

    if remove_source and source_restart.resolve() != target_restart.resolve():
        source_restart.unlink()
        source_metadata.unlink()

    logging.getLogger(__name__).info(
        "handoff %s promoted at model time %s, copied and verified, kept preceding %s, removed older %s, source restart %s",
        target_restart.name,
        expected.strftime("%Y-%m-%d %H:%M:%S"),
        ", ".join(sorted(path.name for path in old_handoffs if path.resolve() in retained)) or "none",
        ", ".join(dropped) or "none",
        "removed" if remove_source else "kept",
        extra={"kind": "housekeeping"},
    )
    return HandoffCheckpoint(
        model_time=expected_model_time.astimezone(timezone.utc),
        restart_file=target_restart,
        metadata_file=target_metadata,
    )
