"""Atomic forecast-chain state and resumable issue coordination."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import time
from typing import Callable, Mapping, Sequence
from contextlib import contextmanager

from .archive import ArchivedIssue, HandoffCheckpoint, archive_issue_outputs, promote_handoff_checkpoint
from .issue_archive import prune_transient_top_products
from .runner import SegmentResult, validate_segment_overlap

UTC = timezone.utc


def _issue_time(value: datetime, label: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    result = value.astimezone(UTC)
    if result.minute or result.second or result.microsecond or result.hour not in {0, 12}:
        raise ValueError(f"{label} must be a 00:00 or 12:00 UTC issue")
    return result


@dataclass(frozen=True)
class ChainState:
    last_completed_issue: datetime
    next_expected_issue: datetime
    active_handoff: Path
    handoff_model_time: datetime
    archive_dir: Path

    def validate(self) -> None:
        last = _issue_time(self.last_completed_issue, "last_completed_issue")
        expected = _issue_time(self.next_expected_issue, "next_expected_issue")
        if expected - last != timedelta(hours=12):
            raise ValueError("next_expected_issue must be exactly 12 hours after last_completed_issue")
        if self.handoff_model_time.tzinfo is None:
            raise ValueError("handoff_model_time must be timezone-aware")
        from .validation import restart_time
        if restart_time(self.active_handoff) != self.handoff_model_time.astimezone(UTC):
            raise ValueError("active handoff metadata does not match chain state")
        if not self.archive_dir.is_dir():
            raise ValueError(f"archive directory does not exist: {self.archive_dir}")


class ChainStateStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> ChainState:
        try:
            payload = json.loads(self.path.read_text())
            state = ChainState(
                last_completed_issue=datetime.fromisoformat(payload["last_completed_issue"]),
                next_expected_issue=datetime.fromisoformat(payload["next_expected_issue"]),
                active_handoff=Path(payload["active_handoff"]),
                handoff_model_time=datetime.fromisoformat(payload["handoff_model_time"]),
                archive_dir=Path(payload["archive_dir"]),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid chain state at {self.path}: {exc}") from exc
        state.validate()
        return state

    def save(self, state: ChainState) -> None:
        state.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        payload = asdict(state)
        for key, value in tuple(payload.items()):
            if isinstance(value, datetime): payload[key] = value.astimezone(UTC).isoformat()
            elif isinstance(value, Path): payload[key] = str(value.resolve())
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, self.path)


def seed_chain_state(
    source_store: ChainStateStore,
    target_store: ChainStateStore,
    checkpoint_dir: Path,
    archive_dir: Path,
) -> ChainState:
    """Copy one validated operational handoff into a new chain workspace."""
    if target_store.path.exists():
        raise FileExistsError(f"target chain state already exists: {target_store.path}")
    checkpoint_dir = Path(checkpoint_dir)
    existing = tuple(checkpoint_dir.glob("handoff_*.unf")) if checkpoint_dir.is_dir() else ()
    if existing:
        raise FileExistsError(
            "target checkpoint directory already contains handoffs: "
            + ", ".join(str(path) for path in existing)
        )

    source = source_store.load()
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    promoted = promote_handoff_checkpoint(
        source.active_handoff,
        checkpoint_dir,
        expected_model_time=source.handoff_model_time,
        remove_source=False,
    )
    seeded = ChainState(
        source.last_completed_issue,
        source.next_expected_issue,
        promoted.restart_file,
        promoted.model_time,
        archive_dir,
    )
    target_store.save(seeded)
    return seeded


@dataclass(frozen=True)
class IssueRun:
    handoff: SegmentResult
    product: SegmentResult
    expected_handoff_time: datetime
    forcing_available_time: datetime | None = None
    simulation_started: datetime | None = None
    simulation_finished: datetime | None = None


IssueWorker = Callable[[datetime, Path | None], IssueRun]


def _apply_top_retention(
    archive_dir: Path,
    issue: datetime,
    permanently_retain_top_issues: Sequence[datetime] | None,
) -> None:
    if permanently_retain_top_issues is None:
        return
    prune_transient_top_products(
        archive_dir,
        current_issue=issue,
        permanently_retained_issues=permanently_retain_top_issues,
    )


def bootstrap_chain(
    issue_time: datetime,
    store: ChainStateStore,
    worker: IssueWorker,
    checkpoint_dir: Path,
    archive_dir: Path,
    *,
    output_selection: Mapping[str, bool] | None = None,
    permanently_retain_top_issues: Sequence[datetime] | None = None,
) -> ChainState:
    """Commit the first cold-start issue when no chain state exists yet."""
    if store.path.exists():
        raise FileExistsError(f"chain state already exists: {store.path}")
    issue = _issue_time(issue_time, "issue_time")
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    with _issue_lock(store):
        run = worker(issue, None)
        validate_segment_overlap(run.handoff, run.product)
        archive_issue_outputs(
            issue,
            (run.handoff.nc_dir, run.product.nc_dir),
            archive_dir,
            output_selection=output_selection,
            forcing_available_time=run.forcing_available_time,
            simulation_started=run.simulation_started,
            simulation_finished=run.simulation_finished,
        )
        _apply_top_retention(archive_dir, issue, permanently_retain_top_issues)
        promoted = promote_handoff_checkpoint(run.handoff.restart_file, checkpoint_dir, expected_model_time=run.expected_handoff_time)
        state = ChainState(issue, issue + timedelta(hours=12), promoted.restart_file, promoted.model_time, archive_dir)
        try:
            store.save(state)
        except Exception:


            raise
        return state


_LOCK_WRITE_GRACE_SECONDS = 10.0


def _pid_alive(pid: int) -> bool:
    """Probe a process without signalling it; os.kill(pid, 0) is not a probe on Windows."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except PermissionError:
            return True
        except OSError:
            return False
        return True
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return kernel32.GetLastError() == 5
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(exit_code)):
            return True
        return exit_code.value == 259
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def _lock_is_stale(lock: Path) -> bool:
    """A lock is stale when its owner is gone; a lock still being written is not."""
    try:
        text = lock.read_text().strip()
        if not text:
            return time.time() - lock.stat().st_mtime > _LOCK_WRITE_GRACE_SECONDS
        return not _pid_alive(int(text.removeprefix("pid=")))
    except (OSError, ValueError):
        return True


@contextmanager
def _issue_lock(store: ChainStateStore):
    lock = store.path.with_suffix(store.path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as exc:
            if _lock_is_stale(lock):
                lock.unlink(missing_ok=True)
                if attempt == 0:
                    continue
            raise RuntimeError(f"another forecast issue coordinator holds {lock}") from exc
    try:
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode())
        finally:
            os.close(descriptor)
        yield
    finally:
        lock.unlink(missing_ok=True)


def process_expected_issue(
    issue_time: datetime,
    store: ChainStateStore,
    worker: IssueWorker,
    checkpoint_dir: Path,
    *,
    output_selection: Mapping[str, bool] | None = None,
    permanently_retain_top_issues: Sequence[datetime] | None = None,
) -> ChainState:
    with _issue_lock(store):
        return _process_expected_issue_locked(
            issue_time,
            store,
            worker,
            checkpoint_dir,
            output_selection=output_selection,
            permanently_retain_top_issues=permanently_retain_top_issues,
        )


def _process_expected_issue_locked(
    issue_time: datetime,
    store: ChainStateStore,
    worker: IssueWorker,
    checkpoint_dir: Path,
    *,
    output_selection: Mapping[str, bool] | None = None,
    permanently_retain_top_issues: Sequence[datetime] | None = None,
) -> ChainState:
    state = store.load()
    issue = _issue_time(issue_time, "issue_time")
    if issue != state.next_expected_issue:
        raise ValueError(f"issue {issue.isoformat()} is not next expected issue {state.next_expected_issue.isoformat()}")
    expected_predecessor = issue - timedelta(minutes=1)
    if state.handoff_model_time.astimezone(UTC) != expected_predecessor:
        raise ValueError(
            f"active handoff is {state.handoff_model_time.isoformat()}; "
            f"issue {issue.isoformat()} requires {expected_predecessor.isoformat()}"
        )
    run = worker(issue, state.active_handoff)
    validate_segment_overlap(run.handoff, run.product)
    archived: ArchivedIssue = archive_issue_outputs(
        issue,
        (run.handoff.nc_dir, run.product.nc_dir),
        state.archive_dir,
        output_selection=output_selection,
        forcing_available_time=run.forcing_available_time,
        simulation_started=run.simulation_started,
        simulation_finished=run.simulation_finished,
    )
    _apply_top_retention(state.archive_dir, issue, permanently_retain_top_issues)
    rollback = Path(checkpoint_dir) / ".state-rollback"
    shutil.rmtree(rollback, ignore_errors=True)
    rollback.mkdir(parents=True)
    old_copy = rollback / state.active_handoff.name
    shutil.copy2(state.active_handoff, old_copy)
    shutil.copy2(state.active_handoff.with_suffix(".txt"), old_copy.with_suffix(".txt"))
    try:
        promoted: HandoffCheckpoint = promote_handoff_checkpoint(run.handoff.restart_file, checkpoint_dir, expected_model_time=run.expected_handoff_time)
        committed = ChainState(issue, issue + timedelta(hours=12), promoted.restart_file, promoted.model_time, state.archive_dir)
        try:
            store.save(committed)
        except Exception:
            promote_handoff_checkpoint(old_copy, checkpoint_dir, expected_model_time=state.handoff_model_time)
            raise
    finally:
        shutil.rmtree(rollback, ignore_errors=True)
    return committed


def process_available_issues(
    store: ChainStateStore,
    available: Callable[[datetime], bool],
    worker: IssueWorker,
    checkpoint_dir: Path,
    *,
    max_issues: int,
    output_selection: Mapping[str, bool] | None = None,
    permanently_retain_top_issues: Sequence[datetime] | None = None,
) -> tuple[ChainState, ...]:
    if max_issues < 1:
        raise ValueError("max_issues must be positive")
    completed = []
    for _ in range(max_issues):
        issue = store.load().next_expected_issue
        if not available(issue):
            break
        completed.append(
            process_expected_issue(
                issue,
                store,
                worker,
                checkpoint_dir,
                output_selection=output_selection,
                permanently_retain_top_issues=permanently_retain_top_issues,
            )
        )
    return tuple(completed)
