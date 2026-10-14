"""Contracts for one ICON-2I forecast issue.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


REQUIRED_GRIB_VARIABLES = (
    "temperature",
    "cloud_cover",
    "relative_humidity_1000hpa",
    "wind_u",
    "wind_v",
    "shortwave_radiation",
    "precipitation",
)


@dataclass(frozen=True)
class IssueInputs:
    """Files needed to prepare one complete AEM3D forcing issue."""

    issue_time: datetime
    grib_files: Mapping[str, Path]
    zone_mapping: Path
    output_dir: Path

    def validate(self) -> tuple[str, ...]:
        """Return actionable validation errors without touching output files."""

        errors: list[str] = []
        if self.issue_time.tzinfo is None:
            errors.append("issue_time must be timezone-aware")
        else:
            utc = self.issue_time.astimezone(timezone.utc)
            if utc.hour not in {0, 12} or utc.minute or utc.second or utc.microsecond:
                errors.append("issue_time must be exactly 00:00 or 12:00 UTC")

        missing = [name for name in REQUIRED_GRIB_VARIABLES if name not in self.grib_files]
        if missing:
            errors.append("missing GRIB variables: " + ", ".join(missing))

        for variable, path in self.grib_files.items():
            if not isinstance(path, Path):
                errors.append(f"GRIB path for {variable!r} must be a pathlib.Path")
            elif not path.is_file():
                errors.append(f"GRIB file for {variable!r} does not exist: {path}")

        if not self.zone_mapping.is_file():
            errors.append(f"zone mapping does not exist: {self.zone_mapping}")

        if self.output_dir.exists() and not self.output_dir.is_dir():
            errors.append(f"output_dir is not a directory: {self.output_dir}")

        return tuple(errors)

    def require_valid(self) -> None:
        """Raise ``ValueError`` if the issue cannot be prepared safely."""

        errors = self.validate()
        if errors:
            raise ValueError("Invalid forecast issue:\n- " + "\n- ".join(errors))

    @property
    def issue_time_utc(self) -> datetime:
        """Return the issue time normalized to UTC."""

        if self.issue_time.tzinfo is None:
            raise ValueError("issue_time must be timezone-aware")
        return self.issue_time.astimezone(timezone.utc)
