"""Data models for mosaic-figures."""

from __future__ import annotations

import csv
import io
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PlotKind(str, Enum):
    """Supported plot types."""

    AUTO = "auto"
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    BOX = "box"
    VIOLIN = "violin"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    STRIP = "strip"
    SWARM = "swarm"


# Explicitly banned kinds — raise ValueError with explanation
BANNED_KINDS = {
    "pie": "Pie charts use angle-based encoding, which humans perceive inaccurately (Cleveland & McGill 1984). Use kind='bar' instead.",
    "donut": "Donut charts have the same perceptual problems as pie charts. Use kind='bar' instead.",
    "3d_bar": "3D bar charts distort heights through perspective projection. Use kind='bar' instead.",
    "3d_pie": "3D pie charts compound angle distortion with perspective. Use kind='bar' instead.",
}


class FigureMode(str, Enum):
    """Figure output mode."""

    PUBLICATION = "publication"
    SHOWCASE = "showcase"


class DataTable:
    """Normalized columnar data representation.

    Internal data structure — all input formats (dict of lists, list of dicts,
    CSV string) are normalized to this form for uniform processing.
    """

    def __init__(self, columns: dict[str, list[Any]]) -> None:
        self.columns = columns
        # Validate equal column lengths
        lengths = {k: len(v) for k, v in columns.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(f"All columns must have equal length, got: {lengths}")

    @property
    def column_names(self) -> list[str]:
        return list(self.columns.keys())

    @property
    def n_rows(self) -> int:
        if not self.columns:
            return 0
        return len(next(iter(self.columns.values())))

    @property
    def n_cols(self) -> int:
        return len(self.columns)

    def is_numeric(self, col: str) -> bool:
        """Check if a column contains numeric data."""
        values = self.columns[col]
        return all(isinstance(v, (int, float)) for v in values if v is not None)

    def is_categorical(self, col: str) -> bool:
        """Check if a column contains categorical (string) data."""
        values = self.columns[col]
        return any(isinstance(v, str) for v in values if v is not None)

    def unique_count(self, col: str) -> int:
        """Number of unique values in a column."""
        return len(set(self.columns[col]))

    def values_per_category(self, cat_col: str, val_col: str) -> dict[str, list[Any]]:
        """Group values by category: {category: [values]}."""
        result: dict[str, list[Any]] = {}
        cats = self.columns[cat_col]
        vals = self.columns[val_col]
        for cat, val in zip(cats, vals):
            key = str(cat)
            if key not in result:
                result[key] = []
            result[key].append(val)
        return result

    @classmethod
    def from_dict(cls, d: dict[str, list[Any]]) -> DataTable:
        """Create from columnar dict: {"col": [values]}."""
        return cls(columns=dict(d))

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> DataTable:
        """Create from row records: [{"col": val, ...}, ...]."""
        if not records:
            raise ValueError("Cannot create DataTable from empty records list")
        all_keys: list[str] = []
        seen: set[str] = set()
        for r in records:
            for k in r:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        columns: dict[str, list[Any]] = {k: [] for k in all_keys}
        for r in records:
            for k in all_keys:
                columns[k].append(r.get(k))
        return cls(columns=columns)

    @classmethod
    def from_csv(cls, csv_string: str) -> DataTable:
        """Create from CSV string."""
        reader = csv.DictReader(io.StringIO(csv_string.strip()))
        records = list(reader)
        if not records:
            raise ValueError("CSV string contains no data rows")
        # Try to convert numeric strings
        columns: dict[str, list[Any]] = {}
        for key in records[0]:
            raw = [r.get(key, "") for r in records]
            converted: list[Any] = []
            for v in raw:
                if v is None or v == "":
                    converted.append(None)
                    continue
                try:
                    converted.append(int(v))
                except ValueError:
                    try:
                        converted.append(float(v))
                    except ValueError:
                        converted.append(v)
            columns[key] = converted
        return cls(columns=columns)

    @classmethod
    def normalize(
        cls, data: dict[str, list[Any]] | list[dict[str, Any]] | str
    ) -> DataTable:
        """Auto-detect format and normalize to DataTable."""
        if isinstance(data, str):
            return cls.from_csv(data)
        if isinstance(data, list):
            return cls.from_records(data)
        if isinstance(data, dict):
            return cls.from_dict(data)
        raise TypeError(
            f"Unsupported data type: {type(data)}. Expected dict, list[dict], or CSV string."
        )


class FigureResult(BaseModel):
    """Result returned by figure()."""

    path: str = Field(description="Output file path")
    legend: str = Field(description="Auto-generated figure legend text")
    advisor_notes: list[str] = Field(
        default_factory=list,
        description="Warnings and recommendations from the advisor",
    )
    kind: str = Field(description="Plot type used (resolved from 'auto' if applicable)")
    width_inches: float = Field(description="Figure width in inches")
    height_inches: float = Field(description="Figure height in inches")
    dpi: int = Field(description="Resolution in dots per inch")
    palette: str = Field(description="Color palette name used")
    log_scale: str | None = Field(
        default=None, description="Which axes use log scale ('x', 'y', 'both', or None)"
    )
    data_summary: str = Field(
        description="Summary of data shape, e.g. '4 groups, 1 series'"
    )
    aggregation: str | None = Field(
        default=None, description="Aggregation applied, e.g. 'mean ± SEM'"
    )
