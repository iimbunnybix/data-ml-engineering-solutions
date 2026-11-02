"""Data contract validation for incoming tabular files (no external dependencies)."""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Column:
    name: str
    dtype: str                         # "int" | "float" | "str" | "date"
    required: bool = True
    aliases: tuple = ()
    min_value: float | None = None
    max_value: float | None = None
    allowed: tuple | None = None
    date_format: str = "%Y-%m-%d"
    max_null_rate: float = 0.05


@dataclass
class Report:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class SchemaViolation(Exception):
    def __init__(self, report: Report):
        super().__init__("; ".join(report.errors))
        self.report = report


def _parse(value: str, col: Column):
    if col.dtype == "int":
        return int(value)
    if col.dtype == "float":
        return float(value)
    if col.dtype == "date":
        return datetime.strptime(value, col.date_format)
    return value


def validate(rows: list[dict], header: list[str], contract: list[Column]) -> tuple[list[dict], Report]:
    report = Report()
    rename = {}
    for col in contract:
        if col.name in header:
            continue
        alias = next((a for a in col.aliases if a in header), None)
        if alias:
            rename[alias] = col.name
            report.warnings.append(f"column '{alias}' mapped to '{col.name}' via alias")
        elif col.required:
            report.errors.append(f"missing required column '{col.name}'")

    known = {c.name for c in contract} | set(rename)
    for extra in sorted(set(header) - known):
        report.warnings.append(f"unexpected new column '{extra}'")

    rows = [{rename.get(k, k): v for k, v in r.items()} for r in rows]
    n = max(len(rows), 1)
    clean = [dict(r) for r in rows]
    for col in contract:
        if not any(col.name in r for r in rows):
            continue
        nulls, bad = 0, []
        for i, r in enumerate(rows):
            raw = (r.get(col.name) or "").strip()
            if raw == "":
                nulls += 1
                clean[i][col.name] = None
                continue
            try:
                v = _parse(raw, col)
            except ValueError:
                bad.append(i)
                continue
            if col.min_value is not None and v < col.min_value or \
               col.max_value is not None and v > col.max_value:
                bad.append(i)
            elif col.allowed is not None and v not in col.allowed:
                bad.append(i)
            clean[i][col.name] = v
        if nulls / n > col.max_null_rate:
            report.errors.append(f"'{col.name}' null rate {nulls / n:.0%} > {col.max_null_rate:.0%}")
        if bad:
            report.errors.append(f"'{col.name}' has {len(bad)} invalid value(s), e.g. row {bad[0] + 1}")
    return clean, report


def load_csv(text: str, contract: list[Column]) -> list[dict]:
    """Load and validate; raise before bad data reaches the model."""
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    clean, report = validate(rows, reader.fieldnames or [], contract)
    if not report.ok:
        raise SchemaViolation(report)
    return clean


CUSTOMER_CONTRACT = [
    Column("customer_id", "str", max_null_rate=0.0),
    Column("monthly_spend", "float", aliases=("monthly_spend_inr",), min_value=0, max_value=1e7),
    Column("signup_date", "date"),
    Column("plan", "str", allowed=("basic", "plus", "pro")),
    Column("tenure_months", "int", min_value=0, max_value=600),
]
