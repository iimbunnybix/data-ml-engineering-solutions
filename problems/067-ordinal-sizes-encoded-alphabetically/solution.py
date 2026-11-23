"""Encode ordered categories (S < M < L < XL) by their real order, not alphabetically.

The returns-risk pipeline turned clothing size into a number with a
"sorted unique values" encoder, the default of many encoders (for example
scikit-learn's OrdinalEncoder). Alphabetically "L" < "M" < "S" < "XL" < "XXL",
so the model saw S as bigger than L, and the business rule
"size_code >= code of L" flagged every order, including size S.

The fix declares the order explicitly, normalises spelling ("xl " -> "XL"),
and refuses unknown sizes instead of silently giving them a code.
"""
from __future__ import annotations

from statistics import correlation

# The real order of sizes, smallest first. This list *is* the encoding.
SIZE_ORDER = ["XS", "S", "M", "L", "XL", "XXL"]


# --- the original, buggy version ---------------------------------------------
def naive_fit_encoding(values: list[str]) -> dict[str, int]:
    """Original encoder: codes follow the alphabetical order of the values seen."""
    codes: dict[str, int] = {}
    for code, value in enumerate(sorted(set(values))):
        codes[value] = code
    return codes


# --- the fix -----------------------------------------------------------------
class UnknownCategory(ValueError):
    """A value that is not in the declared order; it must not get a guessed code."""


def normalise_size(raw_size: str) -> str:
    """Canonical spelling: trimmed and upper-case ("xl " -> "XL")."""
    return raw_size.strip().upper()


def fit_ordinal_encoding(order: list[str]) -> dict[str, int]:
    """Codes from an explicit order; duplicates in the order are a config error."""
    codes: dict[str, int] = {}
    for code, category in enumerate(order):
        if category in codes:
            raise ValueError(f"category {category!r} listed twice in the order")
        codes[category] = code
    return codes


def encode(raw_value: str, codes: dict[str, int]) -> int:
    """Code for one value; unknown values raise instead of being guessed."""
    value = normalise_size(raw_value)
    if value not in codes:
        raise UnknownCategory(f"unknown size {raw_value!r}; expected one of {list(codes)}")
    return codes[value]


def encode_column(raw_values: list[str], codes: dict[str, int]) -> list[int]:
    """Encode a whole column, reporting every unknown value at once."""
    unknown = sorted({value for value in raw_values if normalise_size(value) not in codes})
    if unknown:
        raise UnknownCategory(f"unknown sizes {unknown}; expected one of {list(codes)}")
    return [encode(value, codes) for value in raw_values]


# --- what the downstream code does with the codes ------------------------------
def flag_large_orders(sizes: list[str], codes: dict[str, int], threshold_size: str) -> list[bool]:
    """Business rule: send orders of ``threshold_size`` or larger to a fit check."""
    threshold_code = codes[threshold_size]
    flags = []
    for size in sizes:
        flags.append(codes[normalise_size(size)] >= threshold_code)
    return flags


def size_return_correlation(sizes: list[str], returned: list[int], codes: dict[str, int]) -> float:
    """Pearson correlation between the size code and whether the order came back.

    A linear model can only use the size if this relationship is monotonic
    in the code, so a scrambled order shows up as a weak correlation.
    """
    size_codes = [codes[normalise_size(size)] for size in sizes]
    return correlation(size_codes, returned)
