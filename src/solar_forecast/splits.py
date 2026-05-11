"""
Time-aware train/validation/test splits for solar forecasting.

Why this lives in its own module: every model — baselines, ML, PINN — must use
the *same* splits, or numbers across runs aren't comparable. Centralizing the
split logic guarantees that, and makes it impossible to accidentally leak
future information into training.

The canonical split is chronological:
    train:      2015-01-01 → 2017-12-31   (3 years, model fits these)
    validation: 2018-01-01 → 2018-12-31   (1 year, hyperparameter tuning)
    test:       2019-01-01 → 2019-12-31   (1 year, final reporting only)

Test is touched exactly once — at the end, after all decisions are frozen.
Anything else is hyperparameter-hacking against the test set, which inflates
reported numbers and tanks real-world performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    """Boundaries of the canonical chronological split, inclusive on both ends."""
    train_start: str = "2015-01-01 00:00:00"
    train_end: str = "2017-12-31 23:00:00"
    val_start: str = "2018-01-01 00:00:00"
    val_end: str = "2018-12-31 23:00:00"
    test_start: str = "2019-01-01 00:00:00"
    test_end: str = "2019-12-31 23:00:00"


def split_dataset(
    df: pd.DataFrame,
    split: Split | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Slice the dataset into train, validation, and test by timestamp.

    Returns three views (not copies — the caller doesn't typically mutate, and
    copying 40k rows three times is wasteful). If you plan to mutate, .copy()
    explicitly at the call site.
    """
    s = split or Split()
    train = df.loc[s.train_start:s.train_end]
    val = df.loc[s.val_start:s.val_end]
    test = df.loc[s.test_start:s.test_end]
    return train, val, test
