"""Immutable, shared progress-bar policy for experiment commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from tqdm import tqdm


@dataclass(frozen=True, slots=True)
class InlineProgressPlugin:
    """Create fixed-width progress bars that refresh in place."""

    ncols: int = 80
    dynamic_ncols: bool = False

    def create(
        self,
        iterable: Iterable[Any] | None = None,
        *,
        total: int | None = None,
        desc: str | None = None,
        unit: str = "it",
        disable: bool = False,
        leave: bool = True,
    ) -> tqdm:
        return tqdm(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            disable=disable,
            leave=leave,
            ncols=self.ncols,
            dynamic_ncols=self.dynamic_ncols,
        )


INLINE_PROGRESS = InlineProgressPlugin()
