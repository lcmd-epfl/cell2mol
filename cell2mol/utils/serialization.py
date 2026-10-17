"""Shared file persistence for models with object-graph JSON serialization."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from typing_extensions import Self, deprecated

from cell2mol.my_types import Format

logger = logging.getLogger(__name__)


class ModelFileMixin:
    """Save/load models as JSON by default, with explicit legacy pickle support."""

    def save(self, path: str | Path, *, format: Format = "json") -> None:
        if format == "json":
            self._save_as_json(path)
        elif format == "pickle":
            self._save_as_pickle(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def load(cls, path: str | Path, *, format: Format = "json") -> Self:
        if format == "json":
            return cls._load_from_json(path)
        if format == "pickle":
            return cls._load_from_pickle(path)
        raise ValueError(f"Unsupported format: {format}")

    def _save_as_json(self, path: str | Path) -> None:
        if not str(path).endswith(".json"):
            logger.warning("Use `.json` extension instead for path: %s", path)
        Path(path).write_text(
            self.to_json(indent=1, separators=(",", ":")), encoding="utf-8"
        )

    @classmethod
    def _load_from_json(cls, path: str | Path) -> Self:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    @deprecated("Use json format instead")
    def _save_as_pickle(self, path: str | Path) -> None:
        from cell2mol.utils.bond_records import upgrade_pickle_bonds

        upgrade_pickle_bonds(self)
        logger.warning("Use json format instead")
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    @deprecated("Use json format instead")
    def _load_from_pickle(cls, path: str | Path) -> Any:
        from cell2mol.utils.bond_records import upgrade_pickle_bonds

        with open(path, "rb") as handle:
            obj = pickle.load(handle)
        upgrade_pickle_bonds(obj)
        return obj
