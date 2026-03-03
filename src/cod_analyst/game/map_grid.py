"""Map grid system — 5×5 zone mapping with progressive aliasing.

Converts (x, y) minimap coordinates to grid cells, supports
callout aliases, and provides zone lookup utilities.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MapInfo:
    """Per-map knowledge base with grid aliases and objectives."""
    map_name: str
    aliases: dict[str, str] = field(default_factory=dict)  # {"C2": "mid", ...}
    bombsites: list[str] = field(default_factory=list)      # ["A4", "E4"]
    hardpoints: list[dict] = field(default_factory=list)     # [{"name": "P1", "cells": [...]}]
    spawn_cells: dict[str, list[str]] = field(default_factory=dict)  # {"attack": [...], "defense": [...]}


class MapGrid:
    """5×5 grid overlay for minimap coordinates.

    Divides the minimap into 25 zones (A1–E5).
    Supports progressive aliasing of cells to callout names.

    Grid layout:
         1     2     3     4     5
    A  [A1]  [A2]  [A3]  [A4]  [A5]
    B  [B1]  [B2]  [B3]  [B4]  [B5]
    C  [C1]  [C2]  [C3]  [C4]  [C5]
    D  [D1]  [D2]  [D3]  [D4]  [D5]
    E  [E1]  [E2]  [E3]  [E4]  [E5]
    """

    _ROW_LABELS = "ABCDE"

    def __init__(self, grid_size: int = 5, maps_dir: str | Path | None = None):
        self.grid_size = grid_size
        self._map_info: dict[str, MapInfo] = {}
        self._current_map: str | None = None

        if maps_dir:
            self._load_maps(maps_dir)

    def _load_maps(self, maps_dir: str | Path) -> None:
        """Load per-map knowledge base JSON files."""
        maps_path = Path(maps_dir)
        if not maps_path.exists():
            logger.warning("Maps directory not found: %s", maps_dir)
            return

        for json_path in sorted(maps_path.glob("*.json")):
            try:
                data = json.loads(json_path.read_text())
                map_name = data.get("map_name", json_path.stem)
                self._map_info[map_name.lower()] = MapInfo(
                    map_name=map_name,
                    aliases=data.get("aliases", {}),
                    bombsites=data.get("bombsites", []),
                    hardpoints=data.get("hardpoints", []),
                    spawn_cells=data.get("spawn_cells", {}),
                )
                logger.info("Loaded map: %s (%d aliases)", map_name, len(data.get("aliases", {})))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load map file %s: %s", json_path, e)

    def set_current_map(self, map_name: str) -> None:
        """Set the current map for alias lookups."""
        self._current_map = map_name.lower()

    def coord_to_cell(self, x: float, y: float) -> str:
        """Convert normalized (x, y) position to a grid cell label.

        Parameters
        ----------
        x, y : float
            Normalized coordinates in [0.0, 1.0].

        Returns
        -------
        str
            Grid cell label like "C3" or aliased name like "mid".
        """
        col = min(int(x * self.grid_size), self.grid_size - 1)
        row = min(int(y * self.grid_size), self.grid_size - 1)

        cell = f"{self._ROW_LABELS[row]}{col + 1}"

        # Apply alias if available
        if self._current_map and self._current_map in self._map_info:
            aliases = self._map_info[self._current_map].aliases
            return aliases.get(cell, cell)

        return cell

    def cell_to_raw(self, cell: str) -> str:
        """Convert an aliased cell name back to raw grid label.

        Looks up reverse alias. Returns the input if no alias found.
        """
        if self._current_map and self._current_map in self._map_info:
            aliases = self._map_info[self._current_map].aliases
            reverse = {v: k for k, v in aliases.items()}
            return reverse.get(cell, cell)
        return cell

    def get_cell_center(self, cell: str) -> tuple[float, float]:
        """Get the center coordinates of a grid cell.

        Parameters
        ----------
        cell : str
            Raw grid cell label like "C3".

        Returns
        -------
        tuple[float, float]
            (x, y) center coordinates in [0.0, 1.0].
        """
        raw = self.cell_to_raw(cell)
        if len(raw) < 2 or raw[0] not in self._ROW_LABELS:
            return 0.5, 0.5

        row = self._ROW_LABELS.index(raw[0])
        col = int(raw[1]) - 1

        x = (col + 0.5) / self.grid_size
        y = (row + 0.5) / self.grid_size
        return x, y

    def distance_between_cells(self, cell_a: str, cell_b: str) -> float:
        """Compute Euclidean distance between cell centers."""
        ax, ay = self.get_cell_center(cell_a)
        bx, by = self.get_cell_center(cell_b)
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    def get_bombsites(self) -> list[str]:
        """Get bombsite cells for the current map."""
        if self._current_map and self._current_map in self._map_info:
            return self._map_info[self._current_map].bombsites
        return []

    def get_map_info(self) -> MapInfo | None:
        """Get full map info for the current map."""
        if self._current_map:
            return self._map_info.get(self._current_map)
        return None

    def all_cells(self) -> list[str]:
        """Return all grid cell labels."""
        return [
            f"{self._ROW_LABELS[r]}{c + 1}"
            for r in range(self.grid_size)
            for c in range(self.grid_size)
        ]
