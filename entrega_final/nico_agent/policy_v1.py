

from __future__ import annotations

from pathlib import Path

import numpy as np

from connect4.policy import Policy

from nico_agent.qtable import (
    QTable,
    canonicalize,
    free_cols,
    infer_player,
    state_key,
)

_CACHE: QTable | None = None
_DATA_PATH = Path(__file__).parent / "data" / "qtable_v1.pkl"


def _load_qtable() -> QTable:
    global _CACHE
    if _CACHE is None:
        if _DATA_PATH.exists():
            _CACHE = QTable.load(_DATA_PATH)
        else:

            _CACHE = QTable()
    return _CACHE


class MCPolicy(Policy):
    """V1: Monte Carlo Control con epsilon-greedy. Inferencia greedy."""

    def mount(self) -> None:

        self.qtable = _load_qtable()
        self.rng = np.random.default_rng()

    def act(self, s: np.ndarray) -> int:

        player = infer_player(s)
        canonical = canonicalize(s, player)
        key = state_key(canonical)
        valid = free_cols(s)
        return self.qtable.greedy(key, valid, self.rng)
