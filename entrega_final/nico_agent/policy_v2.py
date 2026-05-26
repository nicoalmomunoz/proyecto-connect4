

from __future__ import annotations

from pathlib import Path

import numpy as np

from connect4.policy import Policy

from nico_agent.heuristic import (
    center_preference,
    find_immediate_winner,
)
from nico_agent.qtable import (
    QTable,
    canonicalize,
    free_cols,
    infer_player,
    state_key,
)


_CACHE = None
_DATA_PATH = Path(__file__).parent / "data" / "qtable_v2.pkl"


def _load_qtable():
    global _CACHE
    if _CACHE is None:
        if _DATA_PATH.exists():
            _CACHE = QTable.load(_DATA_PATH)
        else:
            _CACHE = QTable()
    return _CACHE


class QLearningPolicy(Policy):


    use_heuristic = True

    def mount(self, *args, **kwargs):
        self.qtable = _load_qtable()
        self.rng = np.random.default_rng()

    def act(self, s):
        valid = free_cols(s)
        player = infer_player(s)

        if self.use_heuristic:
            winning = find_immediate_winner(s, player)
            if winning is not None and winning in valid:
                return winning

            blocker = find_immediate_winner(s, -player)
            if blocker is not None and blocker in valid:
                return blocker

        canonical = canonicalize(s, player)
        key = state_key(canonical)

        if key in self.qtable:
            return self.qtable.greedy(key, valid, self.rng)

        if self.use_heuristic:
            return center_preference(valid)
        return int(self.rng.choice(valid))


class QLearningPolicyNoHeuristic(QLearningPolicy):

    use_heuristic = False
