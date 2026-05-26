

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from connect4.connect_state import ConnectState
from connect4.policy import Policy




@dataclass
class ColorStats:
    n: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def loss_rate(self) -> float:
        return self.losses / self.n if self.n else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.n if self.n else 0.0


@dataclass
class MatchStats:
    n_games: int
    wins: int = 0
    losses: int = 0
    draws: int = 0
    as_red: ColorStats = field(default_factory=ColorStats)
    as_yellow: ColorStats = field(default_factory=ColorStats)
    avg_moves: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.n_games if self.n_games else 0.0

    @property
    def loss_rate(self) -> float:
        return self.losses / self.n_games if self.n_games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.n_games if self.n_games else 0.0

    def to_dict(self) -> dict:
        return {
            "n_games": self.n_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "loss_rate": self.loss_rate,
            "draw_rate": self.draw_rate,
            "wins_as_red": self.as_red.wins,
            "losses_as_red": self.as_red.losses,
            "draws_as_red": self.as_red.draws,
            "win_rate_as_red": self.as_red.win_rate,
            "wins_as_yellow": self.as_yellow.wins,
            "losses_as_yellow": self.as_yellow.losses,
            "draws_as_yellow": self.as_yellow.draws,
            "win_rate_as_yellow": self.as_yellow.win_rate,
            "avg_moves": self.avg_moves,
        }



def play_one_game(
    policy_red: Policy,
    policy_yellow: Policy,
    mount_each: bool = True,
) -> tuple[int, int]:

    if mount_each:
        policy_red.mount()
        policy_yellow.mount()

    state = ConnectState()
    moves = 0
    while not state.is_final():
        current = policy_red if state.player == -1 else policy_yellow
        action = current.act(state.board)
        state = state.transition(int(action))
        moves += 1

    return state.get_winner(), moves


def evaluate(
    policy: Policy,
    opponent: Policy,
    n_games: int = 100,
    alternate_first: bool = True,
    seed: int | None = None,
) -> MatchStats:

    rng = np.random.default_rng(seed)

    stats = MatchStats(n_games=n_games)
    total_moves = 0

    for i in range(n_games):
        if alternate_first:
            policy_is_red = (i % 2 == 0)
        else:
            policy_is_red = bool(rng.random() < 0.5)

        if policy_is_red:
            winner, moves = play_one_game(policy, opponent)
            color_stats = stats.as_red
        else:
            winner, moves = play_one_game(opponent, policy)
            color_stats = stats.as_yellow

        total_moves += moves
        color_stats.n += 1

        # Traducir el resultado al punto de vista de `policy`.
        my_color = -1 if policy_is_red else 1
        if winner == my_color:
            stats.wins += 1
            color_stats.wins += 1
        elif winner == 0:
            stats.draws += 1
            color_stats.draws += 1
        else:
            stats.losses += 1
            color_stats.losses += 1

    stats.avg_moves = total_moves / n_games if n_games else 0.0
    return stats


def head_to_head(
    policy_a: Policy,
    policy_b: Policy,
    n_games: int = 100,
    seed: int | None = None,
) -> tuple[MatchStats, MatchStats]:

    stats_a = evaluate(policy_a, policy_b, n_games=n_games, seed=seed)
    stats_b = MatchStats(
        n_games=stats_a.n_games,
        wins=stats_a.losses,
        losses=stats_a.wins,
        draws=stats_a.draws,
    )
    # Espejo por color: si policy_a fue rojo X veces, policy_b fue amarillo X veces.
    stats_b.as_red.n = stats_a.as_yellow.n
    stats_b.as_red.wins = stats_a.as_yellow.losses
    stats_b.as_red.losses = stats_a.as_yellow.wins
    stats_b.as_red.draws = stats_a.as_yellow.draws
    stats_b.as_yellow.n = stats_a.as_red.n
    stats_b.as_yellow.wins = stats_a.as_red.losses
    stats_b.as_yellow.losses = stats_a.as_red.wins
    stats_b.as_yellow.draws = stats_a.as_red.draws
    stats_b.avg_moves = stats_a.avg_moves
    return stats_a, stats_b
