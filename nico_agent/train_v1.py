"""
Script de entrenamiento de V1 (Monte Carlo Control con epsilon-greedy).

Uso:
    python -m nico_agent.train_v1 --episodes 100000

El agente juega contra si mismo `episodes` partidas. epsilon decrece
linealmente de 1.0 a 0.05 a lo largo del entrenamiento. Al final, guarda
la Q-table en data/qtable_v1.pkl para que MCPolicy la cargue.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from connect4.connect_state import ConnectState

from nico_agent.qtable import (
    QTable,
    canonicalize,
    free_cols,
    state_key,
)


# -----------------------------------------------------------------------------
# Politica de exploracion durante entrenamiento
# -----------------------------------------------------------------------------


def select_eps_greedy(
    qtable: QTable,
    key: bytes,
    valid: list[int],
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """epsilon-greedy: explora aleatorio con prob epsilon, sino greedy."""
    if rng.random() < epsilon:
        return int(rng.choice(valid))
    return qtable.greedy(key, valid, rng)


# -----------------------------------------------------------------------------
# Una partida self-play
# -----------------------------------------------------------------------------


def play_episode(
    qtable: QTable,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[list[tuple[bytes, int, int]], int]:
    """Juega una partida self-play y devuelve (trajectory, winner).

    trajectory: lista de (state_key, action, player_que_movio)
        El jugador es -1 o +1. Lo necesitamos para calcular el retorno
        bipolar (mi G no es el mismo que el de mi oponente).
    winner: -1 (rojo), +1 (amarillo), 0 (empate).
    """
    state = ConnectState()
    trajectory: list[tuple[bytes, int, int]] = []

    while not state.is_final():
        canonical = canonicalize(state.board, state.player)
        key = state_key(canonical)
        valid = free_cols(state.board)

        # Asegurar que el estado existe en la tabla antes de seleccionar.
        # Esto cuenta el estado en len(qtable) aunque no lo actualicemos aun.
        qtable.get_or_create(key)

        action = select_eps_greedy(qtable, key, valid, epsilon, rng)
        trajectory.append((key, action, state.player))
        state = state.transition(action)

    return trajectory, state.get_winner()


# -----------------------------------------------------------------------------
# Actualizacion First-Visit MC
# -----------------------------------------------------------------------------


def first_visit_update(
    qtable: QTable,
    trajectory: list[tuple[bytes, int, int]],
    winner: int,
) -> None:
    """First-Visit MC: solo actualiza la PRIMERA visita de cada (s, a).

    Retorno bipolar: cada paso del episodio se evalua desde la perspectiva
    del jugador que movio. Si el jugador eventualmente gano, G = +1; si
    perdio G = -1; empate G = 0. No descontamos (gamma = 1) porque no hay
    recompensas intermedias y los episodios son cortos.
    """
    seen: set[tuple[bytes, int]] = set()
    for key, action, player in trajectory:
        if (key, action) in seen:
            continue
        seen.add((key, action))

        if winner == 0:
            G = 0.0
        elif winner == player:
            G = 1.0
        else:
            G = -1.0

        q, n = qtable.get_or_create(key)
        n[action] += 1
        # Promedio incremental: equivalente a Q = mean(returns vistos).
        q[action] += (G - q[action]) / n[action]


# -----------------------------------------------------------------------------
# Loop de entrenamiento
# -----------------------------------------------------------------------------


def train(
    num_episodes: int,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    seed: int = 42,
    log_every: int = 5000,
    qtable: QTable | None = None,
) -> QTable:
    """Entrena V1 durante `num_episodes` partidas self-play."""
    rng = np.random.default_rng(seed)
    if qtable is None:
        qtable = QTable()

    t0 = time.time()
    for ep in range(num_episodes):
        # epsilon annealed linealmente de eps_start a eps_end.
        eps = eps_start + (eps_end - eps_start) * (ep / max(num_episodes - 1, 1))

        trajectory, winner = play_episode(qtable, eps, rng)
        first_visit_update(qtable, trajectory, winner)

        if (ep + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (ep + 1) / elapsed
            print(
                f"  ep {ep+1:>7d}/{num_episodes}  "
                f"|Q|={len(qtable):>7d}  "
                f"eps={eps:.3f}  "
                f"({rate:.0f} eps/s)"
            )

    return qtable


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento V1 (Monte Carlo).")
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "data" / "qtable_v1.pkl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Si --out existe, retoma desde la Q-table guardada en lugar de partir de cero.",
    )
    parser.add_argument(
        "--eps-start",
        type=float,
        default=1.0,
        help="Epsilon inicial. Util bajarlo (p.ej. 0.3) cuando se hace --resume.",
    )
    parser.add_argument(
        "--eps-end",
        type=float,
        default=0.05,
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    qtable: QTable | None = None
    if args.resume and out_path.exists():
        print(f"Reanudando desde {out_path}...")
        qtable = QTable.load(out_path)
        print(f"  Estados existentes: {len(qtable):,}")

    print(
        f"Entrenando V1 (Monte Carlo) con {args.episodes:,} episodios self-play..."
    )
    qtable = train(
        num_episodes=args.episodes,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        seed=args.seed,
        log_every=args.log_every,
        qtable=qtable,
    )

    qtable.save(out_path)
    print(f"\nGuardado en {out_path}")
    print(f"Estados unicos:  {len(qtable):,}")
    print(f"Visitas totales: {qtable.total_visits():,}")


if __name__ == "__main__":
    main()
