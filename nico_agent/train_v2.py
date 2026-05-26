"""
Script de entrenamiento de V2 (Q-Learning Bipolar + UCB + Simetria).

Uso:
    python -m nico_agent.train_v2 --episodes 100000

Diferencias clave vs train_v1.py:
  * UCB1 (Hoja 10) en vez de epsilon-greedy.
  * Actualizacion TD bipolar (Hoja 12) en vez de First-Visit MC.
  * Augmentacion con simetria izquierda-derecha del tablero.
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
# Seleccion UCB1 (Hoja 10)
# -----------------------------------------------------------------------------


def ucb_select(
    qtable: QTable,
    key: bytes,
    valid: list[int],
    c: float,
    rng: np.random.Generator,
) -> int:
    """Selecciona accion segun UCB1 entre las columnas validas.

    UCB1(a) = Q(s, a) + c * sqrt(ln N(s) / N(s, a))

    Si alguna accion valida tiene N(s, a) == 0, la prioriza (de las
    inexploradas, devuelve la mas centrada por preferencia estructural;
    esto reduce ruido inicial).
    """
    q, n = qtable.get_or_create(key)

    # Acciones nunca exploradas: prioridad infinita en UCB1.
    unexplored = [a for a in valid if n[a] == 0]
    if unexplored:
        # Preferir la mas central entre las inexploradas para reducir varianza.
        center_order = [3, 2, 4, 1, 5, 0, 6]
        for a in center_order:
            if a in unexplored:
                return a

    total_n = int(sum(n[a] for a in valid))
    log_total = np.log(max(total_n, 1))

    best_score = -np.inf
    best_actions: list[int] = []
    for a in valid:
        score = float(q[a]) + c * float(np.sqrt(log_total / n[a]))
        if score > best_score:
            best_score = score
            best_actions = [a]
        elif score == best_score:
            best_actions.append(a)

    return int(rng.choice(best_actions))


# -----------------------------------------------------------------------------
# Una partida self-play con UCB1
# -----------------------------------------------------------------------------


def play_episode(
    qtable: QTable,
    c_ucb: float,
    rng: np.random.Generator,
) -> tuple[list[tuple[bytes, int, int, list[int]]], int]:
    """Devuelve (history, winner).

    history: [(state_key, action, player, valid_cols)]
        Guardamos `valid_cols` para no recomputarlo en la fase de actualizacion.
    """
    state = ConnectState()
    history: list[tuple[bytes, int, int, list[int]]] = []

    while not state.is_final():
        player = state.player
        canonical = canonicalize(state.board, player)
        key = state_key(canonical)
        valid = free_cols(state.board)
        qtable.get_or_create(key)

        action = ucb_select(qtable, key, valid, c_ucb, rng)
        history.append((key, action, player, valid))
        state = state.transition(action)

    return history, state.get_winner()


# -----------------------------------------------------------------------------
# Actualizacion TD(0) bipolar
# -----------------------------------------------------------------------------


def bipolar_td_update(
    qtable: QTable,
    history: list[tuple[bytes, int, int, list[int]]],
    winner: int,
    alpha: float,
    gamma: float = 1.0,
    augment_symmetry: bool = True,
) -> None:
    """TD(0) bipolar: actualiza Q(s, a) hacia el target del SIGUIENTE estado
    desde la perspectiva del MISMO jugador (es decir, dos movimientos adelante).

    Como la Q-table esta canonicalizada, el "siguiente estado de este jugador"
    es history[i + 2]. La bipolaridad esta absorbida en la canonicalizacion:
    al mirar el estado dos turnos adelante (que vuelve a ser de este jugador),
    el max sobre Q ya esta del lado correcto.

    Si i+2 >= len(history), el episodio termino antes de que este jugador
    volviera a mover; usamos la recompensa terminal desde su perspectiva.

    Augmentacion: si activada, replicamos la actualizacion en el estado
    espejo izquierda-derecha (siempre que no coincida exactamente con el
    original, para no duplicar updates).
    """
    L = len(history)
    for i, (key, action, player, _valid) in enumerate(history):
        next_idx = i + 2

        if next_idx < L:
            next_key, _, next_player, next_valid = history[next_idx]
            # Sanity: a 2 turnos vuelve a mover el mismo jugador.
            assert next_player == player
            next_q = qtable.q[next_key]
            target = gamma * float(max(next_q[a] for a in next_valid))
        else:
            # Terminal alcanzado dentro de los proximos 2 movimientos.
            if winner == 0:
                target = 0.0
            elif winner == player:
                target = 1.0
            else:
                target = -1.0

        q, n = qtable.get_or_create(key)
        n[action] += 1
        q[action] += alpha * (target - q[action])

        if augment_symmetry:
            mirror_board = (
                np.frombuffer(key, dtype=np.int8).reshape(6, 7)[:, ::-1].copy()
            )
            mirror_key = mirror_board.tobytes()
            if mirror_key != key:  # evita duplicar updates en estados simetricos
                mirror_action = 6 - action
                mq, mn = qtable.get_or_create(mirror_key)
                mn[mirror_action] += 1
                mq[mirror_action] += alpha * (target - mq[mirror_action])


# -----------------------------------------------------------------------------
# Loop de entrenamiento
# -----------------------------------------------------------------------------


def train(
    num_episodes: int,
    alpha: float = 0.1,
    c_ucb: float = 1.41,
    gamma: float = 1.0,
    augment_symmetry: bool = True,
    seed: int = 42,
    log_every: int = 5000,
    qtable: QTable | None = None,
) -> QTable:
    rng = np.random.default_rng(seed)
    if qtable is None:
        qtable = QTable()

    t0 = time.time()
    for ep in range(num_episodes):
        history, winner = play_episode(qtable, c_ucb, rng)
        bipolar_td_update(qtable, history, winner, alpha, gamma, augment_symmetry)

        if (ep + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (ep + 1) / elapsed
            print(
                f"  ep {ep+1:>7d}/{num_episodes}  "
                f"|Q|={len(qtable):>7d}  "
                f"({rate:.0f} eps/s)"
            )

    return qtable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrenamiento V2 (Q-Learning Bipolar + UCB)."
    )
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "data" / "qtable_v2.pkl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--c-ucb", type=float, default=1.41)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--no-symmetry",
        action="store_true",
        help="Desactiva augmentacion con simetria izq-der (para ablacion).",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    qtable: QTable | None = None
    if args.resume and out_path.exists():
        print(f"Reanudando desde {out_path}...")
        qtable = QTable.load(out_path)
        print(f"  Estados existentes: {len(qtable):,}")

    print(
        f"Entrenando V2 (Q-Learning Bipolar + UCB) con "
        f"{args.episodes:,} episodios self-play..."
    )
    print(
        f"  alpha={args.alpha}  c_ucb={args.c_ucb}  gamma={args.gamma}  "
        f"symmetry={'OFF' if args.no_symmetry else 'ON'}"
    )

    qtable = train(
        num_episodes=args.episodes,
        alpha=args.alpha,
        c_ucb=args.c_ucb,
        gamma=args.gamma,
        augment_symmetry=not args.no_symmetry,
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
