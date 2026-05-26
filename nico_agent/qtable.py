"""
Estructura de tabla Q compartida entre V1 (Monte Carlo) y V2 (Q-learning bipolar).

Decision clave: estados CANONICALIZADOS por jugador.
    canonical(s, player) = s * player

Esto significa que la tabla Q se indexa por "como se ve el tablero desde el
punto de vista del jugador a mover". Mi color siempre es +1, el del oponente
siempre es -1. Asi:
  * Una sola entrada Q sirve para los dos jugadores en posiciones equivalentes.
  * La actualizacion bipolar de V2 sale natural: el siguiente estado tambien
    se canonicaliza (con el oponente), entonces "lo bueno para el oponente"
    invierte automaticamente el signo cuando lo veo desde mi perspectiva.

Almacenamiento: dict[bytes -> (Q[7], N[7])] donde:
  * bytes es el tablero canonicalizado serializado (~42 bytes por estado).
  * Q[7] son los Q-values estimados para cada columna (float32).
  * N[7] son los conteos de visitas a cada (estado, accion) (int32).
  * Las acciones invalidas (columnas llenas) tienen Q indefinido pero se
    enmascaran al seleccionar.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# Helpers de estado
# -----------------------------------------------------------------------------


def canonicalize(board: np.ndarray, player: int) -> np.ndarray:
    """Devuelve el tablero desde la perspectiva del `player`.

    Si player == -1 (rojo), multiplica por -1: mis fichas (rojas, originalmente
    -1) quedan como +1. Si player == +1 (amarillo), no cambia nada.
    """
    return (board * player).astype(np.int8)


def state_key(canonical_board: np.ndarray) -> bytes:
    """Clave hashable para diccionarios Python. ~42 bytes por estado."""
    return canonical_board.tobytes()


def free_cols(board: np.ndarray) -> list[int]:
    """Columnas no llenas. Se basa en la fila superior del tablero."""
    return [c for c in range(7) if board[0, c] == 0]


def infer_player(board: np.ndarray) -> int:
    """Infiere a quien le toca: rojo (-1) si #(-1) == #(1), amarillo (+1) si no.

    Funciona porque rojo siempre arranca y juegan alternando.
    """
    red_count = int(np.sum(board == -1))
    yellow_count = int(np.sum(board == 1))
    return -1 if red_count == yellow_count else 1


# -----------------------------------------------------------------------------
# QTable
# -----------------------------------------------------------------------------


class QTable:
    """Tabla Q indexada por bytes del tablero canonicalizado.

    Pensada como diccionario sparse: solo guarda los estados que efectivamente
    se han visitado durante el entrenamiento (los demas no entran).
    """

    def __init__(self) -> None:
        self.q: dict[bytes, np.ndarray] = {}  # state_key -> Q[7] float32
        self.n: dict[bytes, np.ndarray] = {}  # state_key -> N[7] int32

    # --- Tamanio --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.q)

    def __contains__(self, key: bytes) -> bool:
        return key in self.q

    def total_visits(self) -> int:
        return int(sum(arr.sum() for arr in self.n.values()))

    # --- Acceso ---------------------------------------------------------------

    def get_or_create(self, key: bytes) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (Q[7], N[7]) inicializando en ceros si no existe."""
        arr_q = self.q.get(key)
        if arr_q is None:
            arr_q = np.zeros(7, dtype=np.float32)
            arr_n = np.zeros(7, dtype=np.int32)
            self.q[key] = arr_q
            self.n[key] = arr_n
            return arr_q, arr_n
        return arr_q, self.n[key]

    # --- Politica greedy ------------------------------------------------------

    def greedy(
        self,
        key: bytes,
        valid: list[int],
        rng: np.random.Generator,
    ) -> int:
        """Elige la columna con mayor Q entre las validas.

        Si el estado no esta en la tabla, devuelve aleatorio entre validas
        (fallback degenerado, V2 lo reemplaza con una heuristica).
        Empates de Q se rompen aleatoriamente.
        """
        if not valid:
            raise ValueError("Sin columnas validas, juego no deberia continuar.")

        if key not in self.q:
            return int(rng.choice(valid))

        q_vals = self.q[key]
        best_q = max(q_vals[c] for c in valid)
        ties = [c for c in valid if q_vals[c] == best_q]
        return int(rng.choice(ties))

    # --- Persistencia ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"q": self.q, "n": self.n},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "QTable":
        with open(path, "rb") as f:
            data = pickle.load(f)
        t = cls()
        t.q = data["q"]
        t.n = data["n"]
        return t
