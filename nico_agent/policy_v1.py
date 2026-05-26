"""
V1 - Monte Carlo Control con epsilon-greedy (Hoja 11).

Aprendizaje OFFLINE por episodios completos en self-play. La tabla Q se
entrena con `train_v1.py` antes del torneo y se guarda en data/qtable_v1.pkl.
Durante el torneo, mount() la carga (cacheada en variable de modulo para no
releer disco cada juego) y act() es solo greedy lookup en microsegundos.

Resumen del algoritmo (Hoja 11):
  1. Genera un episodio completo en self-play eligiendo acciones con
     epsilon-greedy a partir de la Q-table actual.
  2. Al terminar el episodio, calcula el retorno terminal G desde la
     perspectiva del jugador que movio en cada paso (bipolar: +1 si gano,
     -1 si perdio, 0 si empate).
  3. Aplica First-Visit MC: por cada (estado, accion) visitado en el
     episodio, actualiza Q solo en la PRIMERA aparicion con el promedio
     incremental Q <- Q + (1/N)(G - Q).

Inferencia (esta clase): pura politica greedy. Para estados nunca vistos
durante entrenamiento, fallback a jugada aleatoria valida.
"""

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


# Cache a nivel de modulo: la tabla Q se carga una sola vez por proceso,
# aunque mount() se llame muchas veces (una por juego).
_CACHE: QTable | None = None
_DATA_PATH = Path(__file__).parent / "data" / "qtable_v1.pkl"


def _load_qtable() -> QTable:
    global _CACHE
    if _CACHE is None:
        if _DATA_PATH.exists():
            _CACHE = QTable.load(_DATA_PATH)
        else:
            # Sin tabla entrenada -> tabla vacia, el agente degenera a random.
            # Util para correr smoke tests del codigo sin haber entrenado.
            _CACHE = QTable()
    return _CACHE


class MCPolicy(Policy):
    """V1: Monte Carlo Control con epsilon-greedy. Inferencia greedy."""

    def mount(self) -> None:
        # Cargar (o reusar) la Q-table entrenada offline.
        self.qtable = _load_qtable()
        self.rng = np.random.default_rng()

    def act(self, s: np.ndarray) -> int:
        # Inferir a quien le toca y canonicalizar el tablero a "mi" perspectiva.
        player = infer_player(s)
        canonical = canonicalize(s, player)
        key = state_key(canonical)
        valid = free_cols(s)
        return self.qtable.greedy(key, valid, self.rng)
