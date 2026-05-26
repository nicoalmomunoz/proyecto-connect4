"""
V0 - Politica Aleatoria (baseline).

Esta NO es una version del agente final que va al torneo. Es el oponente
de referencia contra el cual mido a V1 y V2. La rubrica exige no perder
nunca contra esta politica y ganarle al menos 50% de los juegos.
"""

import numpy as np

from connect4.policy import Policy


class RandomPolicy(Policy):
    """Selecciona uniformemente entre las columnas no llenas."""

    def mount(self) -> None:
        # mount() se llama una vez por juego. Inicializo el RNG aqui
        # para evitar que dos partidas seguidas usen la misma secuencia.
        self.rng = np.random.default_rng()

    def act(self, s: np.ndarray) -> int:
        # s es el tablero 6x7: -1 = rojo, 0 = vacio, 1 = amarillo.
        # Una columna esta libre si su casilla mas alta (fila 0) esta vacia.
        available = [c for c in range(7) if s[0, c] == 0]
        return int(self.rng.choice(available))
