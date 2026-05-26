"""
Heuristica de seguridad para V2.

Tres reglas, en orden de prioridad cuando se aplican durante act():
  1. WIN: si tengo una jugada que crea un 4-en-linea inmediato, la juego.
  2. BLOCK: si el oponente tendria un 4-en-linea inmediato en su proximo
     turno, juego en esa columna para bloquearlo.
  3. CENTER FALLBACK: si la Q-table no tiene valor para este estado y
     ninguna de las anteriores aplica, prefiero columnas centrales (3,
     luego 2/4, luego 1/5, luego 0/6). El centro es estrategicamente
     superior en Connect-4 (mas combinaciones de 4-en-linea pasan por el).

La regla 1 + regla 2 sola garantizan no perder contra random:
  - Random nunca aprovechara una victoria que tenga al alcance.
  - Si random crea una amenaza, la bloqueamos.
  - El unico modo de perder seria que random forzara una jugada doble
    (dos amenazas simultaneas), lo cual ocurre con probabilidad muy baja
    si jugamos centrado y greedy en la Q-table.

Esta heuristica se puede DESACTIVAR (parametro use_heuristic=False de
QLearningPolicy) para mostrar empiricamente su aporte en el notebook.
"""

from __future__ import annotations

import numpy as np


COLS = 7
ROWS = 6


def _drop_row(board: np.ndarray, col: int) -> int | None:
    """Devuelve la fila donde caeria una ficha en `col`, o None si esta llena."""
    for r in reversed(range(ROWS)):
        if board[r, col] == 0:
            return r
    return None


def _is_four_in_row(board: np.ndarray, r: int, c: int, player: int) -> bool:
    """Verifica si colocando `player` en (r, c) cierra un 4-en-linea.

    Asume que (r, c) es la posicion donde cae la ficha tras un drop legal.
    Chequea las 4 direcciones (horizontal, vertical, diag /, diag \\).
    """
    # Para no mutar el tablero, evaluamos como si la ficha ya estuviera ahi.
    # Direcciones: (dr, dc)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1  # cuenta la ficha hipotetica en (r, c)
        # Hacia adelante
        rr, cc = r + dr, c + dc
        while 0 <= rr < ROWS and 0 <= cc < COLS and board[rr, cc] == player:
            count += 1
            rr += dr
            cc += dc
        # Hacia atras
        rr, cc = r - dr, c - dc
        while 0 <= rr < ROWS and 0 <= cc < COLS and board[rr, cc] == player:
            count += 1
            rr -= dr
            cc -= dc
        if count >= 4:
            return True
    return False


def find_immediate_winner(board: np.ndarray, player: int) -> int | None:
    """Devuelve la columna donde `player` puede ganar de inmediato, o None.

    Si hay varias columnas ganadoras, devuelve la primera encontrada
    (orden de columnas 0..6).
    """
    for c in range(COLS):
        r = _drop_row(board, c)
        if r is None:
            continue
        if _is_four_in_row(board, r, c, player):
            return c
    return None


# Orden de preferencia centro -> bordes
CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6]


def center_preference(valid: list[int]) -> int:
    """Devuelve la columna mas centrada entre las validas."""
    for c in CENTER_ORDER:
        if c in valid:
            return c
    raise ValueError("Sin columnas validas")
