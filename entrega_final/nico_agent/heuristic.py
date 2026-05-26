

from __future__ import annotations

import numpy as np


COLS = 7
ROWS = 6


def _drop_row(board: np.ndarray, col: int) -> int | None:

    for r in reversed(range(ROWS)):
        if board[r, col] == 0:
            return r
    return None


def _is_four_in_row(board: np.ndarray, r: int, c: int, player: int) -> bool:

    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1  
        rr, cc = r + dr, c + dc
        while 0 <= rr < ROWS and 0 <= cc < COLS and board[rr, cc] == player:
            count += 1
            rr += dr
            cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < ROWS and 0 <= cc < COLS and board[rr, cc] == player:
            count += 1
            rr -= dr
            cc -= dc
        if count >= 4:
            return True
    return False


def find_immediate_winner(board: np.ndarray, player: int) -> int | None:

    for c in range(COLS):
        r = _drop_row(board, c)
        if r is None:
            continue
        if _is_four_in_row(board, r, c, player):
            return c
    return None

CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6]


def center_preference(valid: list[int]) -> int:
    for c in CENTER_ORDER:
        if c in valid:
            return c
    raise ValueError("Sin columnas validas")
