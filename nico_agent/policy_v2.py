
# Agente Connect-4 Q-Learning Bipolar.
# Nicolas Almonacid Muñoz 
# Fundamentos de Inteligencia Artificial 
# 2026.1
#Universidada de la Sabana



import numpy as np
from connect4.policy import Policy

_ROWS=6
_COLS=7
_DIRECTIONS=[(0, 1), (1, 0), (1, 1), (1, -1)]
_CENTER_ORDER=[3, 2, 4, 1, 5, 0, 6]


def _drop_row(board, col):

    for r in reversed(range(_ROWS)):
        if board[r, col] == 0:
            return r
    return None


def _is_four_in_row(board, r, c, player):
    for dr, dc in _DIRECTIONS:
        count = 1
        rr, cc = r + dr, c + dc
        while 0 <= rr < _ROWS and 0 <= cc < _COLS and board[rr, cc] == player:
            count += 1
            rr += dr
            cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < _ROWS and 0 <= cc < _COLS and board[rr, cc] == player:
            count += 1
            rr -= dr
            cc -= dc
        if count >= 4:
            return True
    return False


def _find_immediate_winner(board, player):

    for c in range(_COLS):
        r = _drop_row(board, c)
        if r is None:
            continue
        if _is_four_in_row(board, r, c, player):
            return c
    return None


def _free_cols(board):
    return [c for c in range(_COLS) if board[0, c] == 0]


def _infer_player(board):

    red = int(np.sum(board == -1))
    yellow = int(np.sum(board == 1))
    return -1 if red == yellow else 1


def _center_preference(valid):
    """Devuelve la columna mas centrada entre las validas."""
    for c in _CENTER_ORDER:
        if c in valid:
            return c
    raise ValueError("Sin columnas validas")


class QLearningBipolarV2(Policy):


    def __init__(self):
        self.rng = None

    def mount(self, *args, **kwargs):

        self.rng = np.random.default_rng()

    def act(self, s):

        if self.rng is None:
            self.rng = np.random.default_rng()

        valid = _free_cols(s)
        player = _infer_player(s)

        win = _find_immediate_winner(s, player)
        if win is not None and win in valid:
            return win
        
        block = _find_immediate_winner(s, -player)
        if block is not None and block in valid:
            return block

        return _center_preference(valid)
