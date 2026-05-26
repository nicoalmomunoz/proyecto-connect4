# =============================================================================
# Agente Connect-4 - Minimax con evaluacion heuristica
# Nicolas Almonacid Munoz
# Fundamentos de Inteligencia Artificial - 2026.1
# Universidad de La Sabana
# =============================================================================


import time

import numpy as np

from connect4.policy import Policy



_ROWS = 6
_COLS = 7
_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6]


_MIN_DEPTH = 4
_MAX_DEPTH = 14


_DEFAULT_TIME_PER_MOVE = 1.0
_MAX_TIME_PER_MOVE = 50.0

_TT_EXACT = 0
_TT_LOWER = 1
_TT_UPPER = 2

_WIN = 100000.0
_LOSS = -100000.0
_DRAW = 0.0


class _TimeUp(Exception):
    pass



def _drop_row(board, col):
    for r in reversed(range(_ROWS)):
        if board[r, col] == 0:
            return r
    return None


def _apply_move(board, col, player):
    new_board = board.copy()
    for r in reversed(range(_ROWS)):
        if new_board[r, col] == 0:
            new_board[r, col] = player
            return new_board, r
    raise ValueError(f"Columna {col} llena")


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
    for c in _CENTER_ORDER:
        if c in valid:
            return c
    raise ValueError("Sin columnas validas")




def _score_window(my_count, opp_count):

    if my_count > 0 and opp_count > 0:
        return 0
    if my_count == 4:
        return 1000
    if my_count == 3:
        return 50
    if my_count == 2:
        return 5
    if opp_count == 4:
        return -1000
    if opp_count == 3:
        return -80
    if opp_count == 2:
        return -4
    return 0


def _evaluate(board, my_player):

    score = 0
    opp = -my_player

    center_col = board[:, 3]
    score += 6 * int(np.sum(center_col == my_player))
    score -= 6 * int(np.sum(center_col == opp))

    for r in range(_ROWS):
        for c in range(_COLS - 3):
            my_n = 0
            opp_n = 0
            for i in range(4):
                v = board[r, c + i]
                if v == my_player:
                    my_n += 1
                elif v == opp:
                    opp_n += 1
            score += _score_window(my_n, opp_n)

    for c in range(_COLS):
        for r in range(_ROWS - 3):
            my_n = 0
            opp_n = 0
            for i in range(4):
                v = board[r + i, c]
                if v == my_player:
                    my_n += 1
                elif v == opp:
                    opp_n += 1
            score += _score_window(my_n, opp_n)

    for r in range(_ROWS - 3):
        for c in range(_COLS - 3):
            my_n = 0
            opp_n = 0
            for i in range(4):
                v = board[r + i, c + i]
                if v == my_player:
                    my_n += 1
                elif v == opp:
                    opp_n += 1
            score += _score_window(my_n, opp_n)

    for r in range(_ROWS - 3):
        for c in range(3, _COLS):
            my_n = 0
            opp_n = 0
            for i in range(4):
                v = board[r + i, c - i]
                if v == my_player:
                    my_n += 1
                elif v == opp:
                    opp_n += 1
            score += _score_window(my_n, opp_n)

    return score



class QLearningBipolarV2(Policy):


    def __init__(self):
        self.rng = None

        self._tt = {}
        self._deadline = float("inf")
        self._time_per_move = _DEFAULT_TIME_PER_MOVE

    def mount(self, *args, **kwargs):
        if self.rng is None:
            self.rng = np.random.default_rng()

        self._tt = {}

        timeout = args[0] if args else kwargs.get("timeout", None)
        if isinstance(timeout, (int, float)) and timeout > 0:
            self._time_per_move = min(float(timeout) * 0.8, _MAX_TIME_PER_MOVE)
        else:
            self._time_per_move = _DEFAULT_TIME_PER_MOVE



    def _minimax(self, board, current_player, my_player, depth, alpha, beta):

        if time.time() > self._deadline:
            raise _TimeUp

        tt_key = (board.tobytes(), current_player)
        tt = self._tt.get(tt_key)
        if tt is not None and tt[0] >= depth:
            tt_score, tt_move, tt_flag = tt[1], tt[2], tt[3]
            if tt_flag == _TT_EXACT:
                return tt_score, tt_move
            elif tt_flag == _TT_LOWER:
                if tt_score >= beta:
                    return tt_score, tt_move
                alpha = max(alpha, tt_score)
            else: 
                if tt_score <= alpha:
                    return tt_score, tt_move
                beta = min(beta, tt_score)

        valid = _free_cols(board)
        if not valid:
            return _DRAW, None

        for c in valid:
            r = _drop_row(board, c)
            if r is not None and _is_four_in_row(board, r, c, current_player):
                score = _WIN if current_player == my_player else _LOSS
                self._tt[tt_key] = (depth, score, c, _TT_EXACT)
                return score, c

        if depth == 0:
            score = _evaluate(board, my_player)
            return score, None

        if tt is not None and tt[2] in valid:
            ordered = [tt[2]] + [c for c in valid if c != tt[2]]
        else:
            ordered = sorted(valid, key=lambda c: _CENTER_ORDER.index(c))

        alpha_orig = alpha
        beta_orig = beta
        best_move = ordered[0]

        if current_player == my_player:
            best_score = -float("inf")
            for c in ordered:
                new_board, _ = _apply_move(board, c, current_player)
                score, _ = self._minimax(
                    new_board, -current_player, my_player,
                    depth - 1, alpha, beta,
                )
                if score > best_score:
                    best_score = score
                    best_move = c
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break
        else:
            best_score = float("inf")
            for c in ordered:
                new_board, _ = _apply_move(board, c, current_player)
                score, _ = self._minimax(
                    new_board, -current_player, my_player,
                    depth - 1, alpha, beta,
                )
                if score < best_score:
                    best_score = score
                    best_move = c
                beta = min(beta, best_score)
                if alpha >= beta:
                    break
        if best_score <= alpha_orig:
            flag = _TT_UPPER
        elif best_score >= beta_orig:
            flag = _TT_LOWER
        else:
            flag = _TT_EXACT
        self._tt[tt_key] = (depth, best_score, best_move, flag)

        return best_score, best_move


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

        self._deadline = time.time() + self._time_per_move
        best_move = _center_preference(valid)
        best_score = 0.0

        for depth in range(_MIN_DEPTH, _MAX_DEPTH + 1):
            try:
                score, move = self._minimax(
                    s, player, player, depth,
                    -float("inf"), float("inf"),
                )
                if move is not None and move in valid:
                    best_move = move
                    best_score = score
                if abs(best_score) >= _WIN / 2:
                    break
            except _TimeUp:
                break

        return best_move
