

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np




def canonicalize(board: np.ndarray, player: int) -> np.ndarray:

    return (board * player).astype(np.int8)


def state_key(canonical_board: np.ndarray) -> bytes:

    return canonical_board.tobytes()


def free_cols(board: np.ndarray) -> list[int]:

    return [c for c in range(7) if board[0, c] == 0]


def infer_player(board: np.ndarray) -> int:

    red_count = int(np.sum(board == -1))
    yellow_count = int(np.sum(board == 1))
    return -1 if red_count == yellow_count else 1




class QTable:


    def __init__(self) -> None:
        self.q: dict[bytes, np.ndarray] = {} 
        self.n: dict[bytes, np.ndarray] = {}  



    def __len__(self) -> int:
        return len(self.q)

    def __contains__(self, key: bytes) -> bool:
        return key in self.q

    def total_visits(self) -> int:
        return int(sum(arr.sum() for arr in self.n.values()))



    def get_or_create(self, key: bytes) -> tuple[np.ndarray, np.ndarray]:

        arr_q = self.q.get(key)
        if arr_q is None:
            arr_q = np.zeros(7, dtype=np.float32)
            arr_n = np.zeros(7, dtype=np.int32)
            self.q[key] = arr_q
            self.n[key] = arr_n
            return arr_q, arr_n
        return arr_q, self.n[key]



    def greedy(
        self,
        key: bytes,
        valid: list[int],
        rng: np.random.Generator,
    ) -> int:

        if not valid:
            raise ValueError("Sin columnas validas, juego no deberia continuar.")

        if key not in self.q:
            return int(rng.choice(valid))

        q_vals = self.q[key]
        best_q = max(q_vals[c] for c in valid)
        ties = [c for c in valid if q_vals[c] == best_q]
        return int(rng.choice(ties))

  

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
