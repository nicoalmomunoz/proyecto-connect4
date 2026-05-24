import time
import math
import numpy as np
from connect4.connect_state import ConnectState
from connect4.policy import Policy

class MCTSAgentV1(Policy):
    def __init__(self, time_limit: float = 0.05, rng_seed: int | None = None, c_exploration: float = 1.414):
        self.time_limit = time_limit
        self.rng = np.random.default_rng(rng_seed)
        self.c = c_exploration

    def mount(self, timeout: float | None = None) -> None:
        if timeout is not None:
            # Margen de seguridad muy estricto para el autocalificador
            self.time_limit = max(0.001, float(timeout) * 0.5)
        return None

    def act(self, s: np.ndarray) -> int:
        red_count = int(np.sum(s == -1))
        yellow_count = int(np.sum(s == 1))
        current_player = -1 if red_count == yellow_count else 1

        root_state = ConnectState(board=s, player=current_player)
        stats: dict[bytes, dict[int, list[float]]] = {}

        start = time.perf_counter()
        iterations = 0
        
        # OPTIMIZACIÓN: Límite híbrido (tiempo + tope de iteraciones)
        while time.perf_counter() - start < self.time_limit and iterations < 150:
            self._simulate(root_state, stats)
            iterations += 1

        board_key = s.tobytes()
        if board_key not in stats:
            available = root_state.get_free_cols()
            return int(self.rng.choice(available))

        action_stats = stats[board_key]
        best_action = max(action_stats, key=lambda a: action_stats[a][1])
        return best_action

    def _simulate(self, root_state: ConnectState, stats: dict):
        path: list[tuple[bytes, int, int]] = []
        state = root_state

        while not state.is_final():
            board_key = state.board.tobytes()
            available = state.get_free_cols()

            if board_key not in stats:
                stats[board_key] = {a: [0.0, 0.0] for a in available}
                action = int(self.rng.choice(available))
                path.append((board_key, action, state.player))
                state = state.transition(action)
                break
            else:
                action_stats = stats[board_key]
                total_visits = sum(info[1] for info in action_stats.values())

                best_action = None
                best_uct = -float('inf')

                for a in available:
                    w_i, n_i = action_stats[a]
                    if n_i == 0:
                        uct = float('inf')
                    else:
                        # OPTIMIZACIÓN: math en lugar de np para escalares es mucho más rápido
                        uct = (w_i / n_i) + self.c * math.sqrt(math.log(total_visits) / n_i)
                    
                    if uct > best_uct:
                        best_uct = uct
                        best_action = a

                path.append((board_key, best_action, state.player))
                state = state.transition(best_action)

        winner = self._rollout(state)

        for (board_key, action, player_who_moved) in path:
            stats[board_key][action][1] += 1
            if winner == 0:
                stats[board_key][action][0] += 0.5
            elif winner == player_who_moved:
                stats[board_key][action][0] += 1.0

    def _rollout(self, state: ConnectState) -> int:
        st = ConnectState(board=state.board, player=state.player)
        while not st.is_final():
            available = st.get_free_cols()
            action = int(self.rng.choice(available))
            st = st.transition(action)
        return st.get_winner()