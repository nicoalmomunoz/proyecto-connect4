

import numpy as np

from connect4.policy import Policy


class RandomPolicy(Policy):


    def mount(self) -> None:
 
        self.rng = np.random.default_rng()

    def act(self, s: np.ndarray) -> int:

        available = [c for c in range(7) if s[0, c] == 0]
        return int(self.rng.choice(available))
