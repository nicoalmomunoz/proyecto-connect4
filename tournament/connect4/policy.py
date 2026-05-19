import numpy as np
from abc import ABC, abstractmethod


class Policy(ABC):

    @abstractmethod
    def mount(self, timeout: float | None = None) -> None:
        pass

    @abstractmethod
    def act(self, s: np.ndarray) -> int:
        pass
