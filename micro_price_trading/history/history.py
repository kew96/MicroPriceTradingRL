from abc import ABC, abstractmethod

import numpy as np


class History(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.readable_action_space = self._generate_readable_action_space(*args, **kwargs)

    @property
    @abstractmethod
    def portfolio_history(self) -> np.array:
        raise NotImplementedError

    @property
    @abstractmethod
    def share_history(self) -> np.array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _generate_readable_action_space(*args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _update_history(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _reset_history(self, *args, **kwargs):
        raise NotImplementedError
