from abc import ABC, abstractmethod

import numpy as np




class History(ABC):

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def portfolio_history(self) -> np.array:
        raise NotImplementedError

    @property
    @abstractmethod
    def share_history(self) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def _update_history(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _reset_history(self, *args, **kwargs):
        raise NotImplementedError
