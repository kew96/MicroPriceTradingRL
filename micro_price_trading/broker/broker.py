from abc import ABC, abstractmethod


class Broker(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def trade(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _reset_broker(self, *args, **kwargs):
        raise NotImplementedError
