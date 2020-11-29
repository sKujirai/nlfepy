import logging
from logging import getLogger
from abc import ABCMeta, abstractmethod


class IntegralEquation(metaclass=ABCMeta):
    """
    Base class to solve integral equation
    """

    def __init__(self, *, mesh, val=None, params: dict = {}) -> None:

        self._mesh = mesh
        self._val = val

        self._config = {}
        self._config['penalty_coefficient'] = 1.e8
        self._config['logging'] = False

        for key, value in params.items():
            self._config[key] = value

        self._logger = getLogger('ItgEqn')
        ch = logging.StreamHandler()
        if self._config['logging']:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.WARNING)
        self._logger.addHandler(ch)

    @abstractmethod
    def solve(self) -> None:
        pass
