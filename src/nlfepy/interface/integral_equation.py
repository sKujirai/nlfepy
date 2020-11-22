import logging
from logging import getLogger
from abc import ABCMeta, abstractmethod


class IntegralEquation(metaclass=ABCMeta):
    """
    Base class to solve integral equation
    """

    def __init__(self, params: dict = {}) -> None:

        self.config = {}
        self.config['penalty_coefficient'] = 1.e8
        self.config['logging'] = False

        for key, value in params.items():
            self.config[key] = value

        self.logger = getLogger('ItgEqn')
        ch = logging.StreamHandler()
        if self.config['logging']:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)
        self.logger.addHandler(ch)

    @abstractmethod
    def solve(self) -> None:
        pass
