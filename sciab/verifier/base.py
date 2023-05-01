from abc import ABCMeta, abstractmethod
from ..controller.base import Controller


class VerifierResult(metaclass=ABCMeta):
    """VerifierOutput struct"""
    verified: bool = False


class Verifier(metaclass=ABCMeta):
    """Trainer samples a trajectory and updates the controller"""

    @abstractmethod
    def verify(self, controller: Controller) -> VerifierResult:
        pass
