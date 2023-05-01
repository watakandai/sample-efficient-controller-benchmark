from typing import List, Dict
from .base import Verifier, VerifierResult
from ..trainer.base import Controller
from ..env.base import Env
from .. import SimStatus


class ProbabilisticVerifierResult(VerifierResult):
    trajectories: List[Dict]
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ProbabilisticVerifier(Verifier):
    def __init__(self, env: Env, numSample: int):
        super().__init__()
        self.env = env
        self.numSample = numSample

    def verify(self, controller: Controller) -> ProbabilisticVerifierResult:

        numSample = self.numSample
        trajectories = map(lambda: self.simulate(controller), range(numSample))

        safeTraj = filter(lambda t: t["status"]==SimStatus.SIM_TERMINATED, trajectories)
        verified = len(safeTraj) / len(trajectories) == 1

        return ProbabilisticVerifierResult(
            verified=verified,
            trajectories=trajectories)

    def simulate(self, controller) -> Dict:

        x = self.env.reset()
        X = [x]
        d = False

        while not d:

            u = controller.action(x)
            nx, r, d, info = self.env.step(u)
            X.append(nx)
            x = nx

        return {"X": X, "status": info["status"]}
