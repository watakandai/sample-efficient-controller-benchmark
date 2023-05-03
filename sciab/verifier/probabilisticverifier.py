import numpy as np
from multiprocessing.pool import ThreadPool
from typing import List, Dict
from .base import Verifier, VerifierResult
from ..trainer.base import Controller
from ..env.base import Env
from .. import SimStatus


class ProbabilisticVerifierResult(VerifierResult):
    trajectories: List[Dict] = None
    def __init__(self, verified, trajectories):
        self.verified = verified
        self.trajectories = trajectories


class ProbabilisticVerifier(Verifier):
    def __init__(self, env: Env, numSample: int):
        super().__init__()
        self.env = env
        self.numSample = numSample

    def verify(self, controller: Controller) -> ProbabilisticVerifierResult:

        numSample = self.numSample

        t = ThreadPool(processes=16)
        f = lambda i : self.simulate(controller, i)
        rs = t.map(f, range(numSample))
        t.close()

        trajectories = list(rs)
        safeTraj = list(filter(lambda t: t["status"]==SimStatus.SIM_TERMINATED, trajectories))
        safeTrajProb = len(safeTraj) / len(trajectories)
        verified = safeTrajProb == 1

        print("safeTrajProb: ", safeTrajProb)

        return ProbabilisticVerifierResult(
            verified=verified,
            trajectories=trajectories)

    def simulate(self, controller, iter=None) -> Dict:

        x = self.env.reset()
        X = [x]
        d = False

        while not d:
            u = controller.action(x) # In our case, it will either return an int or List[float]
            nx, r, done, truncated, info = self.env.step(u)
            d = done or truncated
            X.append(nx)
            x = nx

        # print(f"iteration={iter}, status={info['status']}, start={X[0]}, end={X[-1]}\n{np.array(X)}")

        return {"X": X, "status": info["status"]}
