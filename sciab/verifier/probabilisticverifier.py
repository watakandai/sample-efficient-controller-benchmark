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
        # with Pool(16) as p:
        #     trajectories = p.map(f, range(numSample))
        # trajectories = [self.simulate(controller, i) for i in range(numSample)]

        safeTraj = list(filter(lambda t: t["status"]==SimStatus.SIM_TERMINATED, trajectories))
        verified = len(safeTraj) / len(trajectories) == 1

        return ProbabilisticVerifierResult(
            verified=verified,
            trajectories=trajectories)

    def simulate(self, controller, iter=None) -> Dict:

        x = self.env.reset()
        X = [x]
        d = False

        while not d:
            assert(len(x) == 3), info["output"]
            u = controller.action(x) # In our case, it will either return an int or List[float]
            nx, r, done, truncated, info = self.env.step(u)
            d = done or truncated
            X.append(nx)
            x = nx

        print(f"iteration={iter}, status={info['status']}, start={X[0]}, end={X[-1]}")

        return {"X": X, "status": info["status"]}
