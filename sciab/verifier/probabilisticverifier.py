import os
import copy
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
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
    def __init__(self, env: Env, numSample: int, filedir:str=None):
        super().__init__()
        self.env = copy.deepcopy(env)
        self.numSample = numSample
        self.filedir = filedir
        if self.filedir is not None:
            self.fig = plt.figure()
            self.ax1 = self.fig.add_subplot(121)
            self.ax2 = self.fig.add_subplot(122)
        self.counter = 0

    def verify(self, controller: Controller) -> ProbabilisticVerifierResult:

        numSample = self.numSample

        # t = ThreadPool(processes=16)
        # f = lambda i : self.simulate(controller, i)
        # rs = t.map(f, range(numSample))
        # t.close()

        # trajectories = list(rs)
        trajectories = [self.simulate(controller, i) for i in range(numSample)]

        safeTraj = list(filter(lambda t: t["status"]==SimStatus.SIM_TERMINATED, trajectories))
        safeTrajProb = len(safeTraj) / len(trajectories)
        verified = safeTrajProb == 1

        msg = f"safeTrajProb: {safeTrajProb}"
        logging.info(msg)
        print(msg)

        if self.filedir is not None:
            for t in safeTraj:
                self.ax1.plot(list(map(lambda x: x[0], t["X"])),
                             list(map(lambda x: x[1], t["X"])))
            unsafeTraj = list(filter(lambda t: t["status"]!=SimStatus.SIM_TERMINATED, trajectories))
            for t in unsafeTraj:
                self.ax2.plot(list(map(lambda x: x[0], t["X"])),
                             list(map(lambda x: x[1], t["X"])))
            filepath = os.path.join(self.filedir, f'{self.env.system}/ProbabilisticVerifier{self.counter}.png')
            Path(filepath).parents[0].mkdir(parents=True, exist_ok=True)
            self.fig.savefig(filepath)

            self.counter += 1

        return ProbabilisticVerifierResult(
            verified=verified,
            trajectories=trajectories)

    def simulate(self, controller, iter=None) -> Dict:

        x = self.env.reset()
        X = [x]
        d = False
        counter = 0

        while not d:
            u = controller.action(x) # In our case, it will either return an int or List[float]
            nx, r, done, truncated, info = self.env.step(u)
            d = done or truncated
            X.append(nx)
            x = nx
            counter += 1
            if counter >= self.env.spec.max_episode_steps:
                break

        # if info['status'] != SimStatus.SIM_TERMINATED:
        #     print(f"iteration={iter}, status={info['status']}, start={X[0]}, end={X[-1]}\n{np.array(X)}")

        return {"X": X, "status": info["status"]}
