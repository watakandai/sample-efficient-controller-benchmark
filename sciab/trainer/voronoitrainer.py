import os
from pathlib import Path
import matplotlib.pyplot as plt
from .base import Trainer
from ..countersampler.base import CounterExample
from ..controller.voronoicontroller import VoronoiController
from ..env.omplenv import OMPLEnv



class VoronoiTrainer(Trainer):
    def __init__(self, env: OMPLEnv, filedir:str=None, maxTry:int=20):
        super().__init__(env)
        self.controller = VoronoiController()
        self.filedir = filedir
        self.maxTry = maxTry

        xmin = self.env.safeLowerBound[0]
        xmax = self.env.safeUpperBound[0]
        ymin = self.env.safeLowerBound[1]
        ymax = self.env.safeUpperBound[1]

        if self.filedir is not None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.axis([xmin, xmax, ymin, ymax])
            self.counter = 0
        self.Xs = []

    def train(self, counterexample: CounterExample) -> VoronoiController:
        if len(self.controller.X) == 0:
            X = []
            count = 0
            while len(X) == 0:
                if count == self.maxTry:
                    break
                X, A, U, Dt = self.env.sampleBestCostTrajectory(counterexample.x)
                count += 1
        else:
            X, A, U, Dt = self.env.sampleBestCostTrajectory(counterexample.x)

        self.Xs.append(X)
        self.controller.update(X, U, Dt)

        if self.filedir is not None:
            self.plotCEs()

        return self.controller

    def plotCEs(self):
        # for (x, nx) in zip(self.controller.X, self.controller.nX):
        #     self.ax.arrow(x[0], x[1], nx[0] - x[0], nx[1] - x[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
        for X in self.Xs:
            self.ax.plot(list(map(lambda x: x[0], X)),
                         list(map(lambda x: x[1], X)))

        filepath = os.path.join(self.filedir, f'{self.env.system}/VoronoiTrainer{self.counter}.png')
        Path(filepath).parents[0].mkdir(parents=True, exist_ok=True)
        self.fig.savefig(filepath)
        self.counter += 1
        # plt.draw()
        # plt.pause(0.01)

