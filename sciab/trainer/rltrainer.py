import numpy as np
from datetime import datetime
from .base import Trainer
from ..countersampler.base import CounterExample
from ..controller.rlcontroller import RLController
from ..controller.replaybuffer import ReplayBuffer


# This is more like a TD3 Trainer. I don't care for now.
class RLTrainer(Trainer):
    def __init__(self, env,
                 startTrainingEpisode: int=5,
                 modelPath: str="model/rl",
                 date: str=datetime.today().strftime('%Y-%m-%d'),
                 bufferSize: int=int(1e6),
                 batchSize: int=100):
        super().__init__(env)
        self.env = env
        self.counter = 0
        self.startTrainingEpisode = startTrainingEpisode

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        scale = max_action * np.ones(action_dim)

        self.controller = RLController(
            state_dim,
            action_dim,
            scale,
            '_'.join([modelPath, date]))

        self.replay_buffer = ReplayBuffer(
            state_dim,
            action_dim,
            buffer_size=bufferSize,
            batch_size=batchSize)

    def train(self, counterexample: CounterExample) -> RLController:
        """A standard RL that doesn't depend on the counterexample"""

        s = self.env.reset()
        done = False

        while not done:

            if self.counter < self.startTrainingEpisode:
                a = self.env.action_space.sample()
            else:
                a = self.controller.action(s)

            ns, r, d, t, info = self.env.step(a)
            done = d or t
            self.replay_buffer.append(s, a, ns, r, done)

            if self.counter < self.startTrainingEpisode:
                self.controller.update(self.replay_buffer)

        self.counter += 1
        return self.controller
