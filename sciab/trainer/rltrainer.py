import logging
import numpy as np
from datetime import datetime
from .base import Trainer
from ..countersampler.base import CounterExample
from ..controller.rlcontroller import RLController
from ..controller.replaybuffer import ReplayBuffer
from .. import SimStatus


# This is more like a TD3 Trainer. I don't care for now.
class RLTrainer(Trainer):
    def __init__(self, env,
                 startTrainingEpisode: int=100,
                 bufferSize: int=int(1e6),
                 batchSize: int=1000):
        super().__init__(env)
        self.env = env
        self.counter = 0
        self.startTrainingEpisode = startTrainingEpisode

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        scale = max_action * np.ones(action_dim)
        # scale = env.action_space.high

        self.controller = RLController(
            state_dim,
            action_dim,
            scale,
            actor_lr=0.001,
            critic_lr=0.01,
            expl_noise=0.01,        # 1 %
            policy_noise=0.02)      # 2 %

        self.replay_buffer = ReplayBuffer(
            state_dim,
            action_dim,
            buffer_size=bufferSize,
            batch_size=batchSize)

    def train(self, counterexample: CounterExample) -> RLController:
        """A standard RL that doesn't depend on the counterexample"""

        s = self.env.reset()
        done = False
        reward = 0.0
        step = 0

        while not done:

            if self.counter < self.startTrainingEpisode:
                a = self.env.action_space.sample()
            else:
                a = self.controller.action_with_noise(s)

            ns, r, d, t, info = self.env.step(a)
            done = d # or t
            reward += r
            self.replay_buffer.append(s, a, ns, r, done)

            if info["status"] == SimStatus.SIM_TERMINATED:
                print(s, a, ns, r, d, t, info)
                logging.debug(f"{s}, {a}, {ns}, {r}, {d}, {t}, {info}")

            if self.counter > self.startTrainingEpisode:
                losses = self.controller.update(self.replay_buffer)
            step += 1
            s = ns

        if self.counter > self.startTrainingEpisode and \
            self.counter % 1 == 0:
            msg = f"Episode {self.counter}, TotalReward: {reward:.2f}, AveReward: {reward/step:.2f}, LastState: {ns}, Loss: {losses}"
            print(msg)
            logging.debug(msg)


        self.counter += 1
        return self.controller
