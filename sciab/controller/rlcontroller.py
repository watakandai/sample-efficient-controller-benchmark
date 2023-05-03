import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Controller

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def var(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def get_tensor(z):
    if len(z.shape) == 1:
        return var(torch.FloatTensor(z.copy())).unsqueeze(0)
    else:
        return var(torch.FloatTensor(z.copy()))


class RLController(Controller):
    def __init__(
            self,
            state_dim,
            action_dim,
            scale,
            model_path,
            actor_lr=0.0001,
            critic_lr=0.001,
            expl_noise=0.1,
            policy_noise=0.2,
            noise_clip=0.5,
            gamma=0.99,
            policy_freq=2,
            tau=0.005):
        super().__init__()

        self.scale = scale
        self.model_path = model_path

        # parameters
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_freq = policy_freq
        self.tau = tau

        self.actor = TD3Actor(state_dim, action_dim, scale=scale).to(device)
        self.actor_target = TD3Actor(state_dim, action_dim, scale=scale).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic1 = TD3Critic(state_dim, action_dim).to(device)
        self.critic2 = TD3Critic(state_dim, action_dim).to(device)
        self.critic1_target = TD3Critic(state_dim, action_dim).to(device)
        self.critic2_target = TD3Critic(state_dim, action_dim).to(device)

        self.critic1_optimizer = torch.optim.Adam(
                self.critic1.parameters(),
                lr=critic_lr, weight_decay=0.0001
        )
        self.critic2_optimizer = torch.optim.Adam(
                self.critic2.parameters(),
                lr=critic_lr, weight_decay=0.0001
        )

        self._initialize_target_networks()

        self._initialized = False
        self.total_it = 0

    def _initialize_target_networks(self):
        self._update_target_network(self.critic1_target, self.critic1, 1.0)
        self._update_target_network(self.critic2_target, self.critic2, 1.0)
        self._update_target_network(self.actor_target, self.actor, 1.0)
        self._initialized = True

    def _update_target_network(self, target, origin, tau):
        for target_param, origin_param in zip(target.parameters(), origin.parameters()):
            target_param.data.copy_(tau * origin_param.data + (1.0 - tau) * target_param.data)

    def save(self, timestep=None):
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.mkdir(os.path.dirname(self.model_path))

        if timestep:
            model_path = self.model_path + '_%i'%(timestep)
        else:
            model_path = self.model_path

        torch.save(self.actor.state_dict(), model_path+"_actor.h5")
        torch.save(self.actor_optimizer.state_dict(), model_path+"_actor_optimizer.h5")
        torch.save(self.critic1.state_dict(), model_path+"_critic1.h5")
        torch.save(self.critic2.state_dict(), model_path+"_critic2.h5")
        torch.save(self.critic1_optimizer.state_dict(), model_path+"_critic1_optimizer.h5")
        torch.save(self.critic2_optimizer.state_dict(), model_path+"_critic2_optimizer.h5")

    def load(self, timestep=-1):
        if timestep>0:
            model_path = self.model_path + '_%i'%(timestep)
        else:
            model_path = self.model_path

        self.actor.load_state_dict(torch.load(model_path+"_actor.h5"))
        self.actor_optimizer.load_state_dict(torch.load(model_path+"_actor_optimizer.h5"))
        self.critic1.load_state_dict(torch.load(model_path+"_critic1.h5"))
        self.critic2.load_state_dict(torch.load(model_path+"_critic2.h5"))
        self.critic1_optimizer.load_state_dict(torch.load(model_path+"_critic1_optimizer.h5"))
        self.critic2_optimizer.load_state_dict(torch.load(model_path+"_critic2_optimizer.h5"))

    def update(self, replay_buffer, iterations=1):
        if not self._initialized:
            self._initialize_target_networks()

        self.total_it += 1

        states, actions, n_states, rewards, not_done = replay_buffer.sample()

        with torch.no_grad():
            noise = (
                torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            #n_actions = (
            #    self.actor_target(states) + noise
            #).clamp(-self.scale[0], self.scale[0])

            n_actions = self.actor_target(states) + noise
            n_actions = torch.min(n_actions,  self.actor.scale)
            n_actions = torch.max(n_actions, -self.actor.scale)

            target_Q1 = self.critic1_target(n_states, n_actions)
            target_Q2 = self.critic2_target(n_states, n_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + not_done * self.gamma * target_Q
            target_Q_detached = target_Q.detach()

        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(current_Q1, target_Q_detached)
        critic2_loss = F.mse_loss(current_Q2, target_Q_detached)
        critic_loss = critic1_loss + critic2_loss

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            a = self.actor(states)
            Q1 = self.critic1(states, a)
            actor_loss = -Q1.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._update_target_network(self.critic1_target, self.critic1, self.tau)
            self._update_target_network(self.critic2_target, self.critic2, self.tau)
            self._update_target_network(self.actor_target, self.actor, self.tau)

            return {'actor_loss': actor_loss, 'critic_loss': critic_loss}

        return {'critic_loss': critic_loss}

    def action(self, state, to_numpy=True):
        state = get_tensor(state)
        action = self.actor(state)

        if to_numpy:
            # return action.cpu().data.numpy().squeeze()
            return action.cpu().data.numpy()

        # return action.squeeze()
        return action

    def action_with_noise(self, state, to_numpy=True):
        state = get_tensor(state)
        action = self.actor(state)

        action = action + self._sample_exploration_noise(action) # actions.shape
        action = torch.min(action,  self.actor.scale)
        action = torch.max(action, -self.actor.scale)

        if to_numpy:
            # return action.cpu().data.numpy().squeeze()
            return action.cpu().data.numpy()

        # return action.squeeze()
        return action

    def _sample_exploration_noise(self, actions): # shape
        scale = self.expl_noise * self.actor.scale
        mean = torch.zeros(actions.size()).to(device)
        var = torch.ones(actions.size()).to(device)
        return torch.normal(mean, scale*var)

        # return self._exploration_noise.sample(sample_shape=shape)

    def evaluate_policy(self, env, eval_episodes=10, render=False, save_video=False, sleep=-1):
        if save_video:
            from OpenGL import GL
            env = gym.wrappers.Monitor(env, directory='video',
                                    write_upon_reset=True, force=True, resume=True, mode='evaluation')
            render = False

        rewards = []
        for e in range(eval_episodes):
            s = env.reset()
            done = False
            steps = 1
            reward_episode_sum = 0

            while not done:
                if render:
                    env.render()
                if sleep>0:
                    time.sleep(sleep)
                a = self.policy(s)
                n_s, r, done, _ = env.step(a)

                s = n_s
                reward_episode_sum += r
                steps += 1
            else:
                print("Rewards in Episode %i: %.2f"%(e, reward_episode_sum/steps))
                rewards.append(reward_episode_sum)

        return np.array(rewards)


class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TD3Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q


class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, scale):
        super(TD3Actor, self).__init__()
        if scale is None:
            scale = torch.ones(state_dim)
        else:
            scale = get_tensor(scale)
        self.scale = nn.Parameter(scale.clone().detach(), requires_grad=False)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.scale * torch.tanh(self.l3(a))
