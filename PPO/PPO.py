import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

import xpc


def scale_actions_to_correct(actions):
    actions[3] = actions[3] - 1
    actions[4] = max(actions[4], 0)
    return actions


def to_probs(action):
    return (torch.sigmoid(action) + 1) / 2

def calc_adv(reward, critic_value):
    return reward - critic_value


class PPO:
    def __init__(self, policy_class, env):
        # Enviroment info

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Init hyperparameters
        self.lr = 0.001
        self.clip_range = 0.2

        # Init actor and critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim)
        self.critic = policy_class(self.obs_dim, 1)

        # Init optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Collect previous exp
        self.states, self.actions, self.rewards, self.old_policy_probs = self.collect_experience(self.env, self.actor,
                                                                                                 num_epochs=5,
                                                                                                 num_steps=100000)

        self.critic_values = []

    def ppo_loss(self, old_policy_probs, new_policy_probs, advantages, clip_range):
        ratio = new_policy_probs / old_policy_probs
        clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        return -torch.min(surrogate1, surrogate2).mean()

    def collect_experience(self, env, policy_network, num_epochs, num_steps):
        states = []
        actions = []
        rewards = []
        old_policy_probs = []

        for epoch in range(num_epochs):
            state = env.reset()
            done = False
            step = 0
            while not done and step < num_steps:
                try:
                    print(xpc.XPlaneConnect().getDREFs(
                        ['sim/flightmodel/controls/elv_trim', 'sim/flightmodel/controls/ail_trim',
                         'sim/flightmodel/controls/rud_trim', 'sim/flightmodel/engine/ENGN_thro_override',
                         'sim/flightmodel/controls/parkbrake']))

                    with torch.no_grad():
                        action = policy_network(torch.tensor(state, dtype=torch.float))
                        action_probs = to_probs(action)

                    action = np.squeeze(action.numpy())
                    next_state, reward, done, _ = self.env.step(scale_actions_to_correct(action))

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    old_policy_probs.append(action_probs)

                    state = next_state
                except:
                    time.sleep(10)
                    break
        return states, actions, rewards, old_policy_probs

    def train_critic(self, num_critic_epochs):
        for epoch in range(num_critic_epochs):
            print(f'Number epoch critic = {epoch}')
            for i in range(len(self.states)):
                state = self.states[i]
                expected_value = self.critic(torch.tensor(state, dtype=torch.float))
                target = torch.tensor([self.rewards[i]], dtype=torch.float)

                critic_loss = F.mse_loss(expected_value, target)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

    def train_actor(self, num_epoch, num_steps, clip_range):
        for epoch in range(num_epoch):
            state = self.env.reset()
            done = False
            old_policy_probs = torch.tensor(np.array(self.old_policy_probs[:5]), dtype=torch.float)
            while not done:
                try:
                    print(xpc.XPlaneConnect().getDREFs(
                        ['sim/flightmodel/controls/elv_trim', 'sim/flightmodel/controls/ail_trim',
                         'sim/flightmodel/controls/rud_trim', 'sim/flightmodel/engine/ENGN_thro_override',
                         'sim/flightmodel/controls/parkbrake']))

                    new_actions = self.actor(torch.tensor(state, dtype=torch.float))
                    new_actions = np.squeeze(new_actions.detach().numpy())
                    new_state, reward, done, _ = self.env.step(scale_actions_to_correct(new_actions))

                    new_policy_probs = to_probs(torch.tensor(new_actions, dtype=torch.float))
                    ppo_loss = self.ppo_loss(old_policy_probs, new_policy_probs, calc_adv(reward, self.critic(torch.tensor(state, dtype=torch.float))),
                                             self.clip_range)
                    old_policy_probs = new_policy_probs

                    self.actor_optim.zero_grad()
                    actor_loss = -ppo_loss
                    actor_loss.backward()
                    self.actor_optim.step()

                    state = new_state
                except:
                    time.sleep(3)
                    break
