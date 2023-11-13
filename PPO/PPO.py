import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import xpc

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


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
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.act_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

        # Init optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Collect previous exp
        self.states, self.actions, self.rewards, self.old_policy_probs = self.collect_experience(self.env,
                                                                                                 num_epochs=200)

        self.critic_values = []

    def ppo_loss(self, old_policy_probs, new_policy_probs, advantages, clip_range):
        ratio = new_policy_probs / old_policy_probs
        clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        return -torch.min(surrogate1, surrogate2).mean()

    def collect_experience(self, env, num_epochs):
        states = []
        actions = []
        rewards = []
        old_policy_probs = []

        for epoch in range(num_epochs):
            state = env.reset()
            done = False
            step = 0
            policy_network = nn.Sequential(
                nn.Linear(self.obs_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, self.act_dim),
                nn.Tanh()
            )
            while not done and step <= 20000:
                try:
                    with torch.no_grad():
                        action = policy_network(torch.tensor(state, dtype=torch.float))
                        action_probs = to_probs(action)

                    scaled_action = scale_actions_to_correct(np.squeeze(action.numpy()))
                    next_state, reward, done, _ = self.env.step(scaled_action)

                    if step % 300 == 0:
                        print(f'Тангаж: {scaled_action[0]}, Крен: {scaled_action[1]}, Рысканье: {scaled_action[2]}, '
                              f'Сила тяги: {scaled_action[3]}, Тормоз: {scaled_action[4]}')

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    old_policy_probs.append(action_probs)

                    state = next_state
                    step += 1
                except Exception as e:
                    # Обработка ошибки и вывод сообщения
                    print(f"Произошла ошибка: {e}")
                    time.sleep(5)
                    break
        return states, actions, rewards, old_policy_probs

    def train_critic(self, num_critic_epochs):
        print('Started training critic')
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

    def train_actor(self, num_epoch):
        print('Started training actor')
        for epoch in range(num_epoch):
            state = self.env.reset()
            done = False
            step = 0
            old_policy_probs = torch.tensor(np.array(self.old_policy_probs[:5]), dtype=torch.float)
            while not done:
                try:
                    new_actions = self.actor(torch.tensor(state, dtype=torch.float))
                    scaled_action = scale_actions_to_correct(np.squeeze(new_actions.detach().numpy()))
                    new_state, reward, done, _ = self.env.step(scale_actions_to_correct(scaled_action))

                    new_policy_probs = to_probs(torch.tensor(scaled_action, dtype=torch.float))
                    ppo_loss = self.ppo_loss(old_policy_probs, new_policy_probs,
                                             calc_adv(reward, self.critic(torch.tensor(state, dtype=torch.float))),
                                             self.clip_range)
                    old_policy_probs = new_policy_probs

                    self.actor_optim.zero_grad()
                    actor_loss = -ppo_loss
                    actor_loss.backward()
                    self.actor_optim.step()
                    if step % 300 == 0:
                        print(
                            f'Тангаж: {scaled_action[0]}, Крен: {scaled_action[1]}, Рысканье: {scaled_action[2]}, '
                            f'Сила тяги: {scaled_action[3]}, Тормоз: {scaled_action[4]}')
                        print('------------------------------------------------------------------------------------------------------------------------------------------------------')
                        print(f'Reward: {reward}, PPO loss: {ppo_loss}')
                        print(
                              "=======================================================================================================================================================")

                    state = new_state
                    step += 1
                except Exception as e:
                    print(e)
                    time.sleep(5.0)
                    break

        torch.save(self.actor.state_dict(), 'model_scripted.pt')
