import torch
from torch.optim import Adam


class PPO:
    def __init__(self, policy_class, env):
        # Enviroment info
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Init actor and critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim)
        self.critic = policy_class(self.obs_dim, 1)

        # Init optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    def ppo_loss(self, old_policy_probs, new_policy_probs, advantages, clip_range):
        ratio = new_policy_probs / old_policy_probs
        clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        return -torch.min(surrogate1, surrogate2).mean()

    def collect_experience(self, env, policy_network, num_epochs):
        states = []
        actions = []
        rewards = []
        old_policy_probs = []

        for epoch in range(num_epochs):
            state = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    action_probs = policy_network(torch.tensor(state, dtype=torch.float))

    def scale_actions_to_correct(self, actions):
        actions = actions.detach().numpy()
        actions[3] = actions[3] - 1
        actions[4] = max(actions[4], 0)
        return actions
