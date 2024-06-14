import random

import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.distributions import MultivariateNormal
from tqdm import tqdm
from env4UC import *
import argparse


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


class MixedPolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, discrete_action_dims, continuous_action_dim):
        super(MixedPolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.discrete_heads = nn.ModuleList([nn.Linear(hidden_dim, dim) for dim in discrete_action_dims])
        self.mean_continuous = nn.Linear(hidden_dim, continuous_action_dim)
        self.std_continuous = nn.Linear(hidden_dim, continuous_action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        probs_discrete = [F.softmax(head(x), dim=1) for head in self.discrete_heads]
        mean_continuous = torch.sigmoid(self.mean_continuous(x))
        std_continuous = F.softplus(self.std_continuous(x)) + 1e-5
        return probs_discrete, mean_continuous, std_continuous


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)

        return out


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size) if len(self.buffer) >= batch_size else self.buffer


class H_PPO:
    ''' H_PPO算法,采用截断方式  混合动作空间 包括控制启停的离散动作和输出功率的连续动作 两个策略网络共用一个价值网络 '''

    def __init__(self, state_dim, hidden_dim, discrete_action_dims, continuous_action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = MixedPolicyNet(state_dim, hidden_dim, discrete_action_dims, continuous_action_dim).to(device)

        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.replay_buffer = ReplayBuffer(capacity=1000)

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        probs_discrete, mean_continuous, std_continuous = self.actor(state)

        # 离散动作采样
        actions_discrete = [torch.distributions.Categorical(probs).sample().item() for probs in probs_discrete]

        # 连续动作采样
        actions_continuous_dist = torch.distributions.Normal(mean_continuous, std_continuous)
        actions_continuous = actions_continuous_dist.sample().squeeze().detach().cpu().numpy()

        # 返回混合动作
        return actions_discrete, actions_continuous

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float32).to(self.device)
        actions_discrete = torch.tensor(np.array(transition_dict['actions_discrete']), dtype=torch.int64).view(-1, 8).to(
            self.device)
        actions_continuous = torch.tensor(np.array(transition_dict['actions_continuous']),
                                         dtype=torch.float).view(-1, 8).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']),
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']),
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                      td_delta.cpu()).to(self.device)
        states=states.squeeze(1)
        old_probs_discrete, old_mean_continuous, old_std_continuous = self.actor(states)
        # old_probs_discrete=torch.tensor(old_probs_discrete,dtype=torch.float32)
        old_probs_discrete=torch.stack(old_probs_discrete,dim=0)
        old_probs_discrete=old_probs_discrete.transpose(0,1)


        # 原始离散动作概率密度计算
        # torch.Size([216, 8, 2]) ------shape
        old_log_probs_discrete = []
        for i in range(old_probs_discrete.shape[0]):
            old_log_probs_discrete.append(
                torch.sum(torch.log(torch.gather(old_probs_discrete[i], 1, actions_discrete[i].unsqueeze(1)))))
        old_log_probs_discrete = torch.stack(old_log_probs_discrete).detach()
        # 原始连续动作概率密度计算
        continuous_action_dists = torch.distributions.Normal(old_mean_continuous.clone().detach(),old_std_continuous.clone().detach())
        old_log_probs_continuous = continuous_action_dists.log_prob(actions_continuous.clone().detach())
        # 训练
        for _ in range(self.epochs):
            probs_discrete, mean_continuous, std_continuous = self.actor(states)
            probs_discrete = torch.stack(probs_discrete, dim=0)
            probs_discrete = probs_discrete.transpose(0, 1)
            # 离散动作loss
            log_probs_discrete = []
            for i in range(probs_discrete.shape[0]):
                log_probs_discrete.append(
                    torch.sum(torch.log(torch.gather(probs_discrete[i], 1, actions_discrete[i].unsqueeze(1)))))
            log_probs_discrete = torch.stack(log_probs_discrete)
            ratio_discrete = torch.exp(log_probs_discrete - old_log_probs_discrete)
            surr1_discrete = ratio_discrete * advantage
            surr2_discrete = torch.clamp(ratio_discrete, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss_discrete = torch.mean(-torch.min(surr1_discrete, surr2_discrete))
            # 连续动作空间loss
            action_dists_continuous= torch.distributions.Normal(mean_continuous,  std_continuous)
            log_probs_continuous= action_dists_continuous.log_prob(actions_continuous)
            ratio_continuous= torch.exp(log_probs_continuous - old_log_probs_continuous)
            surr1_continuous = ratio_continuous * advantage
            surr2_continuous = torch.clamp(ratio_continuous, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss_continuous = torch.mean(-torch.min(surr1_continuous, surr2_continuous))
            actor_loss = actor_loss_discrete + actor_loss_continuous
            # critic loss
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='H-PPO Hyperparameters')
    parser.add_argument('--actor_lr', type=float, default=1e-3, help='Learning rate for the actor')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='Learning rate for the critic')
    # parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers')
    parser.add_argument('--gamma', type=float, default=0.98, help='Discount factor')
    parser.add_argument('--lmbda', type=float, default=0.95, help='Lambda for GAE')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--eps', type=float, default=0.2, help='H-PPO clip parameter')

    args = parser.parse_args()
    actor_lr = args.actor_lr
    critic_lr = args.critic_lr
    hidden_dim = args.hidden_dim
    gamma = args.gamma
    lmbda = args.lmbda
    epochs = args.epochs
    eps = args.eps
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    env = Make_UC_env()
    env_name = 'Unit-Commitment'
    torch.manual_seed(0)
    state_dim = env.num_gen + 2  # 1 表示下一时刻的需求和timesetp
    continuous_action_dim = env.num_gen
    discrete_action_dims = [2, 2, 2, 2, 2, 2, 2, 2]  # 每个维度上有两个离散动作
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = H_PPO(state_dim, hidden_dim, discrete_action_dims, continuous_action_dim,
                  actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
    max_dict_lengh = 50
    log = {'mean_timesteps': [],
           'mean_reward': []}
    for i in range(5):

        transition_dict = {'states': [], 'actions_discrete': [],'actions_continuous':[], 'next_states': [], 'rewards': [],
                           'dones': []}
        if i % 10 == 0:
            print("Episode: ", i)
        epoch_rewards = []
        while len(transition_dict['states']) < max_dict_lengh:
            state = env.reset()
            done = False
            time_step = 0
            epo_reward = 0
            while not done:
                state=state.reshape((-1, state_dim))
                actions_discrete, actions_continuous= agent.take_action(state)
                print(actions_discrete, actions_continuous)
                next_state, reward, done = env.step((actions_discrete, actions_continuous))
                print(next_state,"next_state")
                print(reward,"reward")
                print(done,"done")
                epo_reward += reward
                transition_dict['states'].append(state)
                transition_dict['actions_discrete'].append(actions_discrete)
                transition_dict['actions_continuous'].append(actions_continuous)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                time_step += 1
                if done:
                    epoch_rewards.append(epo_reward)
        log['mean_reward'].append(np.mean(epoch_rewards))
        agent.update(transition_dict)



