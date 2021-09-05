import gym
import sys
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions.normal import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class PolicyNet(nn.Module):
    def __init__(self, observations, actions):
        super(PolicyNet, self).__init__()
        self.policy_fc1 = nn.Linear(observations, 128)
        self.policy_fc2 = nn.Linear(128, 128)
        self.policy_fc3 = nn.Linear(128, 64)
        self.policy_mean_out = nn.Linear(64, actions)
        self.policy_logstd_out = nn.Linear(64, actions)

    def forward(self, x, deterministic = False, with_logprob = True):
        h = torch.relu(self.policy_fc1(x))
        h = torch.relu(self.policy_fc2(h))
        h = torch.relu(self.policy_fc3(h))
        mean = torch.tanh(self.policy_mean_out(h))
        std = torch.exp(torch.clamp(self.policy_logstd_out(h), LOG_STD_MIN, LOG_STD_MAX))
        pi_distribution = Normal(mean, std)
        if deterministic:
            pi_action = mean
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis = -1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        return pi_action, logp_pi

class ValueNet(nn.Module):
    def __init__(self, observations, actions):
        super(ValueNet, self).__init__()
        self.critic_fc1 = nn.Linear(observations + actions, 128)
        self.critic_fc2 = nn.Linear(128, 128)
        self.critic_fc3 = nn.Linear(128, 64)
        self.critic_out = nn.Linear(64, 1)

    def forward(self, x, a):
        ch1 = torch.relu(self.critic_fc1(torch.cat([x,a], dim = 1)))
        ch2 = torch.relu(self.critic_fc2(ch1))
        ch3 = torch.relu(self.critic_fc3(ch2))
        return self.critic_out(ch3)

class SAC():
    def __init__(self, states_dim, action_space):
        self.learning_rate = 1e-4
        self.memory_capacity = 10000
        self.batch_size = 64
        self.action_space = action_space
        self.alpha = 0.15
        self.gamma = 0.9
        self.polyak = 0.995
        self.update_every = 1
        self.update_after = 1000
        self.render = True

        self.policy_net = PolicyNet(states_dim,action_space.shape[0]).to(device)
        self.policy_optim = optim.Adam(self.policy_net.parameters(), lr= self.learning_rate)

        self.value_net1 = ValueNet(states_dim,action_space.shape[0]).to(device)
        self.value_net_target1 = ValueNet(states_dim,action_space.shape[0]).to(device)
        self.value_net_target1.load_state_dict(self.value_net1.state_dict())
        self.value_optim1 = optim.Adam(self.value_net1.parameters(), lr= self.learning_rate)

        self.value_net2 = ValueNet(states_dim,action_space.shape[0]).to(device)
        self.value_net_target2 = ValueNet(states_dim,action_space.shape[0]).to(device)
        self.value_net_target2.load_state_dict(self.value_net2.state_dict())
        self.value_optim2 = optim.Adam(self.value_net2.parameters(), lr= self.learning_rate)

        for param in self.value_net_target1.parameters():
            param.requires_grad = False
        for param in self.value_net_target2.parameters():
            param.requires_grad = False
        self.replay_buffer = ReplayMemory(self.memory_capacity)

    def get_action(self, state, deterministic = False):
        with torch.no_grad():
            action, _ = self.policy_net(state, deterministic = deterministic, with_logprob = False)
            return torch.max(torch.min(action, torch.tensor(self.action_space.high).to(device)), torch.tensor(self.action_space.low).to(device))

    def train_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None, None
        qloss_list1, qloss_list2, ploss_list = [],[],[]
        for c in range(self.update_every):
            transitions = self.replay_buffer.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            next_state_batch = torch.cat(batch.next_state)
            done_batch = torch.cat(batch.done)
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            with torch.no_grad():
                target_action, action_lopprob = self.policy_net(next_state_batch)
                target_action = torch.max(torch.min(target_action, torch.tensor(self.action_space.high).to(device)), torch.tensor(self.action_space.low).to(device))
                value_target = torch.min(self.value_net_target1(next_state_batch, target_action), self.value_net_target2(next_state_batch, target_action))
                td_target = reward_batch + self.gamma * (1 - done_batch) * (value_target - self.alpha * action_lopprob.unsqueeze(1))
            
            self.value_optim1.zero_grad()
            state_action = self.value_net1(state_batch, action_batch)
            q_loss1 = F.mse_loss(state_action, td_target)
            q_loss1.backward()
            self.value_optim1.step()
            qloss_list1.append(q_loss1.cpu().detach().numpy())
            
            self.value_optim2.zero_grad()
            state_action = self.value_net2(state_batch, action_batch)
            q_loss2 = F.mse_loss(state_action, td_target)
            q_loss2.backward()
            self.value_optim2.step()
            qloss_list2.append(q_loss2.cpu().detach().numpy())

            self.policy_optim.zero_grad()
            action, action_logp = self.policy_net(state_batch)
            value1 = self.value_net1(state_batch, action)
            value2 = self.value_net2(state_batch, action)
            q_pi = torch.min(value1, value2)
            policy_loss = (self.alpha * action_logp.unsqueeze(1) - q_pi).mean()
            policy_loss.backward()
            self.policy_optim.step()
            ploss_list.append(policy_loss.cpu().detach().numpy())

            with torch.no_grad():
                for param, target_param in zip(self.value_net1.parameters(), self.value_net_target1.parameters()):
                    target_param.data.mul_(self.polyak)
                    target_param.data.add_(param.data * (1-self.polyak))
                for param, target_param in zip(self.value_net2.parameters(), self.value_net_target2.parameters()):
                    target_param.data.mul_(self.polyak)
                    target_param.data.add_(param.data * (1-self.polyak))

        return  np.mean(qloss_list1), np.mean(qloss_list2), np.mean(ploss_list)

env = gym.make('FetchReach-v1')
env.env.reward_type = 'dense'
env = gym.wrappers.filter_observation.FilterObservation(env, filter_keys=['observation', 'desired_goal'])
env = gym.wrappers.flatten_observation.FlattenObservation(env)
agent = SAC(13, env.action_space)

scores, episodes = [], []
score_avg = 0

def state_preprocessing(state):
    return torch.tensor(state.copy(), dtype = torch.float32).unsqueeze(0).to(device)

initial_count = 0
num_episodes = 2000
for e in range(num_episodes):
    done = False
    score = 0
    qloss_list1, qloss_list2, ploss_list = [], [], []
    state = state_preprocessing(env.reset())

    for c in count():
        if agent.render:
            env.render()
        if initial_count <= agent.update_after:
            action = torch.tensor(env.action_space.sample(), dtype=torch.float32).unsqueeze(0).to(device)
            initial_count += 1
        else:
            action = agent.get_action(state)
        next_state, reward, done, info = env.step(action.squeeze(0).cpu().detach().numpy())
        score += reward
        next_state = state_preprocessing(next_state)
        reward = torch.tensor([[reward]], dtype=torch.float32).to(device)
        done = torch.tensor([[1 if done else 0]]).to(device)
        agent.replay_buffer.push(state, action, next_state, reward, done)
        state = next_state
        
        if c % agent.update_every == 0:
            qloss1, qloss2, ploss = agent.train_model()
            if qloss1 != None:
                qloss_list1.append(qloss1)
            if qloss2 != None:
                qloss_list2.append(qloss2)
            if ploss != None:
                ploss_list.append(ploss)

        if done:
            #score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            print("episode : {:3d} | end at : {:3d} steps | score : {:3.3f} | q_loss1 = {:3.3f} | q_loss2 = {:3.3f} | p_loss = {:.3f} | end with success : {:s}".format(e, c, score, np.mean(qloss_list1), np.mean(qloss_list2), np.mean(ploss_list), 'True' if info['is_success'] else 'False'))
            #if score_avg > 900:
            #    sys.exit()
            break
    torch.save(agent.policy_net.state_dict(), './pnet.pth')
    torch.save(agent.value_net1.state_dict(), './vnet1.pth')
    torch.save(agent.value_net2.state_dict(), './vnet2.pth')