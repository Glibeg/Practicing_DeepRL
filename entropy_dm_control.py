import gym
import os
import sys
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import dm_control.suite as suite
import dm_control.viewer as viewer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions.normal import Normal
import torch.distributions

from vime import VIME
from pathlib import Path
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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

STD_MAX_NOISE = 1

class PolicyNet(nn.Module):
    def __init__(self, observations, actions, act_limit):
        super(PolicyNet, self).__init__()
        self.policy_fc1 = nn.Linear(observations, 400)
        self.policy_fc2 = nn.Linear(400, 300)
        self.policy_fc3 = nn.Linear(300, 200)
        self.policy_mean_out = nn.Linear(200, actions)
        self.policy_logstd_out = nn.Linear(200, actions)
        self.act_limit = act_limit

    def forward(self, x):#, deterministic = False, with_logprob = True):
        h = F.elu(self.policy_fc1(x))
        h = F.elu(self.policy_fc2(h))
        h = F.elu(self.policy_fc3(h))

        mean = torch.tanh(self.policy_mean_out(h))
        std = 0.1 + (STD_MAX_NOISE - 0.1) * torch.sigmoid(self.policy_logstd_out(h))

        pi_distribution = Normal(mean, std)

        pi_action = pi_distribution.rsample()

        logp_pi = pi_distribution.log_prob(pi_action).sum(axis = -1)
        logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        
        return pi_action, logp_pi

class ValueNet(nn.Module):
    def __init__(self, observations, actions):
        super(ValueNet, self).__init__()
        self.critic_fc1 = nn.Linear(observations + actions, 400)
        self.critic_fc2 = nn.Linear(400, 400)
        self.critic_fc3 = nn.Linear(400, 300)
        self.critic_out = nn.Linear(300, 1)

    def forward(self, x, a):
        ch1 = F.elu(self.critic_fc1(torch.cat([x,a], dim = 1)))
        ch2 = F.elu(self.critic_fc2(ch1))
        ch3 = F.elu(self.critic_fc3(ch2))
        return self.critic_out(ch3)

class SAC():
    def __init__(self, states_dim, action_space):
        self.learning_rate = 0.0005
        self.memory_capacity = int(1e+6)
        self.batch_size = 256
        self.action_space = action_space
        self.alpha = 0.1
        self.gamma = 0.9
        self.polyak = 0.995
        self.train_period = 200
        self.start_train_after = 20000
        self.train_num = 10
        self.render = True
        self.gradient_clip = 1.

        self.policy_net = PolicyNet(states_dim,action_space.shape[0], torch.tensor(self.action_space.maximum.copy(), dtype = torch.float32).to(device)).to(device)
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

    def get_action(self, state):
        with torch.no_grad():
            a, logp = self.policy_net(state)
            return torch.max(a, torch.tensor(self.action_space.minimum.copy(), dtype = torch.float32).to(device))

    def train_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        qloss_list, ploss_list = [],[]
        for c in range(self.train_num):
            transitions = self.replay_buffer.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            next_state_batch = torch.cat(batch.next_state)
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            with torch.no_grad():
                target_action, target_action_logprob = self.policy_net(next_state_batch)
                target_action = torch.max(target_action, torch.tensor(self.action_space.minimum.copy(), dtype = torch.float32).to(device))
                value_target = torch.min(self.value_net_target1(next_state_batch, target_action), self.value_net_target2(next_state_batch, target_action))
                td_target = reward_batch + self.gamma * (value_target - self.alpha * target_action_logprob.unsqueeze(1))
            
            self.value_optim1.zero_grad()
            self.value_optim2.zero_grad()
            state_action1 = self.value_net1(state_batch, action_batch)
            state_action2 = self.value_net2(state_batch, action_batch)
            q_loss = F.mse_loss(state_action1, td_target) + F.mse_loss(state_action2, td_target)
            q_loss.backward()
            self.value_optim1.step()
            self.value_optim2.step()
            qloss_list.append(q_loss.cpu().detach().numpy())

            self.policy_optim.zero_grad()
            action, action_logprob = self.policy_net(state_batch)
            value1 = self.value_net1(state_batch, action)
            value2 = self.value_net2(state_batch, action)
            q_pi = torch.min(value1, value2)
            policy_loss = (self.alpha * action_logprob.unsqueeze(1) - q_pi).mean()
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
        
        return  np.mean(qloss_list), np.mean(ploss_list)

def state_preprocessing(state):
    return torch.tensor(state.copy(), dtype = torch.float32).unsqueeze(0).to(device)

parser = argparse.ArgumentParser()
parser.add_argument('-domain', action = 'store', default= 'quadruped')
parser.add_argument('-task', action = 'store', default= 'run')
parser.add_argument('-time', default = 30, type = int)
parser.add_argument('-episodes', default = 3000, type = int)
parser.add_argument('--test', action = 'store_true')
parser.add_argument('--load', action = 'store_true')
args = parser.parse_args()
#env = gym.make('FetchReach-v1')
env = suite.load(domain_name = args.domain, task_name = args.task, environment_kwargs = {'flat_observation' : True}, task_kwargs = {'time_limit' : args.time})
#env.env.reward_type = 'dense'
#env = gym.wrappers.filter_observation.FilterObservation(env, filter_keys=['observation', 'desired_goal'])
#env = gym.wrappers.flatten_observation.FlattenObservation(env)
agent = SAC(env.observation_spec()['observations'].shape[0], env.action_spec())
Path('/sac').mkdir(exist_ok = True)
#vime = VIME(90, env.action_spec().shape[0], device = device, hidden_layer_size = 256)
if args.load:
    agent.policy_net.load_state_dict(torch.load(f'./sac/{args.domain}_{args.task}_pnet.pth'))
    agent.value_net1.load_state_dict(torch.load(f'./sac/{args.domain}_{args.task}_vnet1.pth'))
    agent.value_net2.load_state_dict(torch.load(f'./sac/{args.domain}_{args.task}_vnet2.pth'))
    agent.value_net_target1.load_state_dict(torch.load(f'./sac/{args.domain}_{args.task}_vnet1.pth'))
    agent.value_net_target2.load_state_dict(torch.load(f'./sac/{args.domain}_{args.task}_vnet2.pth'))

scores, episodes = [], []
score_avg = 0

def policy_by_agent(time_step):
    torch_action =  agent.get_action(state_preprocessing(time_step.observation['observations']))
    return torch_action.squeeze(0).cpu().detach().numpy()

if args.test:
    agent.policy_net.load_state_dict(torch.load(f'./sac/{args.domain}_{args.task}_pnet.pth'))
    viewer.launch(env, policy = policy_by_agent)
    exit(0)

interaction_count = 0
num_episodes = args.episodes
for e in range(num_episodes):
    done = False
    score = 0
    qloss_list, ploss_list = [], []
    time_step = env.reset()
    #o = time_step.observation['observations']
    state = state_preprocessing(time_step.observation['observations'])

    for c in count():
        if interaction_count <= agent.start_train_after:
            action = np.random.uniform(env.action_spec().minimum, env.action_spec().maximum, env.action_spec().shape)
            action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            action = agent.get_action(state)
        
        a = action.squeeze(0).cpu().detach().numpy()
        time_step = env.step(a)
        #n_o = time_step.observation['observations']
        next_state = state_preprocessing(time_step.observation['observations'])
        r = time_step.reward
        score += r

        #info_gain, log_likelihood = vime.calc_info_gain(o,a,n_o)
        #vime.memorize_episodic_info_gains(info_gain)
        #r = vime.calc_curiosity_reward(r, info_gain)
        reward = torch.tensor([[r]], dtype=torch.float32).to(device)

        agent.replay_buffer.push(state, action, next_state, reward)
        state = next_state
        #o = n_o
        
        if interaction_count % agent.train_period == 0 and interaction_count > agent.start_train_after:
            qloss, ploss = agent.train_model()
            if qloss != None:
                qloss_list.append(qloss)
            if ploss != None:
                ploss_list.append(ploss)
        interaction_count += 1

        if time_step.step_type == 2:
            qloss_avg = 0.0 if len(qloss_list) == 0 else np.mean(qloss_list)
            ploss_avg = 0.0 if len(ploss_list) == 0 else np.mean(ploss_list)
            print("episode : {:4d} | end at : {:3d} steps | total interactions : {:7d} | score : {:7.3f} | q_loss = {:7.3f} | p_loss = {:7.3f} ".format(e, c+1, interaction_count, score, qloss_avg, ploss_avg))
            break
    #for _ in range(100):
    #    transitions = agent.replay_buffer.sample(agent.batch_size)
    #    batch = Transition(*zip(*transitions))
    #    next_state_batch = torch.cat(batch.next_state)
    #    state_batch = torch.cat(batch.state)
    #    action_batch = torch.cat(batch.action)
    #    elbo = vime.update_posterior(state_batch, action_batch, next_state_batch)
    torch.save(agent.policy_net.state_dict(), f'./sac/{args.domain}_{args.task}_pnet.pth')
    torch.save(agent.value_net1.state_dict(), f'./sac/{args.domain}_{args.task}_vnet1.pth')
    torch.save(agent.value_net2.state_dict(), f'./sac/{args.domain}_{args.task}_vnet2.pth')
env.close()