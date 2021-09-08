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

import argparse

from vime import VIME

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#torch.set_default_dtype(torch.float32)
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
"""
STD_MAX_NOISE = 1

class PolicyNet(nn.Module):
    def __init__(self, observations, actions, act_limit):
        super(PolicyNet, self).__init__()
        self.policy_fc1 = nn.Linear(observations, 100)
        self.policy_fc2 = nn.Linear(100, 300)
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

        return pi_action, logp_pi"""
class PolicyNet(nn.Module):
    def __init__(self, observations, actions):
        super().__init__()
        self.fc1 = nn.Linear(observations, 200)
        self.fc2 = nn.Linear(200, 300)
        self.fc3 = nn.Linear(300, 400)
        self.fc4 = nn.Linear(400, 400)
        self.out = nn.Linear(400, actions)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        h = torch.relu(self.fc4(h))
        return torch.tanh(self.out(h))
"""
class ValueNet(nn.Module):
    def __init__(self, observations, actions):
        super(ValueNet, self).__init__()
        self.critic_fc1 = nn.Linear(observations + actions, 100)
        self.critic_fc2 = nn.Linear(100, 400)
        self.critic_fc3 = nn.Linear(400, 300)
        self.critic_out = nn.Linear(300, 1)

    def forward(self, x, a):
        ch1 = F.elu(self.critic_fc1(torch.cat([x,a], dim = 1)))
        ch2 = F.elu(self.critic_fc2(ch1))
        ch3 = F.elu(self.critic_fc3(ch2))
        return self.critic_out(ch3)"""

class ValueNet(nn.Module):
    def __init__(self, observations, actions):
        super().__init__()
        self.fc1 = nn.Linear(observations + actions, 300)
        self.fc2 = nn.Linear(300, 400)
        self.fc3 = nn.Linear(400, 500)
        self.fc4 = nn.Linear(500, 400)
        self.out = nn.Linear(400, 1)
    def forward(self, x, a):
        h = torch.relu(self.fc1(torch.cat([x,a], dim = -1)))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        h = torch.relu(self.fc4(h))
        return self.out(h)

class DDPG():
    def __init__(self, states_dim, action_space):
        self.learning_rate = 0.0005
        self.memory_capacity = 200000
        self.batch_size = 64
        self.action_space = action_space
        self.noise_strength = 0.1
        self.gamma = 0.9
        self.polyak = 0.995
        self.update_every = 20
        self.update_after = 1000
        self.render = True

        self.policy_net = PolicyNet(states_dim,action_space.shape[0]).to(device)
        self.policy_net_target = PolicyNet(states_dim,action_space.shape[0]).to(device)
        self.policy_net_target.load_state_dict(self.policy_net.state_dict())
        self.policy_optim = optim.Adam(self.policy_net.parameters(), lr= self.learning_rate)

        self.value_net = ValueNet(states_dim,action_space.shape[0]).to(device)
        self.value_net_target = ValueNet(states_dim,action_space.shape[0]).to(device)
        self.value_net_target.load_state_dict(self.value_net.state_dict())
        self.value_optim = optim.Adam(self.value_net.parameters(), lr= self.learning_rate)

        for param in self.value_net_target.parameters():
            param.requires_grad = False
        for param in self.policy_net_target.parameters():
            param.requires_grad = False
        self.replay_buffer = ReplayMemory(self.memory_capacity)

    def get_action(self, state):
        with torch.no_grad():
            return self.policy_net(state)

    def train_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        qloss_list, ploss_list = [],[]
        for c in range(self.update_every):
            transitions = self.replay_buffer.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            next_state_batch = torch.cat(batch.next_state)
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            with torch.no_grad():
                target_action = self.policy_net_target(next_state_batch)
                value_target = self.value_net_target(next_state_batch, target_action)
                td_target = reward_batch + self.gamma * value_target
            #print('state : ', state_batch, '\naction : ', action_batch, '\nnext_state : ', next_state_batch, '\ndone : ', done_batch, '\nreward : ', reward_batch, '\ntarget_action : ', target_action, '\nvalue_target : ', value_target, '\ntd_target : ', td_target)
            #exit(0)
            self.value_optim.zero_grad()
            state_action_value = self.value_net(state_batch, action_batch)
            q_loss = F.mse_loss(state_action_value, td_target)
            #print("q_loss",q_loss.view(-1),"\nstate_action1",state_action1.view(-1),"\nstate_action2",state_action2.view(-1),"\ntd_target",td_target.view(-1))
            q_loss.backward()
            self.value_optim.step()
            qloss_list.append(q_loss.cpu().detach().numpy())

            self.policy_optim.zero_grad()
            action = self.policy_net(state_batch)
            value = self.value_net(state_batch, action)
            policy_loss = -value.mean()
            policy_loss.backward()
            self.policy_optim.step()
            ploss_list.append(policy_loss.cpu().detach().numpy())

            #print('state_action_value : ', state_action_value, '\nq_loss : ', q_loss, '\np_loss : ', policy_loss)
            #print('state_action_value : ', state_action_value, '\ntd_target : ', td_target, '\nq_loss : ', q_loss)
            #print('action : ', action, '\nvalue : ', value, '\npolicy_loss : ', policy_loss)
            #exit(0)
            with torch.no_grad():
                for param, target_param in zip(self.value_net.parameters(), self.value_net_target.parameters()):
                    target_param.data.mul_(self.polyak)
                    target_param.data.add_(param.data * (1-self.polyak))
                for param, target_param in zip(self.policy_net.parameters(), self.policy_net_target.parameters()):
                    target_param.data.mul_(self.polyak)
                    target_param.data.add_(param.data * (1-self.polyak))
        return  np.mean(qloss_list), np.mean(ploss_list)

def state_preprocessing(state):
    return torch.tensor(state.copy(), dtype = torch.float32).unsqueeze(0).to(device)

parser = argparse.ArgumentParser()
parser.add_argument('-domain', action = 'store', default= 'reacher')
parser.add_argument('-task', action = 'store', default= 'easy')
parser.add_argument('--test', action = 'store_true')
parser.add_argument('-hidden', nargs = '*', default = ['200', '300', '300'], type = int)
parser.add_argument('-activation', action = 'store', default = 'relu', choices = ['relu', 'silu', 'elu'])
args = parser.parse_args()
#print(args)

#env = gym.make('FetchReach-v1')
env = suite.load(domain_name = args.domain, task_name = args.task, environment_kwargs = {'flat_observation' : True})#, task_kwargs = {'time_limit' : 60})
"""time_step = env.reset()
print(state_preprocessing(time_step.observation['observations']))"""
#env.env.reward_type = 'dense'
#env = gym.wrappers.filter_observation.FilterObservation(env, filter_keys=['observation', 'desired_goal'])
#env = gym.wrappers.flatten_observation.FlattenObservation(env)
agent = DDPG(env.observation_spec()['observations'].shape[0], env.action_spec())
#agent.policy_net.load_state_dict(torch.load('./pnet.pth'))
#agent.value_net.load_state_dict(torch.load('./vnet.pth'))
#vime = VIME(3, env.action_spec().shape[0], device = device, hidden_layer_size = 128)

scores, episodes = [], []
score_avg = 0

def policy_by_agent(time_step):
    torch_action =  agent.get_action(state_preprocessing(time_step.observation['observations']))
    return torch_action.squeeze(0).cpu().detach().numpy()

if args.test:
    agent.policy_net.load_state_dict(torch.load(f'./ddpg/{args.domain}_{args.task}_pnet.pth'))
    viewer.launch(env, policy = policy_by_agent)
    exit(0)
#viewer.launch(env, policy=policy_by_agent)
#exit(0)
interaction_count = 0
num_episodes = 1000
for e in range(num_episodes):
    done = False
    score = 0
    qloss_list, ploss_list = [], []
    time_step = env.reset()
    #o = time_step.observation['observations']
    state = state_preprocessing(time_step.observation['observations'])

    for c in count():
        if interaction_count <= agent.update_after:
            action = np.random.uniform(env.action_spec().minimum, env.action_spec().maximum, env.action_spec().shape)
            action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            with torch.no_grad():
                action = agent.get_action(state)
                epsilon = torch.randn_like(action) * agent.noise_strength
                action = torch.min(torch.max(action + epsilon, torch.tensor(agent.action_space.minimum, dtype = torch.float32).to(device)), torch.tensor(agent.action_space.maximum, dtype = torch.float32).to(device))
        a = action.squeeze(0).cpu().detach().numpy()
        time_step = env.step(a)
        #print(time_step)
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
        interaction_count += 1
        #o = n_o
        
        if interaction_count % agent.update_every == 0:
            qloss, ploss = agent.train_model()
            if qloss != None:
                qloss_list.append(qloss)
            if ploss != None:
                ploss_list.append(ploss)

        if time_step.step_type == 2:
            #score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            print("episode : {:3d} | end at : {:3d} steps | score : {:3.3f} | q_loss = {:3.3f} | p_loss = {:.3f} ".format(e, c, score, np.mean(qloss_list), np.mean(ploss_list)))
            #if score_avg > 900:
            #    sys.exit()
            break
    #for _ in range(100):
    #    transitions = agent.replay_buffer.sample(agent.batch_size)
    #    batch = Transition(*zip(*transitions))
    #    next_state_batch = torch.cat(batch.next_state)
    #    state_batch = torch.cat(batch.state)
    #    action_batch = torch.cat(batch.action)
    #    elbo = vime.update_posterior(state_batch, action_batch, next_state_batch)
    torch.save(agent.policy_net.state_dict(), f'./ddpg/{args.domain}_{args.task}_pnet.pth')
    torch.save(agent.value_net.state_dict(), f'./ddpg/{args.domain}_{args.task}_vnet.pth')
env.close()