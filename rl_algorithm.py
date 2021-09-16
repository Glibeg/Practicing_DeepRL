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
from pathlib import Path

from vime import VIME

from agent import DDPG

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def state_preprocessing(state):
    return torch.tensor(state.copy(), dtype = torch.float32).unsqueeze(0).to(device)

parser = argparse.ArgumentParser()
parser.add_argument('-agent', action = 'store', choices = ['ddpg','sac'])
parser.add_argument('-domain', action = 'store', default= 'reacher')
parser.add_argument('-task', action = 'store', default= 'easy')
parser.add_argument('-hidden', nargs = '*', default = ['200', '300', '300'], type = int)
parser.add_argument('-activation', action = 'store', default = 'relu', choices = ['relu', 'silu', 'elu'])
parser.add_argument('-lr', action = 'store', default = 5e-4, type = float)
parser.add_argument('--test', action = 'store_true')
args = parser.parse_args()

activation_dict = {'relu' : nn.ReLU, 'silu' : nn.SiLU, 'elu' : nn.ELU}

#print(args)
Path(f'/{args.agent}').mkdir(exist_ok= True)
env = suite.load(domain_name = args.domain, task_name = args.task, environment_kwargs = {'flat_observation' : True})#, task_kwargs = {'time_limit' : 60})

class PolicyNet(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_layers, activation):
        super(PolicyNet, self).__init__()
        layers = []
        for i in range(len(hidden_layers)):
            layers += [nn.Linear(([observation_dim] + hidden_layers)[i], hidden_layers[i]), activation()]
        self.mlp = nn.Sequential(*layers)
        self.policy_mean_out = nn.Linear(hidden_layers[-1], action_dim)
        self.policy_logstd_out = nn.Linear(hidden_layers[-1], action_dim)

    def forward(self, x):#, deterministic = False, with_logprob = True):
        h = self.mlp(x)
        mean = torch.tanh(self.policy_mean_out(h))
        std = 0.1 + (0.9) * torch.sigmoid(self.policy_logstd_out(h))
        return mean, std

class ValueNet(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_layers, activation):
        super(ValueNet, self).__init__()
        layers = []
        for i in range(len(hidden_layers)):
            layers += [nn.Linear(([observation_dim + action_dim] + hidden_layers)[i], hidden_layers[i]), activation()]
        self.mlp = nn.Sequential(*layers)
        self.critic_out = nn.Linear(hidden_layers[-1], 1)

    def forward(self, x, a):
        h = self.mlp(torch.cat([x,a], dim = -1))
        return self.critic_out(h)

policynet = PolicyNet(env.observation_spec()['observations'].shape[0], env.action_spec().shape[0], args.hidden, activation_dict[args.activation])
policyOptim = optim.Adam(policynet.parameters(), lr= args.lr)
valuenet = ValueNet(env.observation_spec()['observations'].shape[0], env.action_spec().shape[0], args.hidden, activation_dict[args.activation])
valueOptim = optim.Adam(valuenet.parameters(), lr= args.lr)
exit()
agent = DDPG.DDPG(env.observation_spec()['observations'].shape[0], env.action_spec())

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
initial_count = 0
num_episodes = 1000
for e in range(num_episodes):
    done = False
    score = 0
    qloss_list, ploss_list = [], []
    time_step = env.reset()
    #o = time_step.observation['observations']
    state = state_preprocessing(time_step.observation['observations'])

    for c in count():
        if initial_count <= agent.update_after:
            action = np.random.uniform(env.action_spec().minimum, env.action_spec().maximum, env.action_spec().shape)
            action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
            initial_count += 1
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
        done = 1.0 if time_step.step_type == 2 else 0.0
        score += r
        done = torch.tensor([[done]]).to(device)

        #info_gain, log_likelihood = vime.calc_info_gain(o,a,n_o)
        #vime.memorize_episodic_info_gains(info_gain)
        #r = vime.calc_curiosity_reward(r, info_gain)
        reward = torch.tensor([[r]], dtype=torch.float32).to(device)

        agent.replay_buffer.push(state, action, next_state, reward, done)
        state = next_state
        #o = n_o
        
        if c % agent.update_every == 0:
            qloss, ploss = agent.train_model()
            if qloss != None:
                qloss_list.append(qloss)
            if ploss != None:
                ploss_list.append(ploss)

        if time_step.step_type == 2:
            print("episode : {:3d} | end at : {:3d} steps | score : {:3.3f} | q_loss = {:3.3f} | p_loss = {:.3f} ".format(e, c, score, np.mean(qloss_list), np.mean(ploss_list)))
            break
    torch.save(agent.policy_net.state_dict(), f'./{args.agent}/{args.domain}_{args.task}_pnet.pth')
    torch.save(agent.value_net.state_dict(), f'./{args.agent}/{args.domain}_{args.task}_vnet.pth')
env.close()