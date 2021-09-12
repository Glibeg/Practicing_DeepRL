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

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def state_preprocessing(state):
    return torch.tensor(state.copy(), dtype = torch.float32).unsqueeze(0).to(device)

parser = argparse.ArgumentParser()
parser.add_argument('-agent', action = 'store', choices = ['ddpg','sac'])
parser.add_argument('-env', action = 'store', choices = ['dm_control','openai_gym'])
parser.add_argument('-domain', action = 'store', default= 'reacher')
parser.add_argument('-task', action = 'store', default= 'easy')
parser.add_argument('--test', action = 'store_true')

args = parser.parse_args()

agent_available_list = ['ddpg','sac']
env_available_list = ['dm_control','openai_gym']

if args.agent not in agent_available_list or args.env not in env_available_list:
    print(f'{args.agent} or {args.env} is not valid input')
    exit(0)
#print(args)
Path(f'/{args.agent}').mkdir(exist_ok= True)
env = suite.load(domain_name = args.domain, task_name = args.task, environment_kwargs = {'flat_observation' : True})#, task_kwargs = {'time_limit' : 60})
agent = DDPG(env.observation_spec()['observations'].shape[0], env.action_spec())

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