import gym
import sys
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from agent.DDPG import DDPG
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNet(nn.Module):
    def __init__(self, observations, actions):
        super(PolicyNet, self).__init__()
        self.policy_fc1 = nn.Linear(observations, 256)
        self.policy_fc2 = nn.Linear(256, 128)
        self.policy_fc3 = nn.Linear(128, 64)
        self.policy_out = nn.Linear(64, actions)

    def forward(self, x):
        ch1 = torch.relu(self.policy_fc1(x))
        ch2 = torch.relu(self.policy_fc2(ch1))
        ch3 = torch.relu(self.policy_fc3(ch2))
        return torch.tanh(self.policy_out(ch3))

class ValueNet(nn.Module):
    def __init__(self, observations, actions):
        super(ValueNet, self).__init__()
        self.critic_fc1 = nn.Linear(observations + actions, 256)
        self.critic_fc2 = nn.Linear(256, 128)
        self.critic_fc3 = nn.Linear(128, 64)
        self.critic_out = nn.Linear(64, 1)

    def forward(self, x, a):
        ch1 = torch.relu(self.critic_fc1(torch.cat([x,a], dim = 1)))
        ch2 = torch.relu(self.critic_fc2(ch1))
        ch3 = torch.relu(self.critic_fc3(ch2))
        return torch.tanh(self.critic_out(ch3))

env = gym.make('Ant-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
policyNet = PolicyNet(state_size, action_size).to(device)
policyOptim = optim.Adam(policyNet.parameters(), lr= 0.00025, weight_decay= 1e-3)
valueNet = ValueNet(state_size, action_size).to(device)
valueOptim = optim.Adam(valueNet.parameters(), lr= 0.00025, weight_decay=1e-3)
agent = DDPG(policyNet, valueNet, policyOptim, valueOptim, env.action_space)

scores, p_loss_, q_loss_ = [], [], []
score_avg = 0

num_episodes = 1000
for e in range(num_episodes):
    done = False
    score = 0
    qloss_list, ploss_list = [], []
    state = env.reset()
    state = torch.tensor(state.copy(), dtype= torch.float32).unsqueeze(0).to(device)
    for c in count():
        if agent.render:
            env.render()
        action = agent.get_action(state, 0.1)
        next_state, reward, done, info = env.step(action.squeeze(0).cpu().detach().numpy())
        score += reward
        next_state = torch.tensor(next_state.copy(), dtype= torch.float32).unsqueeze(0).to(device)
        agent.save_transition(state, action, done, reward.astype(np.float32), next_state)
        state = next_state
        
        qloss, ploss = agent.train_model()
        if qloss != None:
            qloss_list.append(qloss)
        if ploss != None:
            ploss_list.append(ploss)

        if done:
            scores.append(score)
            p_loss_.append(np.mean(ploss_list))
            q_loss_.append(np.mean(qloss_list))

            plt.figure(0)
            plt.clf()
            plt.plot(scores, label="score")
            plt.title("DDPG - HalfCheetah")
            plt.xlabel('episodes(5000 step)')
            plt.ylabel('score')
            plt.legend()

            plt.figure(1)
            plt.clf()
            plt.plot(p_loss_, label="policy_loss")
            plt.plot(q_loss_, label="q_loss")
            plt.title("HalfCheetah")
            plt.xlabel('episodes(5000 step)')
            plt.legend()
            plt.pause(0.003)
            if is_ipython:
                display.clear_output(wait=True)
                display.display(plt.gcf())

            score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            print("episode : {:3d} | score_avg : {:3.2f} | q_loss = {:3.3f} | p_loss = {:.3f}".format(e, score_avg, np.mean(qloss_list), np.mean(ploss_list)))
            if score_avg > 900:
                sys.exit()
            break