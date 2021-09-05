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
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """transition 저장"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

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
        return F.leaky_relu(self.policy_out(ch3), 0.2)

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
        return F.leaky_relu(self.critic_out(ch3), 0.2)

class TD3():
    def __init__(self, state_dims, action_space):
        self.learning_rate = 0.001
        self.memory_capacity = 10000
        self.batch_size = 32
        self.action_space = action_space
        self.gamma = 0.9
        self.polyak = 0.995
        self.target_noise = 0.2
        self.target_clip = 0.5
        self.update_every = 30
        self.policy_delay = 2
        self.random_selection = 10000
        self.update_after = 1000
        self.render = True

        self.policy_net = PolicyNet(state_dims,action_space.shape[0]).to(device)
        self.policy_net_target = PolicyNet(state_dims,action_space.shape[0]).to(device)
        self.policy_net_target.load_state_dict(self.policy_net.state_dict())
        self.policy_optim = optim.Adam(self.policy_net.parameters(), lr= self.learning_rate)

        self.value_net1 = ValueNet(state_dims,action_space.shape[0]).to(device)
        self.value_net_target1 = ValueNet(state_dims,action_space.shape[0]).to(device)
        self.value_net_target1.load_state_dict(self.value_net1.state_dict())
        self.value_optim1 = optim.Adam(self.value_net1.parameters(), lr= self.learning_rate)

        self.value_net2 = ValueNet(state_dims,action_space.shape[0]).to(device)
        self.value_net_target2 = ValueNet(state_dims,action_space.shape[0]).to(device)
        self.value_net_target2.load_state_dict(self.value_net2.state_dict())
        self.value_optim2 = optim.Adam(self.value_net2.parameters(), lr= self.learning_rate)

        for param in self.policy_net_target.parameters():
            param.requires_grad = False
        for param in self.value_net_target1.parameters():
            param.requires_grad = False
        for param in self.value_net_target2.parameters():
            param.requires_grad = False
        self.replay_buffer = ReplayMemory(self.memory_capacity)

    def get_action(self, state, noise):
        with torch.no_grad():
            action = self.policy_net(state)
            noise_v = torch.randn(self.action_space.shape).to(device) * noise
            return torch.max(torch.min(action + noise_v, torch.tensor(self.action_space.high).to(device)), torch.tensor(self.action_space.low).to(device))

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
                target_action = self.policy_net_target(next_state_batch)
                noise = torch.clamp(torch.randn(target_action.shape).to(device) * self.target_noise, min=-self.target_clip, max=self.target_clip)
                target_action = torch.max(torch.min(target_action + noise, torch.tensor(self.action_space.high).to(device)), torch.tensor(self.action_space.low).to(device))
                value_target = torch.min(self.value_net_target1(next_state_batch, target_action), self.value_net_target2(next_state_batch, target_action))
                td_target = reward_batch + self.gamma * (1 - done_batch) * value_target
            
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

            if c % self.policy_delay == 0:
                self.policy_optim.zero_grad()
                policy_loss = self.value_net1(state_batch, self.policy_net(state_batch))
                policy_loss = -policy_loss.mean()
                policy_loss.backward()
                self.policy_optim.step()
                ploss_list.append(policy_loss.cpu().detach().numpy())

                with torch.no_grad():
                    for param, target_param in zip(self.policy_net.parameters(), self.policy_net_target.parameters()):
                        target_param.data.mul_(self.polyak)
                        target_param.data.add_(param.data * (1-self.polyak))
                    for param, target_param in zip(self.value_net1.parameters(), self.value_net_target1.parameters()):
                        target_param.data.mul_(self.polyak)
                        target_param.data.add_(param.data * (1-self.polyak))
                    for param, target_param in zip(self.value_net2.parameters(), self.value_net_target2.parameters()):
                        target_param.data.mul_(self.polyak)
                        target_param.data.add_(param.data * (1-self.polyak))

        return  np.mean(qloss_list1), np.mean(qloss_list2), np.mean(ploss_list)

env = gym.make('HalfCheetah-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
agent = TD3(state_size, env.action_space)

scores, p_loss_, q_loss1_, q_loss2_ = [], [], [], []
score_avg = 0

initial_count = 0
num_episodes = 1000
for e in range(num_episodes):
    done = False
    score = 0
    qloss_list1, qloss_list2, ploss_list = [], [], []
    state = env.reset()
    state = torch.tensor(state.copy(), dtype= torch.float32).unsqueeze(0).to(device)

    for c in count():
        if agent.render:
            env.render()
        if initial_count <= agent.random_selection:
            action = torch.tensor(env.action_space.sample(), dtype=torch.float32).unsqueeze(0).to(device)
            initial_count += 1
        else:
            action = agent.get_action(state, 0.1)
        next_state, reward, done, info = env.step(action.squeeze(0).cpu().detach().numpy())
        score += reward
        next_state = torch.tensor(next_state.copy(), dtype= torch.float32).unsqueeze(0).to(device)
        reward = torch.tensor([[reward.astype(np.float32)]]).to(device)
        done = torch.tensor([[1 if done else 0]]).to(device)
        agent.replay_buffer.push(state, action, next_state, reward, done)
        state = next_state
        
        if initial_count > agent.update_after and c % agent.update_every == 0:
            qloss1, qloss2, ploss = agent.train_model()
            
            if qloss1 != None:
                qloss_list1.append(qloss1)
            if qloss2 != None:
                qloss_list2.append(qloss2)
            if ploss != None:
                ploss_list.append(ploss)

        if done:
            scores.append(score)
            p_loss_.append(np.mean(ploss_list))
            q_loss1_.append(np.mean(qloss_list1))
            q_loss2_.append(np.mean(qloss_list2))

            plt.figure(0)
            plt.clf()
            plt.plot(scores, label="score")
            plt.title("TD3 - HumanoidStandUp")
            plt.xlabel('episodes(5000 step)')
            plt.ylabel('score')
            plt.legend()

            plt.figure(1)
            plt.clf()
            plt.plot(p_loss_, label="policy_loss")
            plt.plot(q_loss1_, label="q_net1_loss")
            plt.plot(q_loss2_, label="q_net2_loss")
            plt.title("HalfCheetah")
            plt.xlabel('episodes(5000 step)')
            plt.legend()
            plt.pause(0.003)
            if is_ipython:
                display.clear_output(wait=True)
                display.display(plt.gcf())
            score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            print("episode : {:3d} | score_avg : {:3.2f}".format(e, score_avg))#, np.mean(qloss_list1), np.mean(qloss_list2), np.mean(ploss_list)))
            if score_avg > 900:
                sys.exit()
            break
    torch.save(agent.policy_net.state_dict(), './pnet.pth')
    torch.save(agent.value_net1.state_dict(), './vnet1.pth')
    torch.save(agent.value_net2.state_dict(), './vnet2.pth')