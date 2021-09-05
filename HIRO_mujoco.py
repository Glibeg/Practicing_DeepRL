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

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'goal', 'action', 'reward', 'done', 'next_state', 'next_goal'))
MetaTransition = namedtuple('MetaTransition',
                        ('state_seq', 'goal_seq', 'action_seq', 'reward_seq', 'next_state'))

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


class MetaReplayMemory(ReplayMemory):
    def __init__(self, capacity):
        super(MetaReplayMemory, self).__init__(capacity)
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = MetaTransition(*args)
        self.position = (self.position + 1) % self.capacity

class LowPolicy(nn.Module):
    def __init__(self, dim_states, num_actions):
        super(LowPolicy, self).__init__()
        self.fc1 = nn.Linear(dim_states, 32)
        self.fc1_g = nn.Linear(dim_states, 32)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, s, g):
        x = F.relu(self.fc1(s))
        gz = F.relu(self.fc1_g(g))
        x = F.relu(self.fc2(torch.cat([x, gz], dim = -1)))
        return torch.tanh(self.fc3(x))

class LowPolicyValue(nn.Module):
    def __init__(self, dim_states, num_actions):
        super(LowPolicyValue, self).__init__()
        self.fc1 = nn.Linear(dim_states, 32)
        self.fc1_g = nn.Linear(dim_states, 32)
        self.fc1_a = nn.Linear(num_actions, 32)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, s, g, a):
        x = F.relu(self.fc1(s))
        x2 = F.relu(self.fc1_g(g))
        x3 = F.relu(self.fc1_a(a))
        x = torch.tanh(self.fc2(torch.cat([x, x2], dim = -1)))
        return torch.tanh(self.fc3(torch.cat([x, x3], dim = -1)))

class HighPolicy(nn.Module):
    def __init__(self, dim_states):
        super(HighPolicy, self).__init__()
        self.fc1 = nn.Linear(dim_states, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, dim_states)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return torch.tanh(self.fc3(h))

class HighPolicyValue(nn.Module):
    def __init__(self, dim_states):
        super(HighPolicyValue, self).__init__()
        self.fc1 = nn.Linear(dim_states, 32)
        self.fc1_g = nn.Linear(dim_states, 32)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, s, g):
        h = torch.relu(self.fc1(s))
        h2 = torch.relu(self.fc1_g(g))
        h = torch.relu(self.fc2(torch.cat((h, h2), dim = -1)))
        return torch.tanh(self.fc3(h))

class HIRO():
    batch_size = 32
    learning_rate = 0.00025
    memory_size = 10000
    gamma = 0.9
    gamma_meta = 0.9
    polyak = 0.995
    target_noise = 0.2
    target_clip = 0.5
    update_every = 10
    policy_delay = 2
    random_selection = 10000
    update_after = 1000
    highpolicydelay = 3
    gradient_clip = 5.0
    render = True
    def __init__(self, num_states, num_actions, action_space):
        self.num_actions = num_actions
        self.action_space = action_space

        self.lowpolicy = LowPolicy(num_states,  num_actions).to(device)
        self.lowpolicytarget = LowPolicy(num_states, num_actions).to(device)
        self.lowpolicytarget.load_state_dict(self.lowpolicy.state_dict())
        self.lowMemory = ReplayMemory(self.memory_size)
        self.lowpolicyOptim = optim.Adam(self.lowpolicy.parameters(), lr = self.learning_rate)

        self.lowpolicyValue1 = LowPolicyValue(num_states, num_actions).to(device)
        self.lowpolicyValue2 = LowPolicyValue(num_states, num_actions).to(device)
        self.lowpolicyValueTarget1 = LowPolicyValue(num_states, num_actions).to(device)
        self.lowpolicyValueTarget2 = LowPolicyValue(num_states, num_actions).to(device)
        self.lowpolicyValueTarget1.load_state_dict(self.lowpolicyValue1.state_dict())
        self.lowpolicyValueTarget2.load_state_dict(self.lowpolicyValue2.state_dict())
        self.lowpolicyValueOptim1 = optim.Adam(self.lowpolicyValue1.parameters(), lr= self.learning_rate)
        self.lowpolicyValueOptim2 = optim.Adam(self.lowpolicyValue2.parameters(), lr= self.learning_rate)

        self.highpolicy = HighPolicy(num_states).to(device)
        self.highpolicytarget = HighPolicy(num_states).to(device)
        self.highpolicytarget.load_state_dict(self.highpolicy.state_dict())
        self.highMemory = MetaReplayMemory(self.memory_size)
        self.highpolicyOptim = optim.Adam(self.highpolicy.parameters(), lr = self.learning_rate)

        self.highpolicyValue1 = HighPolicyValue(num_states).to(device)
        self.highpolicyValue2 = HighPolicyValue(num_states).to(device)
        self.highpolicyValueTarget1 = HighPolicyValue(num_states).to(device)
        self.highpolicyValueTarget2 = HighPolicyValue(num_states).to(device)
        self.highpolicyValueTarget1.load_state_dict(self.highpolicyValue1.state_dict())
        self.highpolicyValueTarget2.load_state_dict(self.highpolicyValue2.state_dict())
        self.highpolicyValueOptim1 = optim.Adam(self.highpolicyValue1.parameters(), lr= self.learning_rate)
        self.highpolicyValueOptim2 = optim.Adam(self.highpolicyValue2.parameters(), lr= self.learning_rate)

        for param in self.lowpolicytarget.parameters():
            param.requires_grad = False
        for param in self.lowpolicyValueTarget1.parameters():
            param.requires_grad = False
        for param in self.lowpolicyValueTarget2.parameters():
            param.requires_grad = False
        for param in self.highpolicytarget.parameters():
            param.requires_grad = False
        for param in self.highpolicyValueTarget1.parameters():
            param.requires_grad = False
        for param in self.highpolicyValueTarget2.parameters():
            param.requires_grad = False

    def selectaction(self, state, goal, noise):
        with torch.no_grad():
            action = self.lowpolicy(state, goal)
            noise_v = torch.randn(self.action_space.shape).to(device) * noise
            return torch.max(torch.min(action + noise_v, torch.tensor(self.action_space.high).to(device)), torch.tensor(self.action_space.low).to(device))

    def selectgoal(self, state, noise):
        with torch.no_grad():
            goal = self.highpolicy(state)
            noise_v = torch.randn(goal.shape).to(device) * noise
            return goal + noise_v

    def train_low_polciy(self):
        if len(self.lowMemory) < self.batch_size:
            return None, None
        low_q_list, ploss_list = [], []
        for c in range(self.update_every):
            transitions = self.lowMemory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            next_state_batch = torch.cat(batch.next_state)
            done_batch = torch.cat(batch.done)
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            goal_batch = torch.cat(batch.goal)
            next_goal_batch = torch.cat(batch.next_goal)
                
            with torch.no_grad():
                target_action = self.lowpolicytarget(next_state_batch, next_goal_batch)
                noise = torch.clamp(torch.randn(target_action.shape).to(device) * self.target_noise, min=-self.target_clip, max=self.target_clip)
                target_action = torch.max(torch.min(target_action + noise, torch.tensor(self.action_space.high).to(device)), torch.tensor(self.action_space.low).to(device))
                value_target = torch.min(self.lowpolicyValueTarget1(next_state_batch, next_goal_batch, target_action), self.lowpolicyValueTarget2(next_state_batch, next_goal_batch, target_action))
                td_target = reward_batch + self.gamma * (1 - done_batch) * value_target
            
            self.lowpolicyValueOptim1.zero_grad()
            self.lowpolicyValueOptim2.zero_grad()
            state_action1 = self.lowpolicyValue1(state_batch, goal_batch, action_batch)
            state_action2 = self.lowpolicyValue2(state_batch, goal_batch, action_batch)
            q_loss1 = F.mse_loss(state_action1, td_target)
            q_loss2 = F.mse_loss(state_action2, td_target)
            q_loss = q_loss1 + q_loss2
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.lowpolicyValue1.parameters(), self.gradient_clip)
            torch.nn.utils.clip_grad_norm_(self.lowpolicyValue2.parameters(), self.gradient_clip)
            self.lowpolicyValueOptim1.step()
            self.lowpolicyValueOptim2.step()
            low_q_list.append(q_loss.cpu().detach().numpy())

            if c % self.policy_delay == 0:
                self.lowpolicyOptim.zero_grad()
                policy_loss = self.lowpolicyValue1(state_batch, goal_batch, self.lowpolicy(state_batch, goal_batch))
                policy_loss = -policy_loss.mean()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lowpolicy.parameters(), self.gradient_clip)
                self.lowpolicyOptim.step()
                ploss_list.append(policy_loss.cpu().detach().numpy())

                with torch.no_grad():
                    for param, target_param in zip(self.lowpolicy.parameters(), self.lowpolicytarget.parameters()):
                        target_param.data.mul_(self.polyak)
                        target_param.data.add_(param.data * (1-self.polyak))
                    for param, target_param in zip(self.lowpolicyValue1.parameters(), self.lowpolicyValueTarget1.parameters()):
                        target_param.data.mul_(self.polyak)
                        target_param.data.add_(param.data * (1-self.polyak))
                    for param, target_param in zip(self.lowpolicyValue2.parameters(), self.lowpolicyValueTarget2.parameters()):
                        target_param.data.mul_(self.polyak)
                        target_param.data.add_(param.data * (1-self.polyak))
        return np.mean(low_q_list), np.mean(ploss_list)

    def goal_relabeling(self, state_seq, goal_seq, action_seq, next_state_batch):
        with torch.no_grad():
            goal_relabel_list = [goal_seq[:, 0], next_state_batch - state_seq[:, 0]]
            for i in range(8):
                goal_relabel_list.append(next_state_batch - state_seq[:, 0] + torch.randn_like(next_state_batch))
            goal_relabel_list = torch.cat([g.unsqueeze(0) for g in goal_relabel_list])
            log_prob = torch.norm(self.lowpolicy(state_seq[:, 0].unsqueeze(0).repeat(10,1,1), goal_relabel_list) - action_seq[:, 0], p = 2, dim = 2, keepdim = True).pow(2)
            for i in range(self.highpolicydelay):
                log_prob += torch.norm(self.lowpolicy(state_seq[:, i].unsqueeze(0).repeat(10,1,1), goal_relabel_list + state_seq[:, 0] - state_seq[:, i]) - action_seq[:, i], p = 2, dim = 2, keepdim = True).pow(2)
        return goal_relabel_list.gather(0, log_prob.min(0, keepdim = True)[1].repeat(1,1,17)).squeeze(0)

    def train_high_polciy(self):
        if len(self.highMemory) < self.batch_size:
            return None, None
        high_q_list, ploss_list = [], []
        for c in range(self.update_every):
            transitions = self.highMemory.sample(self.batch_size)
            batch = MetaTransition(*zip(*transitions))
            state_seq_batch = torch.cat(batch.state_seq)
            goal_seq_batch = torch.cat(batch.goal_seq)
            action_seq_batch = torch.cat(batch.action_seq)
            reward_seq_batch = torch.cat(batch.reward_seq)
            next_state_batch = torch.cat(batch.next_state)

            goal_seq = self.goal_relabeling(state_seq_batch, goal_seq_batch, action_seq_batch, next_state_batch)
            
            with torch.no_grad():
                target_goal = self.highpolicytarget(next_state_batch)
                noise = torch.clamp(torch.randn(target_goal.shape).to(device) * self.target_noise, min=-self.target_clip, max=self.target_clip)
                target_goal = target_goal + noise
                value_target = torch.min(self.highpolicyValueTarget1(next_state_batch, target_goal), self.highpolicyValueTarget2(next_state_batch, target_goal))
                td_target = reward_seq_batch.sum(dim=1, keepdim=True) + self.gamma_meta * value_target
            
            self.highpolicyValueOptim1.zero_grad()
            self.highpolicyValueOptim2.zero_grad()
            state_action1 = self.highpolicyValue1(state_seq_batch[:,0], goal_seq)
            state_action2 = self.highpolicyValue2(state_seq_batch[:,0], goal_seq)
            q_loss1 = F.mse_loss(state_action1, td_target)
            q_loss2 = F.mse_loss(state_action2, td_target)
            q_loss = q_loss1 + q_loss2
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.highpolicyValue1.parameters(), self.gradient_clip)
            torch.nn.utils.clip_grad_norm_(self.highpolicyValue2.parameters(), self.gradient_clip)
            self.highpolicyValueOptim1.step()
            self.highpolicyValueOptim2.step()
            high_q_list.append(q_loss.cpu().detach().numpy())

            if c % self.policy_delay == 0:
                self.highpolicyOptim.zero_grad()
                policy_loss = self.highpolicyValue1(state_seq_batch[:,0], self.highpolicy(state_seq_batch[:,0]))
                policy_loss = -policy_loss.mean()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.highpolicy.parameters(), self.gradient_clip)
                self.highpolicyOptim.step()
                ploss_list.append(policy_loss.cpu().detach().numpy())

                with torch.no_grad():
                    for param, target_param in zip(self.highpolicy.parameters(), self.highpolicytarget.parameters()):
                        target_param.data.mul_(self.polyak)
                        target_param.data.add_(param.data * (1-self.polyak))
                    for param, target_param in zip(self.highpolicyValue1.parameters(), self.highpolicyValueTarget1.parameters()):
                        target_param.data.mul_(self.polyak)
                        target_param.data.add_(param.data * (1-self.polyak))
                    for param, target_param in zip(self.highpolicyValue2.parameters(), self.highpolicyValueTarget2.parameters()):
                        target_param.data.mul_(self.polyak)
                        target_param.data.add_(param.data * (1-self.polyak))
        return  np.mean(high_q_list), np.mean(ploss_list)

env = gym.make('HalfCheetah-v2')
state_n = env.observation_space.shape[0]

agent = HIRO(state_n,env.action_space.shape[0], env.action_space)

scores, p_loss_, q_loss_, hp_loss_, hq_loss_ = [], [], [], [], []
score_avg = 0

num_episodes = 1000
for e in range(num_episodes):
    done = False
    score = 0
    qloss_list, ploss_list, hqloss_list, hploss_list = [], [], [], []
    state = env.reset()
    state = torch.tensor(state, dtype= torch.float32).unsqueeze(0).to(device)
    goal = agent.selectgoal(state, 0.25)
    k = 0
    state_seq, action_seq, reward_seq, goal_seq = [],[],[],[]
    intrinsic_reward_list = []
    for c in count():
        if agent.render:
            env.render()
        action = agent.selectaction(state, goal, 0.25)
        next_state, reward, done, info = env.step(action.squeeze(0).cpu().detach().numpy())
        score += reward
        next_state = torch.tensor(next_state, dtype= torch.float32).unsqueeze(0).to(device)
        reward = torch.tensor([reward], dtype= torch.float32).to(device)
        done = torch.tensor([[1 if done else 0]]).to(device)
        intrinsic_reward = - torch.norm(state + goal - next_state, 2).unsqueeze(0).unsqueeze(0)
        next_goal = state + goal - next_state
        intrinsic_reward_list.append(intrinsic_reward)
        agent.lowMemory.push(state, goal, action, intrinsic_reward, done, next_state, next_goal)
        state_seq.append(state)
        action_seq.append(action)
        reward_seq.append(reward)
        goal_seq.append(goal)
        state = next_state
        goal = next_goal
        k += 1
        qloss, ploss = agent.train_low_polciy()
        if qloss != None:
            qloss_list.append(qloss)
        if ploss != None:
            ploss_list.append(ploss)
        if k == agent.highpolicydelay:
            k = 0
            state_s = torch.cat([s.unsqueeze(0) for s in state_seq], dim = 1)
            action_s = torch.cat([a.unsqueeze(0) for a in action_seq], dim = 1)
            goal_s = torch.cat([g.unsqueeze(0) for g in goal_seq], dim = 1)
            reward_s = torch.cat([r.unsqueeze(0) for r in reward_seq], dim = 1)
            agent.highMemory.push(state_s, goal_s, action_s, reward_s, state)
            state_seq, action_seq, reward_seq, goal_seq = [],[],[],[]
            hqloss, hploss = agent.train_high_polciy()
            if hqloss != None:
                hqloss_list.append(qloss)
            if hploss != None:
                hploss_list.append(ploss)
        if done:
            scores.append(score)
            p_loss_.append(np.mean(ploss_list))
            q_loss_.append(np.mean(qloss_list))
            hp_loss_.append(np.mean(hploss_list))
            hq_loss_.append(np.mean(hqloss_list))

            fig = plt.figure(0, figsize = (6,8))
            plt.clf()
            ax = fig.add_subplot(3,1,1, title = "DDPG - HalfCheetah", xlabel = 'episodes(5000 step)', ylabel = 'score')
            ax.plot(scores, label="score")
            ax.legend()

            ax = fig.add_subplot(3,1,2, title = "HalfCheetah", xlabel = 'episodes(5000 step)')
            ax.plot(p_loss_, label="policy_loss")
            ax.plot(q_loss_, label="q_net1_loss")
            ax.legend()

            ax = fig.add_subplot(3,1,3, title = "HalfCheetah", xlabel = 'episodes(5000 step)')
            ax.plot(hp_loss_, label="high-policy_loss")
            ax.plot(hq_loss_, label="high-q_net_loss")
            ax.legend()
            plt.pause(0.02)
            score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            print("episode : {:3d} | score_avg : {:3.2f} | q_loss = {:3.3f} | p_loss = {:.3f}".format(e, score_avg, np.mean(qloss_list), np.mean(ploss_list)))
            print("intrinsic_reward : ", np.mean(intrinsic_reward_list))
            if score_avg > 900:
                sys.exit()
            break