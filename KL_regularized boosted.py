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
                        ('state_seq', 'action_seq', 'log_prob_seq', 'reward_seq', 'next_state'))

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
    def __init__(self, observations, actions):
        super(PolicyNet, self).__init__()
        self.policy_fc1 = nn.Linear(observations, 300)
        self.policy_fc2 = nn.Linear(300, 300)
        self.policy_fc3 = nn.Linear(300, 200)
        self.policy_mean_out = nn.Linear(200, actions)
        self.policy_logstd_out = nn.Linear(200, actions)

    def forward(self, x):#, deterministic = False, with_logprob = True):
        h = F.relu(self.policy_fc1(x))
        h = F.relu(self.policy_fc2(h))
        h = F.relu(self.policy_fc3(h))
        mean = torch.tanh(self.policy_mean_out(h))
        std = 0.1 + (STD_MAX_NOISE - 0.1) * torch.sigmoid(self.policy_logstd_out(h))
        return mean, std
        """
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
        """

class ValueNet(nn.Module):
    def __init__(self, observations, actions):
        super(ValueNet, self).__init__()
        self.critic_fc1 = nn.Linear(observations + actions, 400)
        self.critic_fc2 = nn.Linear(400, 400)
        self.critic_fc3 = nn.Linear(400, 300)
        self.critic_out = nn.Linear(300, 1)

    def forward(self, x, a):
        ch1 = F.relu(self.critic_fc1(torch.cat([x,a], dim = -1)))
        ch2 = F.relu(self.critic_fc2(ch1))
        ch3 = F.relu(self.critic_fc3(ch2))
        return self.critic_out(ch3)

class KL():
    def __init__(self, states_dim, default_states_dim, action_space):
        self.learning_rate = 0.0005
        self.memory_capacity = int(2e+6)
        self.batch_size = 64
        self.action_space = action_space
        self.rollout = 10
        self.alpha = 0.1
        self._lambda = 1
        self.gamma = 0.9
        self.polyak = 0.995
        self.update_period = 50
        self.update_times = 50
        self.update_after = 5000
        self.render = True
        self.default_states_dim = default_states_dim

        self.policy_net = PolicyNet(states_dim,action_space.shape[0]).to(device)
        self.policy_net_target = PolicyNet(states_dim,action_space.shape[0]).to(device)
        self.policy_net_target.load_state_dict(self.policy_net.state_dict())
        self.policy_optim = optim.Adam(self.policy_net.parameters(), lr= self.learning_rate)

        self.default_policy_net = PolicyNet(default_states_dim, action_space.shape[0]).to(device)
        self.default_policy_net_target = PolicyNet(default_states_dim, action_space.shape[0]).to(device)
        self.default_policy_net_target.load_state_dict(self.default_policy_net.state_dict())
        self.default_policy_optim = optim.Adam(self.default_policy_net.parameters(), lr = self.learning_rate)

        self.value_net = ValueNet(states_dim,action_space.shape[0]).to(device)
        self.value_net_target = ValueNet(states_dim,action_space.shape[0]).to(device)
        self.value_net_target.load_state_dict(self.value_net.state_dict())
        self.value_optim = optim.Adam(self.value_net.parameters(), lr= self.learning_rate)

        for param in self.value_net_target.parameters():
            param.requires_grad = False
        for param in self.policy_net_target.parameters():
            param.requires_grad = False
        for param in self.default_policy_net_target.parameters():
            param.requires_grad = False
        self.replay_buffer = ReplayMemory(self.memory_capacity)

    def get_action(self, state, deterministic = False):
        with torch.no_grad():
            mean, std = self.policy_net(state)
            pi_distribution = Normal(mean, std)
            action = pi_distribution.sample()
            return torch.max(torch.min(mean if deterministic else action, torch.tensor(self.action_space.maximum, dtype = torch.float32).to(device)), torch.tensor(self.action_space.minimum, dtype = torch.float32).to(device)), pi_distribution.log_prob(action).sum()

    def train_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        qloss_list, ploss_list = [],[]
        for c in range(self.update_times):
            transitions = self.replay_buffer.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            next_state_batch = torch.cat(batch.next_state)
            state_batch = torch.cat(batch.state_seq)
            action_batch = torch.cat(batch.action_seq)
            log_prob_batch = torch.cat(batch.log_prob_seq)
            reward_batch = torch.cat(batch.reward_seq)
            
            mean, std = self.policy_net(state_batch)
            default_mean, default_std = self.default_policy_net(state_batch[...,:self.default_states_dim])
            default_mean_target, default_std_target = self.default_policy_net_target(state_batch[...,:self.default_states_dim])
            pi_distribution = Normal(mean, std)
            pi_zero_distribution = Normal(default_mean, default_std)
            pi_zero_distribution_target = Normal(default_mean_target, default_std_target)
            
            next_mean, next_std = self.policy_net(next_state_batch)
            next_default_mean, next_default_std = self.default_policy_net(next_state_batch[...,:self.default_states_dim])
            next_default_mean_target, next_default_std_target = self.default_policy_net_target(next_state_batch[...,:self.default_states_dim])
            next_pi_distribution = Normal(next_mean, next_std)
            next_pi_zero_distribution = Normal(next_default_mean, next_default_std)
            next_pi_zero_distribution_target = Normal(next_default_mean_target, next_default_std_target)

            pi_distribution_grad_cut = Normal(mean.detach(), std.detach())
            next_pi_distribution_grad_cut = Normal(next_mean.detach(), next_std.detach())
            kl_div_term = torch.distributions.kl.kl_divergence(pi_distribution_grad_cut, pi_zero_distribution).sum(axis = -1, keepdim = True)
            next_kl_div_term = torch.distributions.kl.kl_divergence(next_pi_distribution_grad_cut, next_pi_zero_distribution).sum(axis = -1, keepdim = True)
            kl_div_term_target = torch.distributions.kl.kl_divergence(pi_distribution, pi_zero_distribution_target).sum(axis = -1, keepdim = True)
            next_kl_div_term_target = torch.distributions.kl.kl_divergence(next_pi_distribution, next_pi_zero_distribution_target).sum(axis = -1, keepdim = True)

            with torch.no_grad():
                next_mean_target, next_std_target = self.policy_net_target(next_state_batch)
                next_pi_distribution_target = Normal(next_mean_target, next_std_target)
                target_action = next_pi_distribution_target.sample()
                target_action = torch.max(torch.min(target_action, torch.tensor(self.action_space.maximum, dtype=torch.float32).to(device)), torch.tensor(self.action_space.minimum, dtype=torch.float32).to(device))
                value_target = (self.value_net_target(next_state_batch, target_action) - next_kl_div_term_target).unsqueeze(1)
                #print(value_target, value_target.shape)
                target_q = reward_batch + value_target
                #print(reward_batch, reward_batch.shape)
                next_expected_q = torch.cat([target_q, value_target], dim = 1)[:,1:]
                td_error = reward_batch + self.gamma * next_expected_q - target_q
                #print(td_error, td_error.shape)
                retrace_op = (pi_distribution.log_prob(action_batch).sum(dim = -1, keepdim = True) - log_prob_batch).clamp_min(0).exp().multiply(self._lambda)
                print(retrace_op.view(-1))
                print((F.pad(retrace_op[:,(1):].cumprod(1), (0,0,1,0), value = 1) * torch.pow(self.gamma, torch.arange(self.rollout - 0).unsqueeze(0).unsqueeze(2))).view(-1), retrace_op.shape)
                #retrace_q_tmp_list = []
                #for i in range(self.rollout):
                #    retrace_q_tmp_list.append((td_error[:,i:] * F.pad(retrace_op[:,(i+1):].cumprod(1), (0,0,1,0), value = 1) * torch.pow(self.gamma, torch.arange(self.rollout - i).unsqueeze(0).unsqueeze(2))).sum(dim = 1, keepdim = True))
                #retrace_q = torch.cat(retrace_q_tmp_list, dim = 1) + target_q
                #print(retrace_q, retrace_q.shape)
                retrace_q_tmp_list = [td_error[:,9].unsqueeze(1)]
                tmp = td_error[:,9]
                for i in range(self.rollout - 1):
                    tmp = tmp * self.gamma * retrace_op[:,9 - i] + td_error[:, 8 - i]
                    retrace_q_tmp_list.append(tmp.unsqueeze(1))
                retrace_q = torch.cat(list(reversed(retrace_q_tmp_list)), dim = 1) + target_q
                #print(retrace_q, retrace_q.shape)

            self.policy_optim.zero_grad()
            action = pi_distribution.rsample()
            value = self.value_net_target(state_batch, action)
            policy_loss = self.alpha * torch.cat((kl_div_term_target, next_kl_div_term_target.unsqueeze(1)), dim = 1).mean()- value.mean()
            policy_loss.backward()
            self.policy_optim.step()
            ploss_list.append(policy_loss.cpu().detach().numpy())

            self.value_optim.zero_grad()
            state_action = self.value_net(state_batch, action_batch)
            print(state_action.view(-1), retrace_q.view(-1))
            q_loss = F.mse_loss(state_action, retrace_q)
            q_loss.backward()
            print(q_loss.item())
            self.value_optim.step()
            qloss_list.append(q_loss.cpu().detach().numpy())

            self.default_policy_optim.zero_grad()
            default_policy_loss = torch.cat((kl_div_term, next_kl_div_term.unsqueeze(1)), dim = 1).mean()
            default_policy_loss.backward()
            self.default_policy_optim.step()
            
            with torch.no_grad():
                for param, target_param in zip(self.value_net.parameters(), self.value_net_target.parameters()):
                    target_param.data.mul_(self.polyak)
                    target_param.data.add_(param.data * (1-self.polyak))
                for param, target_param in zip(self.policy_net.parameters(), self.policy_net_target.parameters()):
                    target_param.data.mul_(self.polyak)
                    target_param.data.add_(param.data * (1-self.polyak))
                for param, target_param in zip(self.default_policy_net.parameters(), self.default_policy_net_target.parameters()):
                    target_param.data.mul_(self.polyak)
                    target_param.data.add_(param.data * (1-self.polyak))
        
        return  np.mean(qloss_list), np.mean(ploss_list)

def state_preprocessing(state):
    return torch.tensor(state.copy(), dtype = torch.float32).unsqueeze(0).to(device)

parser = argparse.ArgumentParser()
parser.add_argument('-domain', action = 'store', default= 'quadruped')
parser.add_argument('-task', action = 'store', default= 'fetch')
parser.add_argument('--test', action = 'store_true')
args = parser.parse_args()

#env = gym.make('FetchReach-v1')
env = suite.load(domain_name = args.domain, task_name = args.task, environment_kwargs = {'flat_observation' : True}, task_kwargs = {'time_limit' : 40})
#env.env.reward_type = 'dense'
#env = gym.wrappers.filter_observation.FilterObservation(env, filter_keys=['observation', 'desired_goal'])
#env = gym.wrappers.flatten_observation.FlattenObservation(env)
agent = KL(90, 78, env.action_spec())
#vime = VIME(90, env.action_spec().shape[0], device = device, hidden_layer_size = 256)

scores, episodes = [], []
score_avg = 0

def policy_by_agent(time_step):
    torch_action, _ =  agent.get_action(state_preprocessing(time_step.observation['observations']))
    return torch_action.squeeze(0).cpu().detach().numpy()

if args.test:
    agent.policy_net.load_state_dict(torch.load(f'./{args.domain}_{args.task}_pnet.pth'))
    viewer.launch(env, policy = policy_by_agent)
    exit(0)

interaction_count = 0
num_episodes = 1000
for e in range(num_episodes):
    done = False
    score = 0
    qloss_list, ploss_list = [], []
    state_seq, action_seq, log_prob_seq, reward_seq = [], [], [], []
    time_step = env.reset()
    state = state_preprocessing(time_step.observation['observations'])

    for c in count():
        if interaction_count <= agent.update_after:
            action = np.random.uniform(env.action_spec().minimum, env.action_spec().maximum, env.action_spec().shape)
            action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
            log_prob = -np.log(env.action_spec().maximum - env.action_spec().minimum).sum()
        else:
            action, log_prob = agent.get_action(state)
        
        a = action.squeeze(0).cpu().detach().numpy()
        time_step = env.step(a)
        
        next_state = state_preprocessing(time_step.observation['observations'])
        r = time_step.reward
        score += r
        log_prob = torch.tensor([[log_prob]], dtype = torch.float32).to(device)
        reward = torch.tensor([[r]], dtype=torch.float32).to(device)

        state_seq.append(state)
        action_seq.append(action)
        log_prob_seq.append(log_prob)
        reward_seq.append(reward)
        if (c+1)%agent.rollout == 0:
            agent.replay_buffer.push(torch.cat(state_seq).unsqueeze(0), torch.cat(action_seq).unsqueeze(0), torch.cat(log_prob_seq).unsqueeze(0), torch.cat(reward_seq).unsqueeze(0), next_state)
            state_seq, action_seq, log_prob_seq, reward_seq = [], [], [], []

        state = next_state

        if interaction_count % agent.update_period == 0 and interaction_count > agent.update_after:
            qloss, ploss = agent.train_model()
            if qloss != None:
                qloss_list.append(qloss)
            if ploss != None:
                ploss_list.append(ploss)
        interaction_count += 1
        if time_step.step_type == 2:
            qloss_avg = 0.0 if len(qloss_list) == 0 else np.mean(qloss_list)
            ploss_avg = 0.0 if len(ploss_list) == 0 else np.mean(ploss_list)
            print("episode : {:3d} | end at : {:3d} steps | total interactions : {:7d} | score : {:8.3f} | q_loss = {:7.3f} | p_loss = {:7.3f} ".format(e, c+1, interaction_count, score, qloss_avg, ploss_avg))
            break
    #torch.save(agent.policy_net.state_dict(), f'./{args.domain}_{args.task}_pnet.pth')
    #torch.save(agent.default_policy_net.state_dict(), f'./{args.domain}_{args.task}_dpnet.pth')
    #torch.save(agent.value_net.state_dict(), f'./{args.domain}_{args.task}_vnet.pth')
env.close()