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

STD_MAX_NOISE = 1

class PolicyNet(nn.Module):
    def __init__(self, observations, actions):
        super(PolicyNet, self).__init__()
        self.policy_fc1 = nn.Linear(observations, 400)
        self.policy_fc2 = nn.Linear(400, 300)
        self.policy_fc3 = nn.Linear(300, 200)
        self.policy_mean_out = nn.Linear(200, actions)
        self.policy_logstd_out = nn.Linear(200, actions)

    def forward(self, x):#, deterministic = False, with_logprob = True):
        h = F.elu(self.policy_fc1(x))
        h = F.elu(self.policy_fc2(h))
        h = F.elu(self.policy_fc3(h))
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
        ch1 = F.elu(self.critic_fc1(torch.cat([x,a], dim = 1)))
        ch2 = F.elu(self.critic_fc2(ch1))
        ch3 = F.elu(self.critic_fc3(ch2))
        return self.critic_out(ch3)

class KL():
    def __init__(self, states_dim, default_states_dim, action_space):
        self.learning_rate = 0.0004
        self.memory_capacity = int(1e+6)
        self.batch_size = 256
        self.action_space = action_space
        self.action_high = torch.tensor(self.action_space.maximum.copy(), dtype = torch.float32).to(device)
        self.alpha = 0.1
        self.gamma = 0.99
        self.polyak = 0.995
        self.train_period = 200
        self.start_train_after = 20000
        self.render = True
        self.gradient_clip = 1.
        self.logprob_clip = 10.
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
            eps = 0 if deterministic else torch.randn_like(std)
            return torch.max(torch.tanh(mean + std * eps) * self.action_high, torch.tensor(self.action_space.minimum.copy(), dtype = torch.float32).to(device))

    def train_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        next_state_batch = torch.cat(batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        """
        mean, std = self.policy_net(state_batch)
        default_mean, default_std = self.default_policy_net(state_batch[:,:self.default_states_dim])
        default_mean_target, default_std_target = self.default_policy_net_target(state_batch[:,:self.default_states_dim])
        pi_distribution = Normal(mean, std)
        pi_zero_distribution = Normal(default_mean, default_std)
        pi_zero_distribution_target = Normal(default_mean_target, default_std_target)
        
        next_mean, next_std = self.policy_net(next_state_batch)
        next_default_mean, next_default_std = self.default_policy_net(next_state_batch[:,:self.default_states_dim])
        next_default_mean_target, next_default_std_target = self.default_policy_net_target(next_state_batch[:,:self.default_states_dim])
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
            target_action = torch.max(torch.tanh(target_action) * self.action_high, torch.tensor(self.action_space.minimum.copy(), dtype = torch.float32).to(device))
            value_target = self.value_net_target(next_state_batch, target_action)
            td_target = reward_batch + self.gamma * (value_target - self.alpha * next_kl_div_term_target.detach()) - self.alpha * kl_div_term_target.detach() 
        
        self.value_optim.zero_grad()
        state_action = self.value_net(state_batch, action_batch)
        q_loss = F.mse_loss(state_action, td_target)
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.value_net.parameters(), self.gradient_clip)
        self.value_optim.step()
        qloss_list.append(q_loss.cpu().detach().numpy())

        self.policy_optim.zero_grad()
        action = pi_distribution.rsample()
        action = torch.max(torch.tanh(action) * self.action_high, torch.tensor(self.action_space.minimum.copy(), dtype = torch.float32).to(device))
        value = self.value_net_target(state_batch, action)
        policy_loss = (self.alpha * (kl_div_term_target + next_kl_div_term_target) - value).mean()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.policy_optim.step()
        ploss_list.append(policy_loss.cpu().detach().numpy())

        self.default_policy_optim.zero_grad()
        default_policy_loss = (kl_div_term + next_kl_div_term).mean()
        nn.utils.clip_grad_norm_(self.default_policy_net.parameters(), self.gradient_clip)
        default_policy_loss.backward()
        self.default_policy_optim.step()
        """
        mean, std = self.policy_net(state_batch)
        default_mean, default_std = self.default_policy_net(state_batch[:,:self.default_states_dim])
        default_mean_target, default_std_target = self.default_policy_net_target(state_batch[:,:self.default_states_dim])
        pi_distribution = Normal(mean, std)
        pi_zero_distribution = Normal(default_mean, default_std)
        pi_zero_distribution_target = Normal(default_mean_target, default_std_target)
        
        next_mean, next_std = self.policy_net(next_state_batch)
        next_default_mean, next_default_std = self.default_policy_net(next_state_batch[:,:self.default_states_dim])
        next_default_mean_target, next_default_std_target = self.default_policy_net_target(next_state_batch[:,:self.default_states_dim])
        next_pi_distribution = Normal(next_mean, next_std)
        next_pi_zero_distribution = Normal(next_default_mean, next_default_std)
        next_pi_zero_distribution_target = Normal(next_default_mean_target, next_default_std_target)

        with torch.no_grad():
            next_mean_target, next_std_target = self.policy_net_target(next_state_batch)
            next_pi_distribution_target = Normal(next_mean_target, next_std_target)
            target_action = next_pi_distribution_target.sample()
            log_prob_next_action = next_pi_distribution.log_prob(target_action).sum(axis=-1, keepdim = True)
            log_prob_next_action -= (2*(np.log(2) - target_action - F.softplus(-2*target_action))).sum(axis=1, keepdim = True)
            log_prob_next_action = torch.clamp(log_prob_next_action, max = self.logprob_clip)
            log_prob_next_action_default = next_pi_zero_distribution_target.log_prob(target_action).sum(axis=-1, keepdim = True)
            log_prob_next_action_default -= (2*(np.log(2) - target_action - F.softplus(-2*target_action))).sum(axis=1, keepdim = True)
            log_prob_next_action_default = torch.clamp(log_prob_next_action_default, max = self.logprob_clip)
            target_action = torch.max(torch.tanh(target_action) * self.action_high, torch.tensor(self.action_space.minimum.copy(), dtype = torch.float32).to(device))
            value_target = self.value_net_target(next_state_batch, target_action)
            td_target = reward_batch + self.gamma * (value_target - self.alpha * (log_prob_next_action - log_prob_next_action_default).detach())# - self.alpha * kl_div_term_target.detach() 
        
        self.value_optim.zero_grad()
        state_action = self.value_net(state_batch, action_batch)
        q_loss = F.mse_loss(state_action, td_target)
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.value_net.parameters(), self.gradient_clip)
        self.value_optim.step()
        qloss_list.append(q_loss.cpu().detach().numpy())

        self.policy_optim.zero_grad()
        action = pi_distribution.rsample()
        log_prob_action = pi_distribution.log_prob(action).sum(axis=-1, keepdim = True)
        log_prob_action -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1, keepdim = True)
        log_prob_action_default = pi_zero_distribution_target.log_prob(action).sum(axis=-1, keepdim = True)
        log_prob_action_default -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1, keepdim = True)
        action = torch.max(torch.tanh(action) * self.action_high, torch.tensor(self.action_space.minimum.copy(), dtype = torch.float32).to(device))
        value = self.value_net_target(state_batch, action)
        policy_loss = (self.alpha * (log_prob_action - log_prob_action_default) - value).mean()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.policy_optim.step()
        ploss_list.append(policy_loss.cpu().detach().numpy())

        self.default_policy_optim.zero_grad()
        sample_action = pi_distribution.rsample().detach()
        log_prob_default_update_action = pi_distribution.log_prob(sample_action).sum(axis=-1, keepdim = True)
        log_prob_default_update_action -= (2*(np.log(2) - target_action - F.softplus(-2*sample_action))).sum(axis=1, keepdim = True)
        log_prob_default_update_action_default = pi_zero_distribution.log_prob(sample_action).sum(axis=-1, keepdim = True)
        log_prob_default_update_action_default -= (2*(np.log(2) - sample_action - F.softplus(-2*sample_action))).sum(axis=1, keepdim = True)
        default_policy_loss = (log_prob_default_update_action.detach() - log_prob_default_update_action_default).mean()
        nn.utils.clip_grad_norm_(self.default_policy_net.parameters(), self.gradient_clip)
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

        return  q_loss.cpu().detach().numpy(), policy_loss.cpu().detach().numpy() 

def state_preprocessing(state):
    return torch.tensor(state.copy(), dtype = torch.float32).unsqueeze(0).to(device)

parser = argparse.ArgumentParser()
parser.add_argument('-domain', action = 'store', default= 'quadruped') #quadruped
parser.add_argument('-task', action = 'store', default= 'fetch') #fetch
parser.add_argument('-time', default = 30, type = int)
parser.add_argument('-default_dim', default = 78, type = int) #78
parser.add_argument('--test', action = 'store_true')
parser.add_argument('--load', action = 'store_true')
args = parser.parse_args()

#env = gym.make('FetchReach-v1')
env = suite.load(domain_name = args.domain, task_name = args.task, environment_kwargs = {'flat_observation' : True}, task_kwargs = {'time_limit' : args.time})
#env.env.reward_type = 'dense'
#env = gym.wrappers.filter_observation.FilterObservation(env, filter_keys=['observation', 'desired_goal'])
#env = gym.wrappers.flatten_observation.FlattenObservation(env)
agent = KL(env.observation_spec()['observations'].shape[0], args.default_dim, env.action_spec())
#vime = VIME(90, env.action_spec().shape[0], device = device, hidden_layer_size = 256)

if args.load:
    agent.policy_net.load_state_dict(torch.load(f'./{args.domain}_{args.task}_pnet.pth'))
    agent.policy_net_target.load_state_dict(torch.load(f'./{args.domain}_{args.task}_pnet.pth'))
    agent.default_policy_net.load_state_dict(torch.load(f'./{args.domain}_{args.task}_dpnet.pth'))
    agent.default_policy_net_target.load_state_dict(torch.load(f'./{args.domain}_{args.task}_dpnet.pth'))
    agent.value_net.load_state_dict(torch.load(f'./{args.domain}_{args.task}_vnet.pth'))
    agent.value_net_target.load_state_dict(torch.load(f'./{args.domain}_{args.task}_vnet.pth'))

scores, episodes = [], []
score_avg = 0

def policy_by_agent(time_step):
    torch_action =  agent.get_action(state_preprocessing(time_step.observation['observations']))
    return torch_action.squeeze(0).cpu().detach().numpy()

if args.test:
    agent.policy_net.load_state_dict(torch.load(f'./{args.domain}_{args.task}_pnet.pth'))
    viewer.launch(env, policy = policy_by_agent)
    exit(0)

interaction_count = 0
reward_list, p_loss_list, q_loss_list = [],[],[]
num_episodes = 10000
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
        #reward_list.append(r)
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

            reward_list.append(score/(c+1))
            p_loss_list.append(ploss_avg)
            q_loss_list.append(qloss_avg)
            if (e+1) % 100 == 0:
                reward_avg = sum(reward_list) / len(reward_list)
                reward_min = min(reward_list)
                reward_max = max(reward_list)
                plt.figure(0)
                plt.clf()
                plt.plot(reward_list, label="reward")
                plt.axhline(reward_avg, linestyle = ':', label = 'avg : {:3f}'.format(reward_avg))
                plt.axhline(reward_min, linestyle = ':', label = 'min : {:3f}'.format(reward_min))
                plt.axhline(reward_max, linestyle = ':', label = 'max : {:3f}'.format(reward_max))
                plt.title("Received Average Rewards")
                plt.legend()
                plt.savefig(f'./imgs/rewards_{e+1}.png')
                reward_list = []

                plt.figure(1)
                plt.clf()
                plt.plot(q_loss_list[-100:], label="qloss")
                plt.plot(p_loss_list[-100:], label="ploss")
                plt.title("Average Losses")
                plt.legend()
                plt.savefig(f'./imgs/losses_{e+1}.png')
            print("episode : {:3d} | end at : {:3d} steps | total interactions : {:7d} | score : {:8.3f} | q_loss = {:7.3f} | p_loss = {:7.3f} ".format(e, c+1, interaction_count, score, qloss_avg, ploss_avg))
            break
    #for _ in range(100):
    #    transitions = agent.replay_buffer.sample(agent.batch_size)
    #    batch = Transition(*zip(*transitions))
    #    next_state_batch = torch.cat(batch.next_state)
    #    state_batch = torch.cat(batch.state)
    #    action_batch = torch.cat(batch.action)
    #    elbo = vime.update_posterior(state_batch, action_batch, next_state_batch)
    if e % 100 == 0:
        torch.save(agent.policy_net.state_dict(), f'./{args.domain}_{args.task}_pnet.pth')
        torch.save(agent.default_policy_net.state_dict(), f'./{args.domain}_{args.task}_dpnet.pth')
        torch.save(agent.value_net.state_dict(), f'./{args.domain}_{args.task}_vnet.pth')
env.close()