import os
import copy
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,ExponentialLR

import numpy as np
import datetime
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy


class SAC(object):
    def __init__(self, state_space,
                      action_space,
                      look_back,
                      gamma = 0.99,
                      tau = 0.005,
                      alpha=0.2,
                      lr = 0.0003,
                      weight_decay = 0,
                      target_update_interval = 1,
                      automatic_entropy_tuning = True,
                      cuda = False,
                      policy_type = "Gaussian",
                      num_episodes = 1000,
                      num_iteration = 1440,
                      hidden_size = 256):

        """
        Args:
            tau - target smoothing coefficient(τ) (default: 0.005)
            alpha - Temperature parameter α determines the relative importance of the entropy\
                    term against the reward (default: 0.2)
            target_update_interval - Value target update per no. of updates per step (default: 1)
            lr - learning rate (default: 0.0003)
            gamma - discount factor
            num_episodes -
            hidden_size -
            automatic_entropy_tuning -
            weight_decay - coefficient of l2 regularization term
        """

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.policy_type = policy_type

        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.device = torch.device("cuda" if cuda else "cpu")

        self.critic = QNetwork(state_space,
                                action_space.shape[0],
                                look_back,
                                hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr,weight_decay=weight_decay)
        # self.critic_lr_scheduler = CosineAnnealingWarmRestarts(self.critic_optim, T_0=10,T_mult=1,eta_min=0)

        self.critic_lr_scheduler = ExponentialLR(self.critic_optim, gamma=0.9996)
        self.critic_target = QNetwork(state_space,
                                    action_space.shape[0],
                                    look_back,
                                    hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                # self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.target_entropy = -6
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=lr)

            self.policy = GaussianPolicy(state_space,
                                        action_space.shape[0],
                                        hidden_size,
                                        look_back,
                                        action_space).to(self.device)

            self.policy_optim = Adam(self.policy.parameters(), lr=lr, weight_decay=weight_decay)


        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(state_space,
                                            action_space.shape[0],
                                            hidden_size,
                                            look_back,
                                            action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr,weight_decay=weight_decay)

        # self.policy_lr_scheduler = CosineAnnealingWarmRestarts(self.policy_optim, T_0=num_episodes,T_mult=1, eta_min=0)
        self.policy_lr_scheduler = ExponentialLR(self.policy_optim, gamma=0.999)


    def select_action(self, state, evaluate=False):
        state =  self.normalize(state)
        state = torch.FloatTensor(state).to(self.device)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def normalize(self, state_batch):
        """
        normalize the states for neural network

        Args:
            state_batch - three element tuple (balance, shares, states)
                        # balance (n, look_back)
                        # shares (n, num_stock, look_back)
                        # states (n, num_stock, look_back , feature_num)

        Returns:
            state_batch - matrix form of states
                       # balance (n,num_stock, look_back,     1   )
                       # shares (n, num_stock, look_back,     1   )
                       # states (n, num_stock, look_back , feature_num)
                       # states (n, num_stock, look_back, 1+num_stock+feature_num)
        """
        balance, shares, states = map(np.stack, zip(*state_batch))
        # balance_ = copy.deepcopy(balance)
        # shares_ = copy.deepcopy(shares)
        # states_ = copy.deepcopy(states)


        if len(balance.shape) == 1:
            balance = np.expand_dims(balance, axis = 0)
        if len(shares.shape) == 2:
            shares = np.expand_dims(shares, axis = 0)
        if len(states.shape) == 3:
            states = np.expand_dims(states, axis = 0)
        first_time_balance = np.expand_dims(balance[:,0],axis=-1)
        balance = np.divide(balance, first_time_balance,
                    out=np.zeros_like(balance), where=first_time_balance!=0)

        # normalize the last four price vector by its first value
        states[:,:,:,-4:] /= np.expand_dims(states[:,:,0,-4:],axis=2)

        # replicate balance vector to each asset
        balance = np.tile(balance, (1,shares.shape[1],1))

        balance = np.reshape(balance, (shares.shape[0],shares.shape[1],-1,1))
        shares = np.reshape(shares, (shares.shape[0],shares.shape[1],-1,1))
        # states = np.reshape(states, (states.shape[0],-1))


        return np.concatenate([balance, shares, states],axis = -1)

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        # print("action_batch",action_batch)
        # print("state_batch",state_batch)
        state_batch = self.normalize(state_batch)
        next_state_batch = self.normalize(next_state_batch)

        state_batch = torch.FloatTensor(state_batch).to(self.device)

        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        clip_grad_norm_(self.critic.parameters(), 5)
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        # print('min_qf_pi',min_qf_pi,"state", state_batch)
        # print("policy_loss", policy_loss, "alpha", self.alpha)
        self.policy_optim.zero_grad()
        policy_loss.backward()

        clip_grad_norm_(self.policy.parameters(), 5)
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/{}_sac_checkpoint_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                                        env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
