import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from encoders import CausalMHAEncoder, LSTMEncoder

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        state = torch.flatten(state, start_dim = 1)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs,
                    num_actions,
                    sequence_length,
                    hidden_dim,
                    type = 'transformer'):
        super(QNetwork, self).__init__()

        # Q1 architecture
        if type == 'transformer':
            self.encoder1 = CausalMHAEncoder(num_inputs*num_actions, sequence_length, hidden_dim)
            self.encoder2 = CausalMHAEncoder(num_inputs*num_actions, sequence_length, hidden_dim)
            next_dim = num_inputs*num_actions*sequence_length
        elif type == 'lstm':
            self.encoder1 = LSTMEncoder(num_inputs*num_actions,hidden_dim)
            self.encoder2 = LSTMEncoder(num_inputs*num_actions,hidden_dim)
            next_dim = hidden_dim*num_actions*sequence_length
        else:
            self.encoder1 = nn.Identity()
            self.encoder2 = nn.Identity()
            next_dim = num_inputs*num_actions*sequence_length

        self.linear0 = nn.Linear(next_dim,hidden_dim)
        self.linear1 = nn.Linear(hidden_dim + num_actions, hidden_dim)
        # self.ln1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture

        self.linear4 = nn.Linear(next_dim,hidden_dim)
        self.linear5 = nn.Linear(hidden_dim + num_actions, hidden_dim)
        # self.ln2 = nn.LayerNorm(hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        state = state.permute([0,2,1,3])      # change the length to the second dimension
        state = state.view(state.size(0),state.size(1), -1)

        state1  = self.encoder1(state)
        state1 = torch.flatten(state1, start_dim = 1)

        state1 = F.relu(self.linear0(state1))
        xu = torch.cat([state1, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)


        #############################################
        state2 = self.encoder2(state)
        state2 = torch.flatten(state2, start_dim = 1)

        state2 = F.relu(self.linear4(state2))
        xu = torch.cat([state2, action], 1)

        x2 = F.relu(self.linear5(xu))
        x2 = F.relu(self.linear6(x2))
        x2 = self.linear7(x2)
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs,
                        num_actions,
                        hidden_dim,
                        sequence_length,
                        type = 'transformer',
                        action_space=None):
        super(GaussianPolicy, self).__init__()
        if type == 'transformer':
            self.encoder = CausalMHAEncoder(num_inputs*num_actions, sequence_length, hidden_dim)
            next_dim = num_inputs*num_actions*sequence_length
        elif type == 'lstm':
            self.encoder = LSTMEncoder(num_inputs*num_actions,hidden_dim)
            next_dim = hidden_dim*num_actions*sequence_length
        else:
            self.encoder = nn.Identity()
            next_dim = num_inputs*num_actions*sequence_length
        self.linear1 = nn.Linear(next_dim, hidden_dim)
        # self.ln1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.ln2 = nn.LayerNorm(hidden_dim)

        # self.ln1 = nn.LayerNorm(hidden_dim)
        # self.ln2 = nn.LayerNorm(hidden_dim)
        self.mean_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear2 = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.log_std_linear2 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):

        state = state.permute([0,2,1,3])      # change the length to the second dimension
        state = state.view(state.size(0),state.size(1), -1)

        state = self.encoder(state)


        state = torch.flatten(state, start_dim = 1)

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear1(x)
        mean = self.mean_linear2(mean)
        log_std = self.log_std_linear1(x)
        log_std = self.log_std_linear2(log_std)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs,
                       num_actions,
                       hidden_dim,
                       sequence_length,
                       type = 'transformer',
                       action_space=None):
        super(DeterministicPolicy, self).__init__()

        if type == 'transformer':
            self.encoder = CausalMHAEncoder(num_inputs*num_actions, sequence_length, hidden_dim)
            next_dim = num_inputs*num_actions*sequence_length
        elif type == 'lstm':
            self.encoder = LSTMEncoder(num_inputs*num_actions,hidden_dim)
            next_dim = hidden_dim*num_actions*sequence_length
        else:
            self.encoder = nn.Identity()
            next_dim = num_inputs*num_actions*sequence_length

        self.linear0 = nn.Linear(next_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)


        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        state = state.permute([0,2,1,3])      # change the length to the second dimension
        state = state.view(state.size(0),state.size(1), -1)

        state = self.encoder(state)

        state = torch.flatten(state, start_dim = 1)

        state = F.relu(self.linear0(state))
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
