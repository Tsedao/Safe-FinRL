import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

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
            self.encoder1 = CasualMHAEncoder(num_inputs*num_actions, sequence_length, hidden_dim)
            self.encoder2 = CasualMHAEncoder(num_inputs*num_actions, sequence_length, hidden_dim)
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
            self.encoder = CasualMHAEncoder(num_inputs*num_actions, sequence_length, hidden_dim)
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
            self.encoder = CasualMHAEncoder(num_inputs*num_actions, sequence_length, hidden_dim)
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




class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        self.apply(weights_init_)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, seq_len, dropout_rate=0.6, head_size=8):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)



        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size,
                                      bias=False)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len))
                                     .view(1, 1, seq_len, seq_len))
        self.apply(weights_init_)


    def forward(self, q, k, v, cache=None):
        B, T, C = q.size()
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        if cache is not None and 'encdec_k' in cache:
            k, v = cache['encdec_k'], cache['encdec_v']
        else:
            k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
            v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

            if cache is not None:
                cache['encdec_k'], cache['encdec_v'] = k, v

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        q.mul_(self.scale)
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        x.masked_fill_(self.mask[:,:,:T,:T] == 0, float('-inf'))
        x = torch.softmax(x, dim=-1)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class CasualMHAEncoder(nn.Module):
    def __init__(self, hidden_size, seq_len, filter_size, depth = 3, dropout_rate=0.6):
        """
        Args:
            hidden_dim: d_model dimension of feature in sequence data
            seq_len: the length of sequence data
            filter_size: hidden_dim in FeedForwardNetwork
        """
        super(CasualMHAEncoder, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, seq_len, dropout_rate=0.6)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        self.depth = depth

    def forward(self, x):  # pylint: disable=arguments-differ

        # Smaller transformer that only share paremters
        for i in range(self.depth):
            y = self.self_attention_norm(x)
            y = self.self_attention(y, y, y)
            y = self.self_attention_dropout(y)
            x = x + y

            y = self.ffn_norm(x)
            y = self.ffn(y)
            y = self.ffn_dropout(y)
            x = x + y
        return x

class LSTMEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2, batch_first=True):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first, bidirectional=False)


    def forward(self, input):

        out, (h_n, c_n) = self.lstm(input)

        return out
