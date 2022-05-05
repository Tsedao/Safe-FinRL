import torch
import torch.nn as nn



class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        # self.apply(weights_init_)

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
        ## causal mask
        # tensor([[[[1., 0., 0., 0., 0.],
        #           [1., 1., 0., 0., 0.],
        #           [1., 1., 1., 0., 0.],
        #           [1., 1., 1., 1., 0.],
        #           [1., 1., 1., 1., 1.]]]])
        ##
        # self.apply(weights_init_)


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

        q.mul_(self.scale)        # scale q first
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        # causal self-attention; Self-attend:
        x.masked_fill_(self.mask[:,:,:T,:T] == 0, float('-inf'))
        x = torch.softmax(x, dim=-1)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class CausalMHABlock(nn.Module):
    def __init__(self, hidden_size, seq_len, filter_size, depth = 1, dropout_rate=0.6):
        """
        Args:
            hidden_dim: 'd_model' dimension of feature in sequence data
            seq_len: the length of sequence data
            filter_size: hidden_dim in FeedForwardNetwork
        """
        super(CausalMHABlock, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, seq_len, dropout_rate=0.6)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        self.depth = depth

    def forward(self, x):  # pylint: disable=arguments-differ

        # a compressed transformer that shares paremters
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

class CausalMHAEncoder(nn.Module):
    def __init__(self, hidden_size,
                        seq_len,
                        filter_size,
                        lay_nums = 3,
                        depth = 2,
                        dropout_rate=0.6):
        """
        Args:
            hidden_dim: 'd_model' dimension of feature in sequence data
            seq_len: the length of sequence data
            filter_size: hidden_dim in FeedForwardNetwork
            depth: depth of MHA that shares parameters
            lay_nums: depth of MHA that NOT shares parameters
        """
        super(CausalMHAEncoder, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(lay_nums):
            self.layers.append(CausalMHABlock(hidden_size,
                                                seq_len,
                                                filter_size,
                                                depth,
                                                dropout_rate))

        # self.apply(weights_init_)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LSTMEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2, batch_first=True):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first, bidirectional=False)


    def forward(self, input):

        out, (h_n, c_n) = self.lstm(input)

        return out
