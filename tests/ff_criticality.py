import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Spiking(nn.Module):

    def __init__(self, width, v_rest=0, v_thresh=1, v_decay=0.9, reset_by_sub=True):
        super(Spiking, self).__init__()
        self.v = torch.nn.Parameter(torch.Tensor([v_rest for i in range(width)]))
        self.spiked = torch.nn.Parameter(torch.Tensor([0 for i in range(width)]))
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_decay = v_decay
        self.reset_by_sub = reset_by_sub

    def check_spike(self):
        self.spiked = self.v > self.v_thresh
        if self.reset_by_sub:
            self.v -= self.spiked * (self.v_thresh - self.v_rest)
        else:
            self.v -= self.spiked * (self.v + self.v_rest)

    def forward(self, x):
        self.v *= self.v_decay
        self.v += x
        self.check_spike()
        return self.spiked


class MyDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(MyDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, x):
        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
            mask = binomial.sample(x.size())
            return x * mask * (1.0 / (1 - self.p))
        return x


class MyLinear(nn.Linear):
    def __init__(self, in_feats, out_feats, drop_p, bias=True, drop_input=True):
        super(MyLinear, self).__init__(in_feats, out_feats, bias=bias)
        self.drop_input = drop_input
        self.custom_dropout = MyDropout(p=drop_p)

    def forward(self, input):
        if self.drop_input:
            dropout_value = self.custom_dropout(input)
            return F.linear(dropout_value, self.weight, self.bias)
        else:
            dropout_value = self.custom_dropout(self.weight)
            return F.linear(input, dropout_value, self.bias)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, p, dropout=True):
        super(NeuralNet, self).__init__()

        self.dropout = dropout
        self.layer = nn.ModuleList()
        self.layer.append(nn.Linear(input_size, hidden_size[0]))

        self.activation = nn.ModuleList()
        self.activation.append(Spiking(hidden_size[0]))

        for i in range(len(hidden_size)-1):
            if dropout:
                self.layer.append(MyLinear(hidden_size[i], hidden_size[i+1], p))
            else:
                self.layer.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            self.activation.append(Spiking(hidden_size[i+1]))

        if dropout:
            self.layer.append(MyLinear(hidden_size[-1], num_classes, p))
        else:
            self.layer.append(nn.Linear(hidden_size[-1], num_classes))
        self.activation.append(Spiking(num_classes))

        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer[0](x)
        out = self.activation[0](out)
        for i in range(1, len(self.layer) - 1):
            if self.dropout:
                out = self.layer[i](out)
            else:
                out = self.layer[i](out)
            out = self.activation[i](out)
        if self.dropout:
            out = self.layer[-1](out)
        else:
            out = self.layer[-1](out)
        out = self.activation[-1](out)
        # out = self.LogSoftmax(out)
        return out


