import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

# gpu = True
# if gpu:
#     torch.set_default_tensor_type(torch.cuda.FloatTensor)
#     device = 'cuda'
# else:
#     torch.set_default_tensor_type(torch.FloatTensor)
#     device = 'cpu'

class Spiking(nn.Module):

    def __init__(self, width, v_rest=0, v_thresh=1, v_decay=0.9, reset_by_sub=True, non_spiking=False):
        super(Spiking, self).__init__()
        # self.v = torch.nn.Parameter(torch.Tensor([v_rest for i in range(width)]), requires_grad=False)
        # self.spiked = torch.nn.Parameter(torch.Tensor([0 for i in range(width)]), requires_grad=False)
        self.v = torch.Tensor([v_rest for i in range(width)])
        self.spiked = torch.Tensor([0 for i in range(width)])
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_decay = v_decay
        self.reset_by_sub = reset_by_sub
        self.non_spiking = non_spiking

    def check_spike(self):
        self.spiked = torch.Tensor(self.v > self.v_thresh).type(torch.float32)
        if self.reset_by_sub:
            self.v -= self.spiked * (self.v_thresh - self.v_rest)
        else:
            self.v -= self.spiked * (self.v + self.v_rest)

    def forward(self, x):
        self.v *= self.v_decay
        self.v += x
        if not self.non_spiking:
            self.check_spike()
            return self.spiked
        else:
            return self.v


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
    def __init__(self, input_size, hidden_size, num_classes, p=0.5, dropout=False,
                 non_spiking_output=False, min_exp=0, max_exp=2):
        super(NeuralNet, self).__init__()

        self.dropout = dropout
        self.layer = nn.ModuleList()
        self.layer.append(nn.Linear(input_size, hidden_size[0], bias=False))

        self.activation = nn.ModuleList()
        self.activation.append(Spiking(hidden_size[0]))

        self.pde_params = [
            # "pre_firing",
            # "post_firing",
            "pre_spiked",
            "post_spiked",
            # "pre_time",
            # "post_time",
            # "correlation",
            "weight",
            "error",
        ]
        self.min_exp = min_exp
        self.max_exp = max_exp
        self.exp_range = max_exp - min_exp
        self.number_of_coef = np.power(self.exp_range+1, len(self.pde_params))
        self.pde_coef = np.random.uniform(-2, 2, self.number_of_coef)

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
        self.activation.append(Spiking(num_classes, non_spiking=non_spiking_output))

        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def scale_weights(self, scale):
        for l in range(len(self.layer)):
            self.layer[l].weight.data += scale
            # self.layer[l].bias.data += scale

    def set_pde_params(self, setting_type='normal', start=0, stop=1):
        if setting_type == 'normal':
            self.pde_coef = np.random.normal(start, stop, self.number_of_coef)
        elif setting_type == 'ES':
            self.pde_coef = np.random.normal(start, stop)
        elif setting_type == 'uniform':
            self.pde_coef = np.random.uniform(-2, 2, self.number_of_coef)
        else:
            self.pde_coef = setting_type

    def pde_calc(self, pde_variables):
        all_exp = np.array([[torch.pow(pde_variables[p], exponent=e) for p in range(len(pde_variables))]
                             for e in range(self.min_exp, self.max_exp+1)])
        indexes = list(itertools.product(*[
            list(range(self.min_exp, self.max_exp+1)) for i in range(len(self.pde_params))]))
        delta_w = 0
        for idx, c in enumerate(self.pde_coef):
            base = torch.ones_like(all_exp[0, 0])
            for coef, i in enumerate(indexes[idx]):
                base *= all_exp[i, coef]
            delta_w += c * base
        return delta_w

    def update_weights(self, pre_spikes, post_spikes, old_weights, error):
        pde_var = []
        pre_size = len(pre_spikes)
        post_size = len(post_spikes)
        if "pre_firing" in self.pde_params:
            pde_var.append(torch.stack([pre_spikes for i in range(post_size)]))
        if "post_firing" in self.pde_params:
            pde_var.append(torch.stack([pre_spikes for i in range(post_size)]))
        if "pre_spiked" in self.pde_params:
            pde_var.append(torch.vstack([pre_spikes for i in range(post_size)]))
        if "post_spiked" in self.pde_params:
            pde_var.append(torch.hstack([post_spikes.unsqueeze(1) for i in range(pre_size)]))
        if "pre_time" in self.pde_params:
            pde_var.append(pre_spikes)
        if "post_time" in self.pde_params:
            pde_var.append(pre_spikes)
        if "correlation" in self.pde_params:
            pde_var.append(pre_spikes)
        if "weight" in self.pde_params:
            pde_var.append(old_weights)
        if "error" in self.pde_params:
            pde_var.append(torch.ones_like(old_weights)*error)
        new_weights = self.pde_calc(pde_var)
        return new_weights

    def forward(self, x):
        out = self.layer[0](x)
        out = self.activation[0](out)
        for i in range(1, len(self.layer) - 1):
            out = self.layer[i](out)
            out = self.activation[i](out)
        out = self.layer[-1](out)
        out = self.activation[-1](out)
        # out = self.LogSoftmax(out)
        return out

    def timed_forward(self, x):
        out = self.layer[0](x)
        previous_spikes = self.activation[0].spiked
        out = self.activation[0](out)
        for i in range(1, len(self.layer) - 1):
            next_spikes = self.activation[i].spiked
            out = self.layer[i](previous_spikes)
            out = self.activation[i](out)
            previous_spikes = next_spikes
        out = self.layer[-1](previous_spikes)
        out = self.activation[-1](out)
        # out = self.LogSoftmax(out)
        return out

    def pde_forward(self, x, error, lr):
        out = self.layer[0](x)
        previous_spikes = self.activation[0].spiked
        out = self.activation[0](out)
        d_w = self.update_weights(x, out, self.layer[0].weight.data, error)
        self.layer[0].weight.data += d_w * lr
        for i in range(1, len(self.layer) - 1):
            next_spikes = self.activation[i].spiked
            out = self.layer[i](previous_spikes)
            out = self.activation[i](out)
            d_w = self.update_weights(previous_spikes, out, self.layer[i].weight.data, error)
            self.layer[i].weight.data += d_w * lr
            previous_spikes = next_spikes
        out = self.layer[-1](previous_spikes)
        d_w = self.update_weights(previous_spikes, out, self.layer[-1].weight.data, error)
        self.layer[-1].weight.data += d_w * lr
        out = self.activation[-1](out)
        # out = self.LogSoftmax(out)
        return out