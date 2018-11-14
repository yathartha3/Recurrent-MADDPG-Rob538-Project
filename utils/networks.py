import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out


class RNNNetwork(nn.Module):
    """
    RNN network. Experimenting with using it as both policy and value function.
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):

        super(RNNNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.fc1 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        # input to hidden unit
        self.i2h = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.i2o = nn.Linear(input_dim + hidden_dim, out_dim)
        self.nonlin = nonlin

        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.i2o.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        # Preprocess X into obs for t, t-1 and t-2
        hist_tminus_0 = torch.tensor(np.zeros((1,18)), dtype=torch.float)
        hist_tminus_1 = torch.tensor(np.zeros((1,18)), dtype=torch.float)
        hist_tminus_2 = torch.tensor(np.zeros((1,18)), dtype=torch.float)

        hist_tminus_0[0] = X[0][0:18]
        hist_tminus_1[0] = X[0][18:36]
        hist_tminus_2[0] = X[0][36:54]

        init_memory = torch.tensor(np.zeros((1,self.hidden_dim)), dtype=torch.float)
        # init_memory = np.zeros(self.hidden_dim)
        # init_memory = torch.tensor(init_memory, dtype=torch.float)

        # X = torch.cat([hist_tminus_2.unsqueeze(-1), init_memory], 1)
        X = torch.cat((hist_tminus_2, init_memory), 1)  # this seemed to be working when I was at apt
        X = self.fc1(X)
        X = torch.cat((hist_tminus_1, X), 1)
        X = self.i2h(X)
        X = torch.cat((hist_tminus_0, X), 1)
        X = self.i2o(X)
        X = self.out_fn(X)
        return X

        # X = self.fc1(X)
        # combined = torch.cat((X, hidden), 1)
        # # update the hidden states
        # hidden = self.i2h(combined)
        # # get output
        # output = self.i2o(combined)
        # output = self.out_fn(output)
        # return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

