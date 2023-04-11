import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.build_mlp import make_mlp
from torch.autograd import Variable

class DeepNet6LayerTune(nn.Module): 
    """
    define a  6 fully connected layer neural network for hparams tunning 5 hiddlen layer + 1 output 
    """

    def __init__(self, batch_norm = False, layer_norm = True, constraint = True, l1 = 512, l2=512, l3 = 512, l4=256, l5 = 64):
        super(DeepNet6LayerTune, self).__init__()
        # self.fc1 = nn.Linear(24, l1)
        self.fc1 = nn.Linear(171, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, l3)
        self.fc4 = nn.Linear(l3, l4)
        self.fc5 = nn.Linear(l4, l5)
        self.fc6 = nn.Linear(l5, 5)
        self.bn1 = nn.BatchNorm1d(l1)
        self.bn2 = nn.BatchNorm1d(l2)
        self.bn3 = nn.BatchNorm1d(l3)
        self.bn4 = nn.BatchNorm1d(l4)
        self.bn5 = nn.BatchNorm1d(l5)
        self.ln1 = nn.LayerNorm(l1)
        self.ln2 = nn.LayerNorm(l2)
        self.ln3 = nn.LayerNorm(l3)
        self.ln4 = nn.LayerNorm(l4)
        self.ln5 = nn.LayerNorm(l5)
        self.relu1 = nn.ReLU() 
        self.relu2 = nn.ReLU() 
        self.relu3 = nn.ReLU() 
        self.relu4 = nn.ReLU() 
        self.relu5 = nn.ReLU() 
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.constraint = constraint

    def forward(self, x): 
        if self.batch_norm and not self.layer_norm: 
            x = self.relu1(self.bn1(self.fc1(x)))
            x = self.relu2(self.bn2(self.fc2(x)))
            x = self.relu3(self.bn3(self.fc3(x)))
            x = self.relu4(self.bn4(self.fc4(x)))
            x = self.relu5(self.bn5(self.fc5(x)))
        elif self.layer_norm and not self.batch_norm: 
            x = self.relu1(self.ln1(self.fc1(x)))
            x = self.relu2(self.ln2(self.fc2(x)))
            x = self.relu3(self.ln3(self.fc3(x)))
            x = self.relu4(self.ln4(self.fc4(x)))
            x = self.relu5(self.ln5(self.fc5(x)))
        elif self.layer_norm and self.batch_norm:
            x = self.relu1(self.bn1(self.ln1(self.fc1(x))))
            x = self.relu2(self.bn2(self.ln2(self.fc2(x))))
            x = self.relu3(self.bn3(self.ln3(self.fc3(x))))
            x = self.relu4(self.bn4(self.ln4(self.fc4(x))))
            x = self.relu5(self.bn5(self.ln4(self.fc5(x))))
        else: 
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.relu3(self.fc3(x))
            x = self.relu4(self.fc4(x))
            x = self.relu5(self.fc5(x))

        x = self.fc6(x)

        if self.constraint: 
            # add contraint on prediction c1 -> cause issue for backprop
            # x[:, 0] = F.elu(x[:, 0], 0.25, False) in-place mod -> cause issue for backprop
            new_c1 = F.elu(x[:, 0], 0.25, False) # apply constraint on c1 
            new_c1 = new_c1.reshape(-1, 1)
            return torch.cat([new_c1, x[:, 1:]], 1)
        else:
            return x 

class DeepNet5LayerTune(nn.Module): 
    """
    define a  5 fully connected layer neural network for hparams tunning 4 hl + 1 output 
    """

    def __init__(self, batch_norm = False, layer_norm = True, constraint = True, l1 = 512, l2=512, l3=256, l4 = 64):
        super(DeepNet5LayerTune, self).__init__()
        # self.fc1 = nn.Linear(24, l1)
        self.fc1 = nn.Linear(171, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, l3)
        self.fc4 = nn.Linear(l3, l4)
        self.fc5 = nn.Linear(l4, 5)
        self.bn1 = nn.BatchNorm1d(l1)
        self.bn2 = nn.BatchNorm1d(l2)
        self.bn3 = nn.BatchNorm1d(l3)
        self.bn4 = nn.BatchNorm1d(l4)
        self.ln1 = nn.LayerNorm(l1)
        self.ln2 = nn.LayerNorm(l2)
        self.ln3 = nn.LayerNorm(l3)
        self.ln4 = nn.LayerNorm(l4)
        self.relu1 = nn.ReLU() 
        self.relu2 = nn.ReLU() 
        self.relu3 = nn.ReLU() 
        self.relu4 = nn.ReLU()
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.constraint = constraint

    def forward(self, x): 
        if self.batch_norm and not self.layer_norm: 
            x = self.relu1(self.bn1(self.fc1(x)))
            x = self.relu2(self.bn2(self.fc2(x)))
            x = self.relu3(self.bn3(self.fc3(x)))
            x = self.relu4(self.bn4(self.fc4(x)))
        elif self.layer_norm and not self.batch_norm: 
            x = self.relu1(self.ln1(self.fc1(x)))
            x = self.relu2(self.ln2(self.fc2(x)))
            x = self.relu3(self.ln3(self.fc3(x)))
            x = self.relu4(self.ln4(self.fc4(x)))
        elif self.layer_norm and self.batch_norm:
            x = self.relu1(self.bn1(self.ln1(self.fc1(x))))
            x = self.relu2(self.bn2(self.ln2(self.fc2(x))))
            x = self.relu3(self.bn3(self.ln3(self.fc3(x))))
            x = self.relu4(self.bn4(self.ln4(self.fc4(x))))
        else: 
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.relu3(self.fc3(x))
            x = self.relu4(self.fc4(x))

        x = self.fc5(x)

        if self.constraint: 
            # add contraint on prediction c1 -> cause issue for backprop
            # x[:, 0] = F.elu(x[:, 0], 0.25, False) in-place mod -> cause issue for backprop
            new_c1 = F.elu(x[:, 0], 0.25, False) # apply constraint on c1 
            new_c1 = new_c1.reshape(-1, 1)
            return torch.cat([new_c1, x[:, 1:]], 1)
        else:
            return x 

class DeepNet4LayerTune(nn.Module): 
    """
    define a  4 fully connected layer neural network for hparams tunning: 3hl + 1 output 
    """
    def __init__(self, batch_norm = True, layer_norm = True, constraint = True, l1 = 512, l2=256, l3=64):
        super(DeepNet4LayerTune, self).__init__()
        # self.fc1 = nn.Linear(24, l1)
        self.fc1 = nn.Linear(171, l1)
        # self.fc1 = nn.Linear(17, l1) # remove pH values 
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, l3)
        self.fc4 = nn.Linear(l3, 5)
        self.bn1 = nn.BatchNorm1d(l1)
        self.bn2 = nn.BatchNorm1d(l2)
        self.bn3 = nn.BatchNorm1d(l3)
        self.ln1 = nn.LayerNorm(l1)
        self.ln2 = nn.LayerNorm(l2)
        self.ln3 = nn.LayerNorm(l3)
        self.relu1 = nn.ReLU() 
        self.relu2 = nn.ReLU() 
        self.relu3 = nn.ReLU() 
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.constraint = constraint

    def forward(self, x): 
        if self.batch_norm and not self.layer_norm: 
            x = self.relu1(self.bn1(self.fc1(x)))
            x = self.relu2(self.bn2(self.fc2(x)))
            x = self.relu3(self.bn3(self.fc3(x)))
        elif self.layer_norm and not self.batch_norm: 
            x = self.relu1(self.ln1(self.fc1(x)))
            x = self.relu2(self.ln2(self.fc2(x)))
            x = self.relu3(self.ln3(self.fc3(x)))
        elif self.layer_norm and self.batch_norm:
            x = self.relu1(self.bn1(self.ln1(self.fc1(x))))
            x = self.relu2(self.bn2(self.ln2(self.fc2(x))))
            x = self.relu3(self.bn3(self.ln3(self.fc3(x))))
        else: 
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.relu3(self.fc3(x))

        x = self.fc4(x)

        if self.constraint:
            # add contraint on prediction c1 -> cause issue for backprop
            # x[:, 0] = F.elu(x[:, 0], 0.25, False) in-place mod -> cause issue for backprop
            new_c1 = F.elu(x[:, 0], 0.25, False) # apply constraint on c1 
            new_c1 = new_c1.reshape(-1, 1)
            return torch.cat([new_c1, x[:, 1:]], 1)
        else:
            return x 

class DeepNet3LayerTune(nn.Module): 
    """
    define a  tunable 3 fully connected layer neural network for hparams tunning 
    """

    def __init__(self, batch_norm = True, l1 = 1600, l2=400):
        super(DeepNet3LayerTune, self).__init__()
        self.fc1 = nn.Linear(24, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 5)
        self.bn1 = nn.BatchNorm1d(l1)
        self.bn2 = nn.BatchNorm1d(l2)
        self.relu1 = nn.ReLU() 
        self.relu2 = nn.ReLU() 
        self.batch_norm = batch_norm

    def forward(self, x): 
        if self.batch_norm: 
            x = self.relu1(self.bn1(self.fc1(x)))
            x = self.relu2(self.bn2(self.fc2(x)))
        else: 
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))

        x = self.fc3(x)
        return x 
    
if __name__ == '__main__':
    hparams = {
        "batch_size": 256,
        "input_channels": 24,
        "n_channels": 1024,
        'n_layers' : 2, 
        "layer_norm": True,
        "batch_norm": True,
        "hidden_activation": "Tanh",
        "dropout":None,

        # Training hyperparameters
        "lr": 0.001,
        "patience": 15,
        "factor": 0.3,
        "max_epochs": 3
    }
    
    sim_data = Variable(torch.rand(4,24))
#     print(sim_data[:, :hparams["input_channels"]])
#     net = DeepNet(hparams) 
#     print(net, net.configure_optimizers())
    net = DeepNet3Layer()
    out = net(sim_data)
    print('class', out.size())
    