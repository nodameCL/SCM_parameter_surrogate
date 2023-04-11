from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
# from model.point_dataset import SCMDataset

class PointNetOneX(nn.Module):
    def __init__(self, l1 = 64, l2 = 128, l3=512, l4 = 256, l5 = 64, 
                 dropout = 0, isBN = False, isLN = False, constraint = False, pooling = 'max'):
        super(PointNetOneX, self).__init__()
        
        self.conv1 = torch.nn.Conv1d(6, l1, 1)
        self.conv2 = torch.nn.Conv1d(l1, l2, 1)
        self.conv3 = torch.nn.Conv1d(l2, l3, 1)
        self.bn1 = nn.BatchNorm1d(l1)
        self.bn2 = nn.BatchNorm1d(l2)
        self.bn3 = nn.BatchNorm1d(l3)
        self.ln1 = nn.LayerNorm([l1,56])
        self.ln2 = nn.LayerNorm([l2,56])
        self.ln3 = nn.LayerNorm([l3,56])
        self.l3 = l3
        
        self.fc1 = nn.Linear(l3, l4)
        self.fc2 = nn.Linear(l4, l5)
        self.fc3 = nn.Linear(l5, 5)
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(l4) 
        self.bn2 = nn.BatchNorm1d(l5)
        self.ln1 = nn.LayerNorm(l4)
        self.ln2 = nn.LayerNorm(l5)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        
        self.isBN = isBN
        self.isLN = isLN
        self.constraint = constraint
        self.pooling = pooling 

    def forward(self, x):
        # step 1 learn features from point cloud 
        if self.isBN and not self.isLN: 
            x = self.relu1(self.bn1(self.conv1(x))) # x: [BZ, l1, num_points]
            x = self.relu2(self.bn2(self.conv2(x))) # x: [BZ, l2, num_points]
            x = self.bn3(self.conv3(x))         # x: [BZ, l3, num_points]
        elif self.isLN and not self.isBN: 
            x = self.relu1(self.ln1(self.conv1(x)))
            x = self.relu2(self.ln2(self.conv2(x)))
            x = self.ln3(self.conv3(x))
        else: 
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = self.conv3(x)

        # step 2 maxpooling the learned features 
        if self.pooling == 'max': 
            x = torch.max(x, 2, keepdim=True)[0] # x: [BZ, l3, 1]
        else: 
            x = torch.mean(x, 2, keepdim=True)
            
        # step 2.2 reshape the maxpooled features for step 3
        x = x.view(-1, self.l3) # x: [BZ, l3]

        # step 3 feed extracted features to FC layers 
        if self.isBN and not self.isLN: 
            x = self.relu3(self.bn1(self.fc1(x)))
            x = self.relu4(self.bn2(self.dropout(self.fc2(x))))
        elif self.isLN and not self.isBN: 
            x = self.relu3(self.ln1(self.fc1(x)))
            x = self.relu4(self.ln2(self.dropout(self.fc2(x))))
        else: 
            x = self.relu3(self.fc1(x))
            x = self.relu4(self.dropout(self.fc2(x)))

        x = self.fc3(x)

        if self.constraint: 
            # add contraint on prediction c1 -> cause issue for backprop
            # x[:, 0] = F.elu(x[:, 0], 0.25, False) in-place mod -> cause issue for backprop
            new_c1 = F.elu(x[:, 0], 0.25, False) # apply constraint on c1 
            new_c1 = new_c1.reshape(-1, 1)
            return torch.cat([new_c1, x[:, 1:]], 1)
        else:
            return x 

class PointNetOneXY(nn.Module):
    def __init__(self, l1 = 64, l2 = 128, l3=512, l4 = 256, l5 = 64, 
                 dropout = 0, isBN = False, isLN = False, constraint = False, pooling = 'max'):
        super(PointNetOneXY, self).__init__()
        
        self.conv1 = torch.nn.Conv1d(3, l1, 1)
        self.ln1 = nn.LayerNorm([l1,56])
        self.conv2 = torch.nn.Conv1d(l1, l2, 1)
        self.conv3 = torch.nn.Conv1d(l2, l3, 1)
        self.bn1 = nn.BatchNorm1d(l1)
        self.bn2 = nn.BatchNorm1d(l2)
        self.bn3 = nn.BatchNorm1d(l3)
        self.ln2 = nn.LayerNorm([l2,56])
        self.ln3 = nn.LayerNorm([l3,56])
        self.l3 = l3
        
        self.fc1 = nn.Linear(l3 + 3, l4) # concat with PT 
        self.fc2 = nn.Linear(l4, l5)
        self.fc3 = nn.Linear(l5, 5)
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(l4) 
        self.bn2 = nn.BatchNorm1d(l5)
        self.ln1 = nn.LayerNorm(l4)
        self.ln2 = nn.LayerNorm(l5)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        
        self.isBN = isBN
        self.isLN = isLN
        self.constraint = constraint
        self.pooling = pooling 

    def forward(self, x):
        # step 1 learn features from point cloud 
        if self.isBN and not self.isLN: 
            x = self.relu1(self.bn1(self.conv1(x))) # x: [BZ, l1, num_points]
            x = self.relu2(self.bn2(self.conv2(x))) # x: [BZ, l2, num_points]
            x = self.bn3(self.conv3(x))         # x: [BZ, l3, num_points]
        elif self.isLN and not self.isBN: 
            x = self.relu1(self.ln1(self.conv1(x)))
            x = self.relu2(self.ln2(self.conv2(x)))
            x = self.ln3(self.conv3(x))
        else: 
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = self.conv3(x)

        # step 2 maxpooling the learned features 
        if self.pooling == 'max': 
            x = torch.max(x, 2, keepdim=True)[0] # x: [BZ, l3, 1]
        else: 
            x = torch.mean(x, 2, keepdim=True)
            
        # step 2.1: reshape the maxpooled features for step 3
        x = x.view(-1, self.l3) # x: [BZ, l3]
        
        ## step 3: concate system condition Ns, PZC, C
        x = torch.cat([x, y], 1) # y (BZ, 3)

        # step 4: feed extracted features to FC layers 
        if self.isBN and not self.isLN: 
            x = self.relu3(self.bn1(self.fc1(x)))
            x = self.relu4(self.bn2(self.dropout(self.fc2(x))))
        elif self.isLN and not self.isBN: 
            x = self.relu3(self.ln1(self.fc1(x)))
            x = self.relu4(self.ln2(self.dropout(self.fc2(x))))
        else: 
            x = self.relu3(self.fc1(x))
            x = self.relu4(self.dropout(self.fc2(x)))

        x = self.fc3(x)

        if self.constraint: 
            # add contraint on prediction c1 -> cause issue for backprop
            # x[:, 0] = F.elu(x[:, 0], 0.25, False) in-place mod -> cause issue for backprop
            new_c1 = F.elu(x[:, 0], 0.25, False) # apply constraint on c1 
            new_c1 = new_c1.reshape(-1, 1)
            return torch.cat([new_c1, x[:, 1:]], 1)
        else:
            return x 

class Pointfeat(nn.Module):
    def __init__(self, l1 = 64, l2 = 128, l3 = 512, 
                 isBN = False, isLN = False, pooling = 'max'):
        super(Pointfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(6, l1, 1)
        self.ln1 = nn.LayerNorm([l1,56])
        self.conv2 = torch.nn.Conv1d(l1, l2, 1)
        self.conv3 = torch.nn.Conv1d(l2, l3, 1)
        self.bn1 = nn.BatchNorm1d(l1)
        self.bn2 = nn.BatchNorm1d(l2)
        self.bn3 = nn.BatchNorm1d(l3)
        self.ln2 = nn.LayerNorm([l2,56])
        self.ln3 = nn.LayerNorm([l3,56])
        self.l3 = l3
        self.isBN = isBN
        self.isLN = isLN 
        self.pooling = pooling 

    def forward(self, x): # x: [BZ, input_dim, num_points]
        if self.isBN and not self.isLN: 
            x = F.relu(self.bn1(self.conv1(x))) # x: [BZ, l1, num_points]
            x = F.relu(self.bn2(self.conv2(x))) # x: [BZ, l2, num_points]
            x = self.bn3(self.conv3(x))         # x: [BZ, l3, num_points]
        elif self.isLN and not self.isBN: 
            x = F.relu(self.ln1(self.conv1(x)))
            x = F.relu(self.ln2(self.conv2(x)))
            x = self.ln3(self.conv3(x))
        else: 
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.conv3(x)

        if self.pooling == 'max': 
            x = torch.max(x, 2, keepdim=True)[0] # x: [BZ, l3, 1]
        else: 
            x = torch.mean(x, 2, keepdim=True)
            
        x = x.view(-1, self.l3) # x: [BZ, l3]

        return x 

class PointfeatTune(nn.Module):
    def __init__(self, l1 = 64, l2 = 128, l3 = 512, 
                 isBN = False, isLN = False, addGlobal = False, pooling = 'max'):
        super(PointfeatTune, self).__init__()
        if addGlobal: 
            self.conv1 = torch.nn.Conv1d(3, l1, 1)
            self.ln1 = nn.LayerNorm([l1, 56])
        else: 
            self.conv1 = torch.nn.Conv1d(6, l1, 1)
            self.ln1 = nn.LayerNorm([l1,56])
        self.conv2 = torch.nn.Conv1d(l1, l2, 1)
        self.conv3 = torch.nn.Conv1d(l2, l3, 1)
        self.bn1 = nn.BatchNorm1d(l1)
        self.bn2 = nn.BatchNorm1d(l2)
        self.bn3 = nn.BatchNorm1d(l3)
        self.ln2 = nn.LayerNorm([l2,56])
        self.ln3 = nn.LayerNorm([l3,56])
        self.l3 = l3
        self.isBN = isBN
        self.isLN = isLN 
        self.addGlobal = addGlobal
        self.pooling = pooling 

    def forward(self, x, y): # x: [BZ, input_dim, num_points]
        if self.isBN and not self.isLN: 
            x = F.relu(self.bn1(self.conv1(x))) # x: [BZ, l1, num_points]
            x = F.relu(self.bn2(self.conv2(x))) # x: [BZ, l2, num_points]
            x = self.bn3(self.conv3(x))         # x: [BZ, l3, num_points]
        elif self.isLN and not self.isBN: 
            x = F.relu(self.ln1(self.conv1(x)))
            x = F.relu(self.ln2(self.conv2(x)))
            x = self.ln3(self.conv3(x))
        else: 
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.conv3(x)

        if self.pooling == 'max': 
            x = torch.max(x, 2, keepdim=True)[0] # x: [BZ, l3, 1]
        else: 
            x = torch.mean(x, 2, keepdim=True)
        # x = nn.AvgPool1d(2, stride=2)(x) # avgPooling [BZ, 1024, 35]
        # x = nn.MaxPool1d(2, stride=2)(x)
        # x = torch.flatten(x, start_dim = 1) # (BZ, inputDim)  flatten
        x = x.view(-1, self.l3) # x: [BZ, l3]

        ## concate system condition Ns, PZC, C
        if self.addGlobal: 
            sys_feat = y # (m, 3)
            x = torch.cat([x, sys_feat], 1)

        return x 

class PointNetTune(nn.Module):
    def __init__(self, l1 = 64, l2 = 128, l3=512, l4 = 256, l5 = 64, 
                 dropout = 0, isBN = False, isLN = False, 
                 addGlobal = False, constraint = False, pooling = 'max'):
        super(PointNetTune, self).__init__()
        self.feat = PointfeatTune(l1, l2, l3, isBN, isLN, addGlobal, pooling)
        if addGlobal: 
            self.fc1 = nn.Linear(3+l3, l4)
        else: 
            self.fc1 = nn.Linear(l3, l4)
        self.fc2 = nn.Linear(l4, l5)
        self.fc3 = nn.Linear(l5, 5)
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(l4) 
        self.bn2 = nn.BatchNorm1d(l5)
        self.ln1 = nn.LayerNorm(l4)
        self.ln2 = nn.LayerNorm(l5)
        self.relu = nn.ReLU()
        self.isBN = isBN
        self.isLN = isLN
        self.addGlobal = addGlobal
        self.constraint = constraint

    def forward(self, x, y):
        x  = self.feat(x, y)

        if self.isBN and not self.isLN: 
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        elif self.isLN and not self.isBN: 
            x = F.relu(self.ln1(self.fc1(x)))
            x = F.relu(self.ln2(self.dropout(self.fc2(x))))
        else: 
            x = F.relu(self.fc1(x))
            x = F.relu(self.dropout(self.fc2(x)))

        x = self.fc3(x)

        if self.constraint: 
            # add contraint on prediction c1 -> cause issue for backprop
            # x[:, 0] = F.elu(x[:, 0], 0.25, False) in-place mod -> cause issue for backprop
            new_c1 = F.elu(x[:, 0], 0.25, False) # apply constraint on c1 
            new_c1 = new_c1.reshape(-1, 1)
            return torch.cat([new_c1, x[:, 1:]], 1)
        else:
            return x 

class PointNetTuneDense(nn.Module):
    def __init__(self, l1 = 64, l2 = 128, l3=512, l4 = 256, l5 = 128, l6 = 64, 
                 dropout = 0, isBN = False, isLN = False, 
                 addGlobal = False, constraint = False, pooling = 'max'):
        super(PointNetTuneDense, self).__init__()
        self.feat = PointfeatTune(l1, l2, l3, isBN, isLN, addGlobal, pooling)
        if addGlobal: 
            self.fc1 = nn.Linear(3+l3, l4)
        else: 
            self.fc1 = nn.Linear(l3, l4)
        self.fc2 = nn.Linear(l4, l5)
        self.fc3 = nn.Linear(l5, l6)
        self.fc4 = nn.Linear(l6, 5)
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(l4) 
        self.bn2 = nn.BatchNorm1d(l5)
        self.bn3 = nn.BatchNorm1d(l6)
        self.ln1 = nn.LayerNorm(l4)
        self.ln2 = nn.LayerNorm(l5)
        self.ln3 = nn.LayerNorm(l6)
        self.relu = nn.ReLU()
        self.isBN = isBN
        self.isLN = isLN
        self.addGlobal = addGlobal
        self.constraint = constraint

    def forward(self, x, y):
        x  = self.feat(x, y)

        if self.isBN and not self.isLN: 
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        elif self.isLN and not self.isBN: 
            x = F.relu(self.ln1(self.fc1(x)))
            x = F.relu(self.ln2(self.fc2(x)))
            x = F.relu(self.ln3(self.dropout(self.fc3(x))))
        else: 
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.dropout(self.fc3(x)))

        x = self.fc4(x)

        if self.constraint: 
            # add contraint on prediction c1 -> cause issue for backprop
            # x[:, 0] = F.elu(x[:, 0], 0.25, False) in-place mod -> cause issue for backprop
            new_c1 = F.elu(x[:, 0], 0.25, False) # apply constraint on c1 
            new_c1 = new_c1.reshape(-1, 1)
            return torch.cat([new_c1, x[:, 1:]], 1)
        else:
            return x 

if __name__ == '__main__':

    sim_data = Variable(torch.rand(128,3,2500)) # BZ = 40, 
    sim_scalar = Variable(torch.rand(128,3))

    datapath = '/home/chunhui/Documents/SCM/points'

#     test_dataset = SCMDataset(
#         root=datapath,
#         split='val',
#         data_augmentation=False)

#     testdataloader = torch.utils.data.DataLoader(
#             test_dataset,
#             batch_size=256,
#             num_workers=0)

    cls = PointNettune()
    out = cls(sim_data, sim_scalar)
    # print('class', out.size(), out)
    # print(cls)