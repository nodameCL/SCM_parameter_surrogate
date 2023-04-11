import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
import json
import pandas as pd 

class SCMDataset(data.Dataset):
    def __init__(self,root, split='train', concat=False):
        self.root = root
        self.concat = concat 
        
        if self.concat: # need to concat global after convs 
            self.data_file = os.path.join(root, f'{split}.npz')
        else: 
            self.data_file = os.path.join(root,f'{split}_combined.npz')
            
        self.data = np.load(self.data_file, allow_pickle=True)
        
        self.x = torch.from_numpy(self.data['curve'].astype(np.float32))
        self.y = torch.from_numpy(self.data['y'].astype(np.float32))
        
        if self.concat: 
            self.sys = torch.from_numpy(self.data['sys'].astype(np.float32))
        else: 
            self.sys = torch.from_numpy(np.zeros((self.x.shape[0],3)).astype(np.float32))

    def __getitem__(self, index):
        if self.concat: 
            return self.x[index], self.y[index], self.sys[index]
        
        return self.x[index], self.y[index], self.sys[index] 

    def __len__(self):
        return len(self.x)

if __name__ == '__main__':
    datapath = '/pscratch/sd/c/chunhui/SCM/pointnet_56points/'

    d = SCMDataset(root = datapath, split='train_combined.npz', concat = False)
    print("length of dataset is:", len(d))
   
    ps, label = d[0] # get item when index = 0 
    print("points shape is: ", ps.shape, "len of points: ", len(ps))
    print("label is: ", label.shape, label.type(), label)
    # print("system scalar is: ", sys, sys.shape, sys.type())

    dataset = SCMDataset(
        root=datapath,
        data_augmentation=False)

    test_dataset = SCMDataset(
        root=datapath,
        split='val',
        data_augmentation=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0)

    testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=256,
            num_workers=0)

    print("Lengths are: ", len(dataloader.dataset), len(dataloader))

    for i, (points, target, sys_scalar) in enumerate(testdataloader): 
        if i < 1: 
            print(points.size(), points.size(0))
        else: 
            break 



