from torch.utils.data import Dataset
import pandas as pd 
import numpy as np
import sys
import torch
import torch.utils.data
import os 
from ast import literal_eval

class SurfaceComplexationDataset(Dataset):
    """ surface complexation dataset """
    def __init__(self, 
                 root_dir, 
                 split = 'train'): 
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.root = root_dir 
        self.csv_file = os.path.join(root_dir, '{}.pkl'.format(split))
        self.data = pd.read_pickle(self.csv_file)

        # get inputs and targets 
        self.x = self.data.drop(['C1', 'C2', 'logK1', 'logK2', 'logKc', 'logKa', 'pH', 'sigma', 'zeta'], axis = 1)
        self.x = pd.concat([self.x, 
                       pd.DataFrame(self.data['pH'].to_list(), columns=['pH']*56), 
                       pd.DataFrame(self.data['sigma'].to_list(), columns=['sigma']*56),
                       pd.DataFrame(self.data['zeta'].to_list(), columns=['zeta']*56)], axis=1)
                       
        self.y = self.data[['C1', 'logK1', 'logK2', 'logKc', 'logKa']]

        self.x = torch.from_numpy(np.array(self.x).astype(np.float32))
        self.y = torch.from_numpy(np.array(self.y).astype(np.float32))

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, index): 
        return self.x[index], self.y[index]

class SurfaceComplexationDataset_7points(Dataset):
    """ surface complexation dataset with 7 points extracted from curves"""
    def __init__(self, 
                 root_dir, 
                 split = 'train'): 
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.root = root_dir 
        self.csv_file = os.path.join(root_dir, '{}.pkl'.format(split))
        self.data = pd.read_pickle(self.csv_file)
        
        # get inputs and targets
        self.x = self.data.drop(['C1', 'C2', 'logK1', 'logK2', 'logKc', 'logKa', 'pH', 'sigma', 'zeta'], axis = 1)
        self.x = pd.concat([self.x, 
                       pd.DataFrame(self.data['pH'].to_list(), columns=['pH']*7), 
                       pd.DataFrame(self.data['sigma'].to_list(), columns=['sigma']*7),
                       pd.DataFrame(self.data['zeta'].to_list(), columns=['zeta']*7)], axis=1)
                       
        self.y = self.data[['C1', 'logK1', 'logK2', 'logKc', 'logKa']]
        
        self.x = torch.from_numpy(np.array(self.x).astype(np.float32))
        self.y = torch.from_numpy(np.array(self.y).astype(np.float32))

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


if __name__ == '__main__':
    # d = SurfaceComplexationDataset(root_dir = '/home/chunhui/Documents/find_diff_coeffs/sigma_zeta_input', split='train')
    # d = SurfaceComplexationDataset_whole(root_dir = '/home/chunhui/Documents/find_diff_coeffs/train_val_test', split='test')
    # d = test_dataset(root_dir = '/home/chunhui/Documents/find_diff_coeffs/train_val_test', split='test')
    data_dir = '/home/chunhui/Documents/SCM/exp/test_exp_on_3NN/fitted_LiCL'
    # d = SurfaceComplexationDataset(root_dir = '/home/chunhui/Documents/SCM/7points/points_811', split='val')
    d = SurfaceComplexationDataset_csv(root_dir = data_dir, split='test')
    print("length of dataset is:", len(d), type(d))
   
    x, y = d[0] # get item when index = 0 
    print("shape of input: ", x.shape, type(x), x)
    print("shape of output: ", y.shape, type(y),  y)

    # data_dir = '/home/chunhui/Documents/SCM/7points/points_811'
    data_set = SurfaceComplexationDataset_csv(root_dir=data_dir, split='val')
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=2,
        shuffle=True,
        num_workers=0)
    
    print(len(data_loader), len(data_loader.dataset))
    for i, (inputs, targets) in enumerate(data_loader): 
        if i < 1: 
            print(inputs.shape, targets.shape, inputs[:, :3])
        else: 
            break
