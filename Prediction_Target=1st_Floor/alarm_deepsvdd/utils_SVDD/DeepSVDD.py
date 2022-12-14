# -*- coding: utf-8 -*-
"""
@author: Ruff L, et al. Deep one-class classification. 35th Int Conf Mach Learn ICML 2018 2018;10:6981–96; 
@Modified by Sheng Shi
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

class DeepSVDD():
    """A class for the Deep SVDD method.

    Attributes:
        objective: A string specifying the Deep SVDD objective (either 'one-class' or 'soft-boundary').
        nu: Deep SVDD hyperparameter nu (must be 0 < nu <= 1).
        R: Hypersphere radius R.
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network \phi.
        trainer: DeepSVDDTrainer to train a Deep SVDD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SVDD network.
        results: A dictionary to save the results.
    """

    def __init__(self, nu: float = 0.001):
        """Inits DeepSVDD with one of the two objectives and hyperparameter nu."""
        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.nu = nu
        self.R = 0.0   # hypersphere radius R
        self.c = None  # hypersphere center c
        self.warm_up_n_epochs = 10
        self.net = None  # neural network \phi
        self.trainer = None
        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }


    def build_network(self, inputdim, hidden1, outputdim):
        'Builds the neural network'
        class MyNet(nn.Module):
            def __init__(self,inputdim,hidden1,outputdim):
                super(MyNet,self).__init__()
                self.rep_dim = outputdim # dimension of feature space
                self.fc1 = nn.Linear(inputdim,hidden1, bias=False)
                self.fc2 = nn.Linear(hidden1, self.rep_dim, bias=False)
            def forward(self, x):
                x = F.leaky_relu(self.fc1(x))
                x = self.fc2(x)
                return x.squeeze()
        self.net = MyNet(inputdim,hidden1,outputdim).double()
        

    

    def test(self, test_x, batch_size: int = 64):
        """Tests the Deep SVDD model on the test data."""
        if self.net is None:
            print('Error: please train the network first')
        net = self.net
        test_loader = Data.DataLoader(dataset=myDataLoader(test_x), batch_size=batch_size, shuffle=False)
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs = data
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                scores = dist - self.R ** 2
                # Save triples of (idx, label, score) in a list
                idx_label_score += scores.cpu().data.numpy().tolist()
        self.test_scores = idx_label_score


    def load_model(self, model_path):
        """Load Deep SVDD model from model_path."""
        model_dict = torch.load(model_path)
        self.R = model_dict['R']
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])


class myDataLoader(Dataset):
  def __init__(self,mydata):
    super(myDataLoader,self).__init__()
    self.data = mydata
  def __getitem__(self,idx):
    data = self.data[:,np.newaxis]
    opt = data[idx,:]
    return opt
  def __len__(self):
    return len(self.data)