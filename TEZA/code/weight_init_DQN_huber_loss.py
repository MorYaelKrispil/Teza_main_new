import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR,CyclicLR,OneCycleLR,ReduceLROnPlateau


class DeepQNetwork(nn.Module): #3 layers  # input-input*2-actions
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.input_dims=int(input_dims[0])
        
        self.Number_hidden_nurones=int(n_actions*2) #?? input_dims[0]*2,n_actions*2
        self.n_actions=int(n_actions)
        print('NEW-2!!!!!!!!!!!!!!!!!!')
#init network&weights&bias
        
        
        self.hid1 = nn.Linear(self.input_dims,self.Number_hidden_nurones ) 
        self.oupt = nn.Linear(self.Number_hidden_nurones, self.n_actions)
        nn.init.xavier_uniform_(self.hid1.weight)           #init all weights with xavier_normal_ \xavier_uniform_
                                                              #link: https://pytorch.org/docs/stable/nn.init.html
        nn.init.zeros_(self.hid1.bias)                      #init bias ro 0
        nn.init.xavier_uniform_(self.oupt.weight)
        nn.init.zeros_(self.oupt.bias)


#init loss & optimaizer
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = T.nn.SmoothL1Loss()
        
        self.to(self.device)
        
        
    def forward(self, x):
        z = T.tanh(self.hid1(x))
        z = self.oupt(z)
        return z

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)


    def load_checkpoint(self):
        print('... loading checkpoint ...')
        state_dict = T.load(self.checkpoint_file)
        print(state_dict.keys())
        self.load_state_dict(state_dict)
        self.eval() 
        #to set dropout and batch normalization layers to evaluation mode before running inference.
        #Failing to do this will yield inconsistent inference results.


