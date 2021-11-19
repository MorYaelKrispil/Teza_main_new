import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR,CyclicLR,OneCycleLR,ReduceLROnPlateau


class DeepQNetwork(nn.Module): #4 layers 2 hidden- change to 3 layers 
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.hidden = nn.Linear(*input_dims, 100) #hidden relu
        nn.init.kaiming_normal_(self.hidden.weight,nonlinearity='relu')
        self.output = nn.Linear(100, n_actions)
        nn.init.uniform_(self.output.weight, a=-0.05, b=0.05)
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        #nn.init.xavier_normal_(self.fc1.weight)
        #self.fc2 = nn.Linear(100, 64)
        #self.out = nn.Linear(64, n_actions)
        #nn.init.xavier_normal_(self.out.weight)
        #nn.init.uniform_(self.out.weight, a=-0.05, b=0.05) #waz
        ##try change loss
        #self.loss = nn.MSELoss()
        
        self.loss = T.nn.SmoothL1Loss()
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min') 
        #new lr scadulr 
        #In `min` mode, lr will be reduced when the quantity monitored has stopped decreasing 
                                                                  
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        
        x = self.hidden(state)
        x = F.relu(x)
        action = self.output(x)
        return action

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)


    def load_checkpoint(self):
        print('... loading checkpoint ...')
        state_dict = T.load(self.checkpoint_file)
        print(state_dict.keys())
        self.load_state_dict(state_dict)
        self.eval() #to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will #yield inconsistent inference results.


