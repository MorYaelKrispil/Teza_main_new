
import sys
import os.path
import numpy as np
import torch as T
from DQN_huber_loss import DeepQNetwork
from Replay_memory import ReplayBuffer
import random


class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn', action_space={}):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = action_space
        self.learn_step_counter = 0
        # 16.5 for epsilon changes
        self.steps_done = 0
        self.train_losses = []
        # # to track the validation loss as the model trains
        # self.valid_losses = []
        # to track the average training loss per epoch as the model trains
        self.avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        #self.avg_valid_losses = [] 

        # 16.5 if i chang epsilon eps_start will also changed? if yes change it
        self.eps_start = epsilon

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name + '_' + self.algo + '_q_eval',
                                   chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name + '_' + self.algo + '_q_next',
                                   chkpt_dir=self.chkpt_dir)
        self.q_next.eval()  #new
        self.q_eval.train() #new

    def choose_action(self, observation, mood='Train'):

        self.steps_done += 1
        if mood == 'Train':

            if np.random.random() > self.epsilon:
                state = T.tensor(list(observation.values()), dtype=T.float).to(self.q_eval.device)
                actions = self.q_eval.forward(state)
                # action = T.argmax(actions).item()
                #return Qval array
                return actions, 'not random choice'
            # return action
            else:
                action = random.choice(list(self.action_space.keys()))
                return action, 'random choice'
        if mood == 'Test':
            state = T.tensor(list(observation.values()), dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            return actions, 'not random choice'

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        
        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
            self.q_next.eval()
            

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    # trying to change epsilon diffrently 16.5
    # self.epsilon = self.eps_min + (self.eps_start - self.eps_min) * \
    #     math.exp(-1. * self.steps_done / self.eps_dec)\
    #                         if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
         
        if self.memory.mem_cntr < self.batch_size:
            return
        
        self.q_eval.train()
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]


        # chaneged16.5 if done i dont want to change it tp 0 in this case there is no meaning for done
        # q_next[dones] = 0.0
        q_target = rewards + (self.gamma * q_next)

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        for param in self.q_eval.parameters():  #new clipping
            param.grad.data.clamp_(-1, 1)
            
        
        self.q_eval.optimizer.step()
        self.train_losses.append(loss.item())
        #new lr scadualr
        #self.q_eval.scheduler.step(loss)
        #validation 
        
        
        # q_pred_validation = self.q_eval.forward(validation_states)[indices, validation_actions]
        # q_next_validation = self.q_next.forward(validation_states_).max(dim=1)[0]

        # # chaneged16.5 if done i dont want to change it tp 0 in this case there is no meaning for done
        # # q_next[dones] = 0.0
        # q_target_validation = validation_rewards + self.gamma * q_next_validation

        # loss_validation = self.q_eval.loss(q_target_validation, q_pred_validation)
        # self.valid_losses.append(loss_validation.item())

        
        self.learn_step_counter += 1
        self.decrement_epsilon()
        #new lr scadualr
        # Note that step should be called after validate with validation loss()
        #In `min` mode, lr will be reduced when the quantity monitored has stopped decreasing- i defined with NN class
        
        self.q_eval.scheduler.step(loss)
        

