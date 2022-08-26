from ctypes.wintypes import tagRECT
from psutil import disk_io_counters
from trainer import Trainer
import gym
import torch
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from testing_tools import recap
import numpy as np
import pickle
import highway_env

# =======================
#     HIGHWAY DRIVING 
# =======================

# Construct Environment

env = gym.make('highway-v0')

# CHoose image shape, and set stark contrast betoween green and blu cars
config = {
       "observation": {
           "type": "GrayscaleObservation",
           "observation_shape": (100, 50),
           "stack_size": 4,
           "weights": [0.29, 0.3870, 0.71],  # weights for RGB conversion
           "scaling": 1.5,
       },
       "collision_reward": -1,
       "duration": 1000,
       "simulation_frequency": 15,
       "policy_frequency": 5
   }
env.configure(config)

# CHoose Discrete Meta Actions

env.configure({
    "action": {
        "type": "DiscreteMetaAction"
    }
})







# Decalre Hyperparameters
epochs = 100
updates_per_epoch = 3000
buffer_size = 15000
epsilon_initial = 1
epsilon_final = 0.02
discount = 0.99
update_target = 50
update_gradient = 1
lr = 3e-4
batch_size = 32
greedy_steps = 10000




# Use GPU for tensors
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  



# PREP
print('==============================')
print('------STARTING TRAINING-------')
print('==============================')
    
# Replay buffer object to store observations
buffer = ReplayBuffer(buffer_size)

# Trainer tool
trainer = Trainer(  device = dev,
                    env=env,
                    buffer = buffer,
                    discount = discount,#int(skip[x]),
                    update_target= update_target,
                    gradient_update=update_gradient,
                    epochs = epochs,
                    updates_per_epoch = updates_per_epoch,
                    lr = lr,#lr[x],
                    batch_size = batch_size,#int(bs[x]),
                    eps_initial=epsilon_initial,
                    eps_final = epsilon_final,
                    eps_step=greedy_steps)#gps[x]   )

name_of_run = 'Traffic_first_try'
trainer.populate()
model, loss, ret, q_val, ep_len = trainer.train(name_of_run)



# Create Data logs, Plots and store Trining data
recap(loss,ret,q_val,ep_len,name_of_run)
    
    




