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

# Stable Baselines Environment
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Construct Environment
env = make_atari_env('BreakoutNoFrameskip-v4')
env = VecFrameStack(env, n_stack=4)



# Decalre Hyperparameters
epochs = 100
updates_per_epoch = 5000
runs_per_hp_set = 1 # To compute statistics
buffer_size = 100000
epsilon_initial = 1
epsilon_final = 0.02
discount = 0.99
update_target = 10000
update_gradient = 20
lr = 0.00025
batch_size = 32
greedy_steps = 3000000




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

name_of_run = 'Hparams_3Mil_greedy_lr_0_00025'
trainer.populate()
model, loss, ret, q_val, ep_len = trainer.train(name_of_run)



# Create Data logs, Plots and store Trining data
recap(loss,ret,q_val,ep_len,name_of_run)
    
    




