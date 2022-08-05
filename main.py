from trainer import Trainer
import gym
import torch
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from testing_tools import recap
import numpy as np
import pickle



#Define Environment
env = gym.make('Breakout-v4')


# Non-variable H-Params
epochs = 50
updates_per_epoch = 5e4
runs_per_hp_set = 1 # To compute statistics
buffer_size = 1e5
epsilon_initial = 1
epsilon_final = 0.1


# Define search space

#Skips
skip_min = 1
skip_max = 4
# Learning rates
lr_min = 1e-4
lr_max = 1e-2
# Batch_size
bs_min = 50
bs_max = 300
# Greedy Policy STeps
gps_min = 1e-5
gps_max = 5e-5


# ============================================
#             RANDOM SEARCH
# ============================================

# How many HParam iterations should we evaluate
iterations = 1

# Build uniform distributions
skip  = np.random.uniform(low=skip_min, high=skip_max, size = iterations)
skip = np.round(skip, decimals=0)
lr  = np.random.uniform(low=lr_min, high=lr_max, size = iterations)
lr = np.round(lr, decimals=5)
bs  = np.random.uniform(low=bs_min, high=bs_max, size = iterations)
bs = np.round(bs, decimals=0)
gps  = np.random.uniform(low=gps_min, high=gps_max, size = iterations)
gps = np.round(gps, decimals=5)

# Save Params
#params = np.array([skip,lr,bs,gps])
#with open('Hyperparameters/RS{n}_params.pkl'.format(n=iterations), 'wb') as f:
 #   pickle.dump(params,f)


# Use GPU for tensors
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  



for x in range (iterations):

    # PREP
    print('==============================')
    print('-----STARTING ITERATION-{a}---'.format(a=x+1))
    print('==============================')

    name_of_run = 'RS{n}_sk{a}_lr{b}_bs{c}_gps{d}'.format(n = iterations,a = skip[x], b = lr[x],c=bs[x], d = gps[x])
    print(name_of_run)
    

    losses = []
    returns = []
    q_vals = []
    ep_lens = []

    
    for x in range(runs_per_hp_set):

        # Replay buffer object to store observations
        buffer = ReplayBuffer(buffer_size)

        # Trainer tool
        trainer = Trainer(  device = dev,
                            env=env,
                            buffer = buffer,
                            skip = 4,#int(skip[x]),
                            epochs = epochs,
                            updates_per_epoch = updates_per_epoch,
                            lr = 1e-4,#lr[x],
                            batch_size = 32,#int(bs[x]),
                            eps_initial=epsilon_initial,
                            eps_final = epsilon_final,
                            eps_step=0.9e-6)#gps[x]   )

        name_of_run = 'Hparam_paper'
        trainer.populate()
        model, loss, ret, q_val, ep_len = trainer.train(name_of_run)
        path = 'TrainedModels/' + name_of_run + '_{}'.format(x+1) +'.pt'
        torch.save(model, path)

        losses.append(loss)
        returns.append(ret)
        q_vals.append(q_val)
        ep_lens.append(ep_len)

    # Save data about the iteration
    recap(losses,returns,q_vals,ep_lens,name_of_run, runs_per_hp_set)
    
    




