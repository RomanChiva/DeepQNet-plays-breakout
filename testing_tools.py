from turtle import screensize
import unittest
from trainer import Trainer
import gym
from DQN import DeepQNet
from replay_buffer import ReplayBuffer
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import pickle
# The purpose of this script is to test that the code of the ther 
# scripts was written correctly. 

# Assertions are quite hard to write as a lot of the functions 
# implemented in the code do not have straightforward outputs 
# but rather perform actions on the model.



from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env = make_atari_env('BreakoutNoFrameskip-v4')
env = VecFrameStack(env, n_stack=4)



#model = DeepQNet(86,4,27482)
# Use smaler buffer, for testing purposes being sped up
#buffer = ReplayBuffer(100)

# Trainer tool
#trainer = Trainer(env, 1e-4, 10, buffer,4,10,1500,1,0.2,1e-4)
#trainer.populate()



#  ==== TEST STEP FUNCTION ================
def initialize_step():
    obs = trainer.step(0, init = True)
    obs = obs.numpy()
    print(type(obs))
    print(np.shape(obs))
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(obs[0,0,:,:])
    axs[0, 0].set_title('step1')
    axs[0, 1].imshow(obs[0,1,:,:])
    axs[0, 1].set_title('step2')
    axs[1, 0].imshow(obs[0,2,:,:])
    axs[1, 0].set_title('step3')
    axs[1, 1].imshow(obs[0,3,:,:])
    axs[1, 1].set_title('step4')

    plt.show()

def normal_step():
    obs0 = trainer.step(0, init = True)
    obs,rew,done = trainer.step(2)
    

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(obs[0,0,:,:])
    axs[0, 0].set_title('step1')
    axs[0, 1].imshow(obs[0,1,:,:])
    axs[0, 1].set_title('step2')
    axs[1, 0].imshow(obs[0,2,:,:])
    axs[1, 0].set_title('step3')
    axs[1, 1].imshow(obs[0,3,:,:])
    axs[1, 1].set_title('step4')

    plt.show()

def step_sequence():

    # I set the initial skip size to 4, however through this I saw it was too much 
    # REDUCE SKIP SIZE!!!

    obs = trainer.step(0, init = True)

    while True:

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(obs[0,0,:,:])
        axs[0, 0].set_title('step1')
        axs[0, 1].imshow(obs[0,1,:,:])
        axs[0, 1].set_title('step2')
        axs[1, 0].imshow(obs[0,2,:,:])
        axs[1, 0].set_title('step3')
        axs[1, 1].imshow(obs[0,3,:,:])
        axs[1, 1].set_title('step4')

        plt.show()
        
        action = input('Enter next action:')

        obs,rew,done = trainer.step(int(action))

def steps_in_random_run():

    steps = []

    for x in range(20):
        counter = 1
        obs = trainer.step(0,init=True)
        
        while True:

            obs,rew,done = trainer.step(trainer.env.action_space.sample())
            counter +=1

            if done:
                steps.append(counter)
                break
    
    mean = np.mean(np.array(steps))
    print('Mean number of steps:',mean)
    print('Mean N of frames:',mean*trainer.skip)




def recap(losses, returns, q_vals, ep_len, tag, reps):
    
    fig, axs = plt.subplots(2,2)

    for x in range(reps):
        axs[0,0].plot(range(len(losses[x])),losses[x])
    axs[0,0].set_title('Loss')

    for x in range(reps):
        axs[0,1].plot(range(len(returns[x])),returns[x])
    axs[0,1].set_title('Return')

    for x in range(reps):
        axs[1,0].plot(range(len(q_vals[x])),q_vals[x])
    axs[1,0].set_title('Average Max-Q-Value')
    axs[1,0].set_xlabel('Epochs')

    for x in range(reps):
        axs[1,1].plot(range(len(ep_len[x])),ep_len[x])
    axs[1,1].set_title('Average Episode Length')
    axs[1,1].set_xlabel('Epochs')
    
    plt.tight_layout()
    plt.savefig('recaps/plots/{a}.png'.format(a=tag))

    # Save the numerical data

    array = np.array([losses, returns, q_vals, ep_len])

    filename = tag + '.pkl'

    with open('recaps/numerical_data/{a}'.format(a=filename), 'wb') as f:
        pickle.dump(array,f)


def process_obs(obs,model):

    obs = torch.from_numpy(np.array(obs)).to('cuda:0')
    obs = obs.permute(0,3,1,2)
    q = model(obs.type(torch.float32))
    act = torch.tensor([torch.argmax(q)])
    print(act)
    return act

def watch_model(input_model):

    
    obs = env.reset()
    
    while True:
        env.render()
        act = process_obs(obs, input_model)
        obs,rew,done,_ = env.step(act)
        #time.sleep(0.1)
        if done:
            env.reset()
        

    
if __name__ == '__main__':

    
    model = torch.load('TrainedModels/Hparams_from_KERAS_REDUCE_LR_more_Exploration_0.pt')
    untrained_net = DeepQNet(4).to('cuda:0')
    watch_model(model)




