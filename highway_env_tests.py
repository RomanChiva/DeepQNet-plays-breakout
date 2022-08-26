import gym
import highway_env
from matplotlib import pyplot as plt
import torch
import numpy as np
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

model =  torch.load('TrainedModels/Traffic_first_try_98.pt')

obs = env.reset()

ret = 0
eplen = 0

while True:

    obs = torch.from_numpy(np.array(obs)).type(torch.float32).unsqueeze(0).to('cuda:0')
    q_vals = model.forward(obs)
    action = torch.argmax(q_vals).item()
    obs, reward, done, info = env.step(action)
    ret += reward
    eplen +=1
    #(done)
    env.render()

    if done:
        print(ret, eplen)
        ret=0
        eplen=0
        env.reset()
