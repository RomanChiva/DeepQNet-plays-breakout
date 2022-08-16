import scipy as sp
import numpy as np
import gym
import matplotlib.pyplot as plt
from skimage import color, transform

# Make the obervations from the environment more compact
def feat_extract(obs):
    #Crop area of interest
    obs = obs[52:196,9:152,:]
    # Convert to grayscale
    obs = color.rgb2gray(obs)
    # Rescale
    obs = transform.rescale(obs,0.6,anti_aliasing=True)
    obs[obs[:,:] != 0] = 255
    
    # Return, rescaled cropped grayscale observation
    return obs
# Change

# Check what ur function is doing. If you mand to mod for other games or smth
if __name__ =='__main__':
    env = gym.make('Breakout-v4')
    obs = env.reset()
    obs = feat_extract(obs)
    obs1,a,b,_ = env.step(1)
    obs2,a,b,_ = env.step(2)
    obs2 = feat_extract(obs2)
    print(obs.shape)
    plt.imshow(obs2, cmap='Greys')
    
    plt.show()
