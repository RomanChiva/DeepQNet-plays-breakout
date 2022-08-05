import scipy as sp
import numpy as np
import gym
import matplotlib.pyplot as plt
from skimage import color, transform
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction import image
import copy
import torch

# Make the obervations from the environment more compact
def feat_extract(obs):
    #Crop area of interest
    obs = obs[52:196,9:152,:]
    # Convert to grayscale
    obs = color.rgb2gray(obs)
    # Rescale
    obs = transform.rescale(obs,1,anti_aliasing=True)
    # Return, rescaled cropped grayscale observation
    return obs


# Make the obervations from the environment more compact
def feat_extract2(obs):
    #Crop area of interest
    obs_r = obs[52:196,9:152,:]
    obs_r = color.rgb2gray(obs_r)
    obs_r[obs_r != 0] = 255
    # # Rescale
    obs_r = transform.rescale(obs_r,0.5,anti_aliasing=True)
    print(np.shape(obs_r))

    # ==== FIND CENTROIDS
    non0 = np.transpose(np.nonzero(obs_r))

    dbscan = DBSCAN(eps=3,min_samples=1).fit(non0)
    labels = dbscan.labels_

    len0 = len(non0[labels==0])
    len1 = len(non0[labels==1])
    len2 = len(non0[labels==2])

    centroid_0 = np.append(np.mean(non0[labels==0],axis=0),len0)
    centroid_1 = np.append(np.mean(non0[labels==1],axis=0),len1)
    centroid_2 = np.append(np.mean(non0[labels==2],axis=0),len2)

    centroids = np.array([centroid_0,centroid_1,centroid_2])
    centroids = centroids[centroids[:,2].argsort()]

    if np.isnan(centroids[0,0]):
        centroids[0,0] = centroids[1,0]
        centroids[0,1] = centroids[1,1]

    return torch.tensor([centroids[0,0],centroids[0,1],centroids[1,0],centroids[1,1]])

# Check what ur function is doing. If you mand to mod for other games or smth
if __name__ =='__main__':
    env = gym.make('Breakout-v4')
    obs = env.reset()
    #obs = feat_extract(obs)
    obs1,a,b,_ = env.step(1)
    obs2,a,b,_ = env.step(2) 
    obs3,a,b,_ = env.step(2)
    centroids = feat_extract2(obs)
    print(centroids)
    #plt.imshow(obs2, cmap='Greys')
    
    #plt.show()
