import torch
import numpy as np
import torch.optim
from replay_buffer import Observation
import random
import copy
from feature_extractor import feat_extract
import sys
from DQN import DeepQNet

class Trainer:

    """
        Class designed to train more easily and intuitively.
        Do the dirty work behind the scenes and allow to test
        different models and fine tune hyperparameters more effectively

        Inputs:
            env: Environment we are using to train
            model: The DQN that will be optimized
            buffer: The buffer object used to store experience
            skip: CHoose action every [skip] frames and repeat for others
            n_episodes: Number of episodes
            ep_len_max: Maximum episode length
            eps_initial: epsilon initial value
            eps_final: epsilon final value
            eps_step: steps

    """ 
    def __init__(self,device,env,buffer,skip,epochs,updates_per_epoch,lr,batch_size,eps_initial,eps_final,eps_step) -> None:
        
        # Load external hyperparams
        self.env = env
        self.seed = int(np.random.random()*1e5)
        torch.manual_seed(self.seed)
        self.model = DeepQNet(skip,self.env.action_space.n)
        self.lr = lr
        self.batch_size = batch_size
        self.buffer = buffer
        self.skip = skip
        self.epochs = epochs
        self.updates_per_epoch = updates_per_epoch
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_step = eps_step

        self.device = device
        # Process model using GPU
        self.model.to(device)

        # Obs tensor template
        self.obs_template = torch.empty((1,self.skip,86,86))
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        self.loss_func = torch.nn.MSELoss()

    def step(self,act, init= False):

        # Initialize Episode

        if init:
            obs = copy.copy(self.obs_template)
            obs_s = self.env.reset()
            for x in range(self.skip):
                obs[0,x,:,:] = torch.from_numpy(feat_extract(obs_s))
            
            return obs.to(self.device)

        # Steps that are not initial

        if not init:    
            obs = copy.copy(self.obs_template)
            rew = []
            
            for x in range(self.skip):
                obs_s,rew_s,done,_ = self.env.step(act)
                obs[0,x,:,:] = torch.from_numpy(feat_extract(obs_s))
                rew.append(rew_s)
                
            return obs.to(self.device), sum(rew), done



    def populate(self):
        
        # Initialize buffer with random obervations
        obs0 = self.step(0,init=True)

        while True:
            # Choose Random action
            act = random.randint(0,3)
            # Take action
            obs1,rew,done = self.step(act)
            # Decide if you store it or not for added randomness and variabilioty of experiencws
            if random.randint(0,1):

                observation = Observation(obs0,act,rew,done,obs1)
                self.buffer.append(observation)
            # Check if the buffer is ready
            if self.buffer.ready():
                break
            # Copy obs1 into obs0 for use in the next Observation
            obs0 = copy.copy(obs1)
            # Check if environment needs to be reset
            if done:
                obs0 = self.step(0, init=True)

    def evaluate_batch_loss(self):

        # Lists containing the observations
        states,actions,rewards,dones,next_states = self.buffer.sample(self.batch_size)
        # Array to store losses
        losses = torch.empty(self.batch_size)

        for x in range(self.batch_size):
            #calculate target and predicted
            predicted = self.model.forward(states[x])[actions[x]]
            target = rewards[x] + max(self.model.forward(next_states[x]))
            loss = self.loss_func(predicted,target)
            losses[x] = loss

        # Return mean loss of samples
        return torch.mean(losses)

    def pick_action(self,obs):

        # Calculate Q values
        q_vals = self.model.forward(obs)

        if np.random.random() < self.eps_initial:
                act = self.env.action_space.sample()
        else:
            act = torch.argmax(q_vals)
        
        if self.eps_initial > self.eps_final:
            self.eps_initial -= self.eps_step

        return act, torch.max(q_vals)
    
    def optimize_step(self):

        self.optimizer.zero_grad()
        batch_loss = self.evaluate_batch_loss() 
        batch_loss.backward()
        self.optimizer.step()
        
        return batch_loss
    
    def epoch(self,epoch_number):

        # Run N= batch_size steps, and then perform gradient update

        # Relevant parameters
        reward = 0
        episode_length = []
        q_vals = []
        losses = []

        # Initialize
        gradient_update_counter = 0
        gradient_updates = 0
        episode_length_counter = 0
        episodes = 0
        obs0 = self.step(0,init=True)

        while True:

            # Select an action with greedy policy
            act, q_val = self.pick_action(obs0)
            #Env Step
            obs1, rew, done = self.step(act)
            # Record what you just saw
            self.buffer.append(Observation(obs0,act,rew,done,obs1))
            # Swap observations
            obs0 = copy.copy(obs1)
            # Add to counter 
            gradient_update_counter += 1
            episode_length_counter +=1

            # Append to lists
            q_vals.append(q_val.detach().cpu())

            if rew != 0:
                reward += rew
            

            if gradient_update_counter >= self.batch_size:
                batch_loss = self.optimize_step().detach().cpu()
                losses.append(batch_loss)
                gradient_update_counter = 0
                gradient_updates +=1
                print('Epoch:{e_num}, Mini-Batch:{a}'.format(e_num=epoch_number,a=gradient_updates))
                


            if done:

                obs0 = self.step(0,init=True)
                episode_length.append(episode_length_counter)
                episode_length_counter = 0
                episodes +=1

                # When the final num of gradient updates is reached, stop the loop after the episode ends
                if gradient_updates >= self.updates_per_epoch:
                    break

        epoch_loss = np.mean(np.array(losses))
        epoch_return = reward/episodes 
        average_q_val = np.mean(np.array(q_vals))
        average_episode_length = np.mean(np.array(episode_length))

        print('Loss:', batch_loss)
        print('Return:',epoch_return)
        print('Average Max-Q-Value:', average_q_val)
        print('Average Episode Length:', average_episode_length)

        return batch_loss,epoch_return,average_q_val,average_episode_length
            


    def train(self, name_of_run):

        losses = []
        returns = []
        q_vals = []
        ep_len = []

        for x in range(self.epochs):
            print('EPOCH:{}'.format(x))
            batch_loss,epoch_return,average_q_val,average_episode_length = self.epoch(x)
            losses.append(batch_loss)
            returns.append(epoch_return)
            q_vals.append(average_q_val)
            ep_len.append(average_episode_length)


            path = 'TrainedModels/' + name_of_run + '_{}'.format(x) +'.pt'
            torch.save(self.model, path)


        return self.model, losses, returns, q_vals, ep_len










       


    

    
    
