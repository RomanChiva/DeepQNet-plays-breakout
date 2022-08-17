import torch
import numpy as np
import torch.optim
from replay_buffer import Observation
import random
import copy
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
    def __init__(self,device,env,buffer,discount,update_target,gradient_update,epochs,updates_per_epoch,lr,batch_size,eps_initial,eps_final,eps_step) -> None:
        
        # Load external hyperparams
        self.env = env
        self.seed = int(np.random.random()*1e5)
        torch.manual_seed(self.seed)
        self.model = DeepQNet(self.env.action_space.n)
        self.target = DeepQNet(self.env.action_space.n)
        self.update_target = update_target
        self.gradient_update = gradient_update
        self.lr = lr
        self.discount = discount
        self.batch_size = batch_size
        self.buffer = buffer
        self.epochs = epochs
        self.updates_per_epoch = updates_per_epoch
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_step = eps_step

        self.device = device
        # Process model using GPU
        self.model.to(device)
        self.target.to(device)

        # Obs tensor template
        #self.obs_template = torch.empty((1,self.skip,86,86))
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        self.loss_func = torch.nn.HuberLoss()

    # def step(self,act, init= False):

    #     # Initialize Episode

    #     if init:
    #         obs = copy.copy(self.obs_template)
    #         obs_s = self.env.reset()
    #         for x in range(self.skip):
    #             obs[0,x,:,:] = torch.from_numpy(feat_extract(obs_s))
            
    #         return obs.to(self.device)

    #     # Steps that are not initial

    #     if not init:    
    #         obs = copy.copy(self.obs_template)
    #         rew = []
            
    #         for x in range(self.skip):
    #             obs_s,rew_s,done,_ = self.env.step(act)
    #             obs[0,x,:,:] = torch.from_numpy(feat_extract(obs_s))
    #             rew.append(rew_s)
                
    #         return obs.to(self.device), sum(rew), done



    def populate(self):
        
        # Initialize buffer with random obervations
        obs0 = torch.from_numpy(np.array(self.env.reset())).to(self.device)
        obs0 = torch.unsqueeze(obs0,0).permute(0,3,1,2)

        while True:
            # Choose Random action
            act = random.randint(0,3)
            # Take action
            obs1,rew,done,_ = self.env.step(act)

            # Shape and convert to tensor
            obs1 = torch.from_numpy(np.array(obs1)).to(self.device)
            obs1 = torch.unsqueeze(obs1,0).permute(0,3,1,2)
            # Decide if you store it or not for added randomness and variabilioty of experiencws
            if random.randint(0,1):

                observation = Observation(obs0,act,rew,done,obs1)
                self.buffer.append(observation)
            # Check if the buffer is ready
            if self.buffer.len() > 500:
                print('Buffer Ready, Start Training')
                break
            # Copy obs1 into obs0 for use in the next Observation
            obs0 = copy.copy(obs1)
            # Check if environment needs to be reset
            if done:
                obs0 = torch.from_numpy(np.array(self.env.reset())).to(self.device)
                obs0 = torch.unsqueeze(obs0,0).permute(0,3,1,2)


    def evaluate_batch_loss(self):

        # Lists containing the observations
        states,actions,rewards,dones,next_states = self.buffer.sample(self.batch_size)
        # Array to store losses
        losses = torch.empty(self.batch_size).to(self.device)

        for x in range(self.batch_size):
            #calculate target and predicted
            predicted = self.model.forward(states[x])[actions[x]]
            target = rewards[x] + self.discount*max(self.target.forward(next_states[x]))
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
        losses = self.evaluate_batch_loss() 
        losses.backward()
        self.optimizer.step()
        
        return torch.mean(losses)
    
    def epoch(self,epoch_number):

        # Run N= batch_size steps, and then perform gradient update

        # Relevant parameters
        reward = 0
        q_vals = []
        losses = []

        episode_count = 0

        # Counters
        frame_count = 0

        # Initialize environment
        obs0 = self.env.reset()
        obs0 = obs0 = torch.from_numpy(np.array(self.env.reset())).to(self.device)
        obs0 = torch.unsqueeze(obs0,0).permute(0,3,1,2)

        while True:

            # Update Frame Counter
            frame_count += 1
        
            # Select an action with greedy policy
            act, q_val = self.pick_action(obs0)
            #Env Step
            obs1, rew, done,_ = self.env.step(act)
            obs1 = torch.from_numpy(np.array(obs1)).to(self.device)
            obs1 = torch.unsqueeze(obs1,0).permute(0,3,1,2)
            # Record what you just saw
            self.buffer.append(Observation(obs0,act,rew,done,obs1))
            # Swap observations
            obs0 = copy.copy(obs1)
            

            # Append to lists
            q_vals.append(q_val.detach().cpu())
            reward += rew


            # Gradient Updates
            if frame_count % self.gradient_update == 0:

                batch_loss = self.optimize_step()
                losses.append(batch_loss.detach().cpu())

            
            # Update Target Network
            if frame_count % self.update_target == 0:
                self.target.load_state_dict(self.model.state_dict())
                print('Target Network Updated!')

            # If done, reset the environment
            if done:
                obs0 = self.env.reset()
                obs0 = torch.from_numpy(np.array(self.env.reset())).to(self.device)
                obs0 = torch.unsqueeze(obs0,0).permute(0,3,1,2)
                episode_count +=1

            # End Epoch, Calculate Statistics and Give Feedback
            if frame_count > self.updates_per_epoch*self.gradient_update  and done:
                break



        epoch_loss = np.mean(np.array(losses))
        epoch_return = reward/episode_count 
        average_q_val = np.mean(np.array(q_vals))
        average_episode_length = frame_count/episode_count

        print('Loss:', epoch_loss)
        print('Return:',epoch_return)
        print('Average Max-Q-Value:', average_q_val)
        print('Average Episode Length:', average_episode_length)

        return epoch_loss, epoch_return, average_q_val,average_episode_length
            


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










       


    

    
    
