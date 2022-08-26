import torch
import numpy as np
import torch.optim
from replay_buffer import Observation
import random
import copy
import sys
from DQN import DeepQNet, DeepQNet2
import gc

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
        self.eps_step = (eps_initial-eps_final)/eps_step

        self.device = device
        # Process model using GPU
        self.model.to(device)
        self.target.to(device)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        self.loss_func = torch.nn.HuberLoss(reduction='mean')

   
    def populate(self):
        
        # Initialize buffer with random obervations
        obs0 = self.env.reset()
        

        while True:
            # Choose Random action
            act = self.env.action_space.sample()
            # Take action
            obs1,rew,done,_ = self.env.step(act)

            # Decide if you store it or not for added randomness and variabilioty of experiencws
            if random.randint(0,1):

                observation = Observation(obs0,act,rew,done,obs1)
                self.buffer.append(observation)
            # Check if the buffer is ready
            if self.buffer.len() > 100:
                print('Buffer Ready, Start Training')
                break
            # Copy obs1 into obs0 for use in the next Observation
            obs0 = copy.copy(obs1)
            # Check if environment needs to be reset
            if done:
                obs0 = self.env.reset()
                


    def evaluate_batch_loss(self):

        # Lists containing the observations
        states,actions,rewards,dones,next_states = self.buffer.sample(self.batch_size)
    
        # Prep the batch
        states_tensor = torch.from_numpy(np.array(copy.copy(states))).squeeze().to(self.device).type(torch.float32)
        next_states_tensor = torch.from_numpy(np.array(copy.copy(next_states))).squeeze().to(self.device).type(torch.float32)
        actions_tensor = torch.tensor(copy.copy(actions)).to(self.device)
        actions_tensor = actions_tensor.view(self.batch_size,1)
        rewards_tensor = torch.from_numpy(np.array(copy.copy(rewards))).to(self.device).squeeze().type(torch.float32)
        dones_tensor = list(np.array(copy.copy(dones)))
        
        
        # Predictions given the initial state
        predicted = self.model.forward(states_tensor).gather(1,actions_tensor)[:,0]
       
        # Targets
        targets = self.discount*self.target.forward(next_states_tensor).max(1)[0].detach()
        targets = rewards_tensor + targets

        # Make resulting reward for losing the ball Negarive. Makes training much better, droppoing ball extra unappealing
        targets[dones_tensor] = -1 
        
        # Calculate Batch Loss
        loss = self.loss_func(predicted,targets)  

        # Return mean loss of samples
        return loss

    def pick_action(self,obs):

        # Calculate Q values
        
        with torch.no_grad():
            
            # Process Observation
            
            obs = torch.from_numpy(np.array(obs)).type(torch.float32).unsqueeze(0)
            q_vals = self.model.forward(obs.to(self.device))

            # APply Current Greedy Policy
            if np.random.random() < self.eps_initial:
                    act = self.env.action_space.sample()
            else:
                act = torch.argmax(q_vals).item()
                
            # Take greedy step
            if self.eps_initial > self.eps_final:
                self.eps_initial -= self.eps_step
            # Calculate Mean Q
            mean_q = torch.mean(q_vals).item()

        return act, mean_q
        
    
    def optimize_step(self):

        self.optimizer.zero_grad()
        loss = self.evaluate_batch_loss() 
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
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

        while True:

            # Update Frame Counter
            frame_count += 1
        
            # Select an action with greedy policy
            act, q_val = self.pick_action(obs0)
            #Env Step
            obs1, rew, done,_ = self.env.step(act)
            # Record what you just saw
            self.buffer.append(Observation(obs0,act,rew,done,obs1))
            # Swap observations
            obs0 = copy.copy(obs1)
            
            # Append to lists
            q_vals.append(q_val)
            reward += rew
            


            # Gradient Updates
            if frame_count % self.gradient_update == 0:

                batch_loss = self.optimize_step()
                losses.append(batch_loss)
            
            # Update Target Network
            if frame_count % self.update_target == 0:
                self.target.load_state_dict(self.model.state_dict())
                print('Target Network Updated! , Eps:{}'.format(self.eps_initial))

            # If done, reset the environment
            if done:
                obs0 = self.env.reset()
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
        
        # Lists to hold treining Dtaa
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

            
            # Save Snapshot of Model
            path = 'TrainedModels/' + name_of_run + '_{}'.format(x) +'.pt'
            torch.save(self.model, path)


        return self.model, losses, returns, q_vals, ep_len










       


    

    
    
