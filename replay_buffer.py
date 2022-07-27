import numpy as np
from collections import namedtuple, deque

# Create a named tuple to store experiences gatheted in training
Observation = namedtuple('Observation', 
        field_names=['state','action','reward','done','new_state'])


class ReplayBuffer:

    # Store the gathered experience by the agent as it explores
    # Input: Capacity of the buffer. How much memory does it have
    # Output: Batch Size x (Named tuple (Experience above defined))

    def __init__(self,capacity) -> None:

        # Deque data type convenient here more than a list
        # since we will be appending at the front and deleting
        # at the end very often. This does that for us
        self.capacity = capacity

        self.buffer = deque(maxlen=capacity)

    # Allows us to easily check the length of out buffer
    def len(self):
        return len(self.buffer)
    
    # Check buffer is filled up when popuating at the start of training
    def ready(self):

        if self.len() == self.capacity:
            return True
        else:
            return False
        

    # Add observations to the buffer
    def append(self,observation):
        self.buffer.append(observation)


    # Function used to sample the buffer. Batch size modulates how many samples
    def sample(self, batch_size):

        # Lists to store vars
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []    


        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        #states,actions,rewards,dones,next_states = zip(*(self.buffer[i] for i in indices))

        for x in indices:
            observation = self.buffer[x]
    
            states.append(observation.state)
            actions.append(observation.action)
            rewards.append(observation.reward)
            dones.append(observation.done)
            next_states.append(observation.new_state)

        return (
            states,
            actions,
            rewards,
            dones,
            next_states)

