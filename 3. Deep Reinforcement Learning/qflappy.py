#################
#### IMPORTS ####
#################

# Arrays
import numpy as np
from collections import deque # fixed size FIFO list

# Deep Learning Stuff
import torch
from torch import nn
import gym

# Second set
import flappy_bird_gym
import time

####################
#### DEEPQCLASS ####
####################

class ReplayMemory(object):
    '''
    To perform experience replay.
    We will draw uniformly at random from the pool of stored sample to learn.
    Thi avoids (temporal) correlation between consecutive learning instances.
    '''
    def __init__(self, capacity):
        ''' Initialize a deque with maximum capacity maxlen. '''
        self.memory = deque(maxlen=capacity) # Define a queue with maxlen "capacity"

    def push(self, state, action, next_state, reward):
        ''' Add a new sample to the deque, removes the oldest one if it is already full. '''
        # Add the tuple (state, action, next_state, reward) to the queue
        self.memory.append( (state, action, next_state, reward) )
        
    def sample(self, batch_size):
        ''' Randomly select "batch_size" samples '''
        batch_size = min(batch_size, len(self)) # Get all the samples if the requested batch_size is higher than the number of sample currently in the memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        ''' Return the number of samples currently stored in the memory '''
        return len(self.memory)

class DQN(nn.Module):
    ''' 
    Network for policy network and target network 
    state_space_dim:  (INPUT)  dimension of state space (e.g pixels in a image)
    action_space_dim: (OUTPUT) dimension of action space (e.g go left, go right)
    '''
    def __init__(self, DQN_state_space_dim, DQN_action_space_dim):
        super().__init__()
            
        self.sdim = DQN_state_space_dim
        self.adim = DQN_action_space_dim
            
        self.linear = nn.Sequential(
            nn.Linear(self.sdim, 128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,self.adim)
                )

    def forward(self, x):
        return self.linear(x)

#
#          |---------> [Prediction Network (DQN)]--------
#          |                |                            \
# [INPUT]--|                | Parameter update            \___Loss
#          |               \/                            /
#          |---------> [Target Network (DQN)]------------
class Flappy(nn.Module):
    ''' 
    Handles all the networks, environments, and others
    '''
    def __init__(self):
        super().__init__()
        
        tempenv = flappy_bird_gym.make("FlappyBird-v0")
        self.state_space_dim = tempenv.observation_space.shape[0]
        
        self.action_space_dim = tempenv.action_space.n
        self.policy_net = DQN(self.state_space_dim, self.action_space_dim)
            
        self.policy_net.load_state_dict(torch.load('./models/fb1'))
        
    def choose_action_epsilon_greedy(self, state, epsilon):
        self.policy_net.eval()
        if epsilon > 1 or epsilon < 0:
            raise Exception('The epsilon value must be between 0 and 1')
                
        # Evaluate the network output from the current state
        with torch.no_grad():
            self.policy_net.eval()
            state = torch.tensor(state, dtype=torch.float32) # Convert the state to tensor
            
            net_out = self.policy_net(state)

        # Get the best action (argmax of the network output)
        best_action = int(net_out.argmax())
        
        return best_action, net_out.cpu().numpy()
    
    def play_a_game(self, show = True):
        env = flappy_bird_gym.make("FlappyBird-v0")
        
        # Reset the environment and get the initial state
        state = env.reset()
        
        # Reset the score. The final score will be the total amount of steps before the pole falls
        score = 0
        done = False
        # Go on until the pole falls off or the score reach 490
        while not done:
            # Choose the best action (temperature 0)
            action, q_values = self.choose_action_epsilon_greedy(state, 0)
            
            # Apply the action and get the next state, the reward and a flag "done" that is True if the game is ended
            next_state, reward, done, info = env.step(action)
            
            # Visually render the environment
            if show:
                env.render()
                time.sleep(1 / 30)  # FPS
                print(score)
                
            # Update the final score (+1 for each step)
            score += reward 
            # Set the current state for the next iteration
            state = next_state
            
        # Print the final score
        if show:
            print(f"SCORE: {score}") 
        env.close()
        
        return score, state        
        
player = Flappy()

player.play_a_game()