
import numpy as np
import random
import environment
from math import ceil

env = environment.RCCarEnv()

# New frame is averaged, so 12x16 pixels in the image
values_max = 40
values_min = 25
# Distances are 1-400 rounded to the nearest tenth in 4 directions
max_distance = int(ceil((400 - 10) / 10))
num_dirs = 4

action_size = env._action_spec.maximum + 1
print("Action size ", action_size)

state_size = env._observation_space_size*(values_max-values_min)*max_distance*num_dirs
print("State size ", state_size)

qtable = np.zeros((state_size, action_size))

# Hyperparams
total_episodes = 3        # Total episodes 50000
total_test_episodes = 1     # Total test episodes 100
max_steps = 5                # Max steps per episode 99

learning_rate = 0.7           # Learning rate
gamma = 0.618                 # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration prob

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    obs = env.reset()
    step = 0
    done = False
    
    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0,1)
        
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[obs,:])
        
        # Else doing a random choice --> exploration
        else:
            action = random.randrange(4)
        
        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_obs, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        qtable[obs, action] = qtable[obs, action] + learning_rate * (reward + gamma * 
                                    np.max(qtable[new_obs, :]) - qtable[obs, action])
                
        # Our new state is state
        obs = new_obs
        
        # If done : finish episode
        if done == True: 
            break
    
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 

print(qtable)