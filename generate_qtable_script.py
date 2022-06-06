
import sys
import numpy as np
import random
import environment
from math import ceil
from numpy import savetxt, loadtxt
from errors import SaveAndExitError

def save_and_exit(qtable_file, qtable_save):
    # Save qtable so we don't error out and lose progress
    savetxt(qtable_file, qtable_save, delimiter=',', encoding="utf-8")
    qtable_file.close()
    sys.exit("Saved and Exited...")

# Saved Qtable Params and epsilon
initialized = True
q_filename = "q_table.csv"

# Init environment
env = environment.RCCarEnv()

# Calculate Qtable Size
# New frame is averaged, so 12x16 pixels in the image
values_max = 40
values_min = 25
# Distances are 1-400 rounded to the nearest tenth in 4 directions
max_distance = int(ceil((400 - 10) / 10))
num_dirs = 4
action_size = env.action_spec + 1
print("Action size ", action_size)
state_size = env.observation_space_size*(values_max-values_min)*max_distance*num_dirs
print("State size ", state_size)

# Hyperparams
total_episodes = 1            # Total episodes 50000
total_test_episodes = 1       # Total test episodes 100
max_steps = 149               # Max steps per episode 99

learning_rate = 0.7           # Learning rate
gamma = 0.618                 # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration prob

if initialized:
    # Load previous qtable or initialize
    f = open(q_filename, "r", encoding="utf-8")
    qtable = loadtxt(f, delimiter=',')
    print("Loaded qtable from existing file...")
    f.close()
else:
    print("Creating new qtable...")
    qtable = np.zeros((state_size, action_size))

f = open(q_filename, "w", encoding="utf-8")
try:
    # Take the action (a) and observe the outcome state(s') and reward (r)
    # 2 For life or until learning is stopped
    # Reset the environment
    obs = env.reset().observation
    step = 0
    done = False

    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0,1)

        ## If this number > greater than epsilon --> exploitation
        # (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[obs,:])

        # Else doing a random choice --> exploration
        else:
            action = random.randrange(4)

        try:
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_obs, reward, done, info = env.step(action)
        except SaveAndExitError:
            save_and_exit(f, qtable)
        except KeyboardInterrupt:
            save_and_exit(f, qtable)

        try:
            qtable[obs, action] = qtable[obs, action] + learning_rate * (reward + gamma *
                                        np.max(qtable[info, :]) - qtable[obs, action])
        except IndexError:
            save_and_exit(f, qtable)

        # Our new observation is the previous step's "info"
        obs = info

        # If high reward (found target) finish episode
        if reward == 100:
            break
        # If num steps is too big, end episode
        if step == max_steps - 1:
            print("Ran out of Steps. Ending Episode...")
except SaveAndExitError:
    save_and_exit(f, qtable)
except KeyboardInterrupt:
    save_and_exit(f, qtable)

env.close()
save_and_exit(f, qtable)
