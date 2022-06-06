"""Script to test the qtable of the reinforcement learning heat-seeking car."""
import sys
import numpy as np
import random
import environment
from math import ceil
from numpy import savetxt, loadtxt
from errors import SaveAndExitError

def save_and_exit(qtable_file, qtable_save):
    """Save qtable in case of an error so we don't lose the entire qtable. """
    savetxt(qtable_file, qtable_save, delimiter=',', encoding="utf-8")
    qtable_file.close()
    sys.exit("Saved and Exited...")

# Saved Qtable
INITIALIZED = True
q_filename = "q_table_final.csv"

# Init environment
env = environment.RCCarEnv()

# Calculate Qtable Size
# Max temperature rounded to 40 degrees C (avg human body temp is 37 degrees C)
values_max = 40
# Minimum will be rounded to room temp
values_min = 25
# Distances are 1-400 rounded to the nearest tenth in 4 directions
max_distance = int(ceil((400 - 10) / 10))
# Four ultrasonic sensors in 4 directions
num_dirs = 4
action_size = env.action_spec + 1
print("Action size ", action_size)
# New frame is averaged, so 12x16 pixels in the image (env.observation_space_size)
state_size = env.observation_space_size*(values_max-values_min)*max_distance*num_dirs
print("State size ", state_size)

# Hyperparams (Credit to Thomas Simonini)
total_episodes = 1             # Total episodes 50000
total_test_episodes = 1        # Total test episodes 100
max_steps = 149                # Max steps per episode 99

learning_rate = 0.7            # Learning rate
gamma = 0.618                  # Discounting rate

# Exploration parameters
epsilon = 1.0                  # Exploration rate
max_epsilon = 1.0              # Exploration probability at start
min_epsilon = 0.01             # Minimum exploration probability 
decay_rate = 0.008             # Exponential decay rate for exploration prob

if INITIALIZED:
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
    # Reset the environment  
    obs = env.reset().observation
    step = 0
    done = False

    for step in range(max_steps):
        # This flag shows us if we need to save the q-value (if this step was explorative)
        exploration_flag = False

        # This random number allows us to explore 'epsilon' percent of the time
        exp_exp_tradeoff = random.uniform(0,1)

        # If exp_exp_tradeoff > epsilon, we will exploit the qtable for that observation (take
        # the step with the greatest q-value for a given observation)
        if exp_exp_tradeoff > epsilon:
            # Q-table at a certain observation is 6x12x16x4. We index the first [0,0,0] row
            # with 4 entries, the q-value for each possible turn at a given observation.
            print("Exploiting...")
            row = qtable[obs,:][0][0][0]
            action = np.argmax(row)
            print(action)
        else:
            # Else doing a random choice exploration
            print("Exploring...")
            exploration_flag = True
            action = random.randrange(4)

        try:
            # Take the step (action) and recieve resulting new observation (info) and reward
            new_obs, reward, done, next_obs = env.step(action)
        except SaveAndExitError:
            save_and_exit(f, qtable)
        except KeyboardInterrupt:
            save_and_exit(f, qtable)

        # If you took a new step, not done before, you save to the qtable.
        if exploration_flag:
            try:
                qtable[obs, action] = qtable[obs, action] + learning_rate * (reward + gamma *
                                            np.max(qtable[next_obs, :]) - qtable[obs, action])
            except IndexError:
                save_and_exit(f, qtable)

        # Our new observation is the previous step's "info" (based on how py_env is built)
        obs = next_obs

        # If high reward (found target) finish episode
        if reward == 100:
            break
        # If num steps is too big, end episode
        if step == max_steps - 1:
            print("Ran out of Steps. Ending Episode...")
        # Reduce epsilon EVERY STEP (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*step)
except SaveAndExitError:
    save_and_exit(f, qtable)
except KeyboardInterrupt:
    save_and_exit(f, qtable)

env.close()
save_and_exit(f, qtable)
