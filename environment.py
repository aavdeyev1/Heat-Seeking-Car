"""The environment setup to run the heat-seeking car in.
observation: This is the part of the environment state that the agent can observe to choose its actions at the next step.
reward: The agent is learning to maximize the sum of these rewards across multiple steps.
step_type: Interactions with the environment are usually part of a sequence/episode. e.g. multiple moves in a game of chess. step_type can be either FIRST, MID or LAST to indicate whether this time step is the first, intermediate or last step in a sequence.
discount: This is a float representing how much to weight the reward at the next time step relative to the reward at the current time step.
These are grouped into a named tuple TimeStep(step_type, reward, discount, observation).

The interface that all Python environments must implement is in environments/py_environment.PyEnvironment. The main methods are:


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import helper_funcs as hf
from pipeline import Pipeline
from pipeline_historian import PipelineHistorian

import tensorflow as tf
import numpy as np
import time

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

TIMOUT_MINS = 5
t_dim = (24, 32)

# reset_t_array = np.zeros((24,32), dtype=np.float32)
reset_observation_vector = np.zeros((6,24,32), dtype=np.float32)
reset_observation_vector[:, 0, 0] = [500.0, 500.0, 500.0, 500.0, 0.1, 0]

def mock_buffer_to_arrays(old_d, old_t):
  new_d = old_d
  new_t = old_t
  new_t = new_t*new_t
  return new_d, new_t

class RCCarEnv(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
    
    # [N, E, S, W, %] distances, current % of thermal view
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(6,24,32), dtype=np.float32, minimum=0, maximum=500.0, name='observation')

    self._observation = reset_observation_vector

    self._timout_after = time.time() + 60 * TIMOUT_MINS

    # current % of thermal view
    self._state = self._observation[4, 0, 0]
    self._episode_ended = False
    self._pl_hist = PipelineHistorian()

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._episode_ended = False
    self._pl_hist.reset()
    self._observation = reset_observation_vector
    return ts.restart(reset_observation_vector)

  def _step(self, next_action):
    
    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()
    
    # Make sure episodes don't go on forever.
    if self._observation[4, 0, 0] > .85:
      # End episode if target found, or T% is > .85
      print("Target Found, ending episode...")
      # Send Stop command to car
      self._episode_ended = True
    elif time.time() > self._timout_after:
      print("Timeout...")
      # Send Stop command to car
      self._episode_ended = True
    elif next_action >= 0 and next_action < 4:
      # Calculate prev/current reward
      prev_pl = self._pl_hist.current_pl
      prev_d_score = prev_pl.d_score
      # prev_reward = prev_pl.get_reward(prev_t_percent=prev_t, prev_d_score=prev_d)

      # busy wait til next Buffer comes in
      # TODO: remove + replace with while not DATA_BUFFER.msg_flag:
      while False:
        pass

      # Convert to usable arrays NUMPY arrs
      # t_array, d_array, lat, long = DATA_BUFFER.buffer_to_arrays()
      mock_t = np.array([float(i) for i in range(768)], dtype=np.float32).reshape(t_dim)
      mock_d = np.array([100, 100, 25, 25], dtype=np.float32)

      # Generate pipeline for next step
      new_pipeline = Pipeline(mock_t, mock_d, prev_t_percent=self._observation[4], prev_d_score=prev_d_score)

      # TODO - remove
      new_pipeline.t_percent = self._observation[4, 0, 0]+.2

      self._pl_hist.add_step(new_pipeline)

      # Update observation vector to send to the next time step
      new_observation_vector = np.zeros((6,24,32), dtype=np.float32)
      new_observation_vector[0:4, 0, 0] = [i for i in new_pipeline.d_array]
      new_observation_vector[4, 0, 0] = new_pipeline.t_percent
      new_observation_vector[5] = new_pipeline.t_array
      self._observation = new_observation_vector
      
    else:
      raise ValueError('`action` should be 0 to 3.')

    if self._episode_ended or self._observation[4, 0, 0] > .85:
      # reward = self._pls[self._pl_idx].get_reward(self._observation_spec)
      return ts.termination(reset_observation_vector, reward=100)
    else:
      # Valid action, carry out next turn
      print("Transmitting Next Turn to RC car...")
      # TODO: Uncomment
      # publish.single("HeatSeekingCar/tx", f"{mock_turn}", hostname="test.mosquitto.org")
      # DATA_BUFFER.msg_flag = False

      return ts.transition(
          np.array(new_observation_vector, dtype=np.float32), reward=new_pipeline.reward, discount=1.0)

if __name__ == "__main__":
  test_env = RCCarEnv()
  # utils.validate_py_environment(test_env, episodes=5)

  tf_env = tf_py_environment.TFPyEnvironment(test_env)
  # reset() creates the initial time_step after resetting the environment.
  time_step = tf_env.reset()
  num_steps = 4
  transitions = []
  reward = 0
  for i in range(num_steps):
    action = tf.constant(i)
    # applies the action and returns the new TimeStep.
    next_time_step = tf_env.step(action)
    transitions.append([time_step, action, next_time_step])
    reward += next_time_step.reward
    time_step = next_time_step
    print(time_step)

  np_transitions = tf.nest.map_structure(lambda x: x.numpy(), transitions)
  print('\n'.join(map(str, np_transitions)))
  print('Total reward:', reward.numpy())
  # get_new_action = np.array([0, 1, 2, 3], dtype=np.int32)
  # end_round_action = np.array(4, dtype=np.int32)

  # new_obs_vectors = np.array( [[30, 100, 30, 30, .65],
  #                             [30, 100, 30, 30, .65],
  #                             [30, 100, 30, 30, .65],
  #                             [30, 100, 30, 30, .65],])

  # environment = RCCarEnv()
  # time_step = environment.reset()
  # print(time_step)
  # cumulative_reward = time_step.reward

  # # for i in range(3):
  # #   time_step = environment.step(get_new_card_action)
  # #   print(time_step)
  # #   cumulative_reward += time_step.reward

  # time_step = environment.step(end_round_action)
  # print(time_step)
  # cumulative_reward += time_step.reward
  # print('Final Reward = ', cumulative_reward)



########