"""The environment setup to run the heat-seeking car in.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import helper_funcs as hf
from pipeline import Pipeline
from pipeline_historian import PipelineHistorian
from connector import Connector

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

TIMOUT_MINS = 15
# t_dim = (24, 32)
t_dim = (12, 16)

reset_observation_vector = np.zeros((6,t_dim[0],t_dim[1]), dtype=np.int32)
reset_observation_vector[:, 0, 0] = [500.0, 500.0, 500.0, 500.0, 0.1, 0]


class RCCarEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        
        # [N, E, S, W, %] distances, current % of thermal view
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(6,t_dim[0],t_dim[1]), dtype=np.int32, minimum=0, maximum=500, name='observation')

        self._observation = reset_observation_vector

        self._timout_after = time.time() + 60 * TIMOUT_MINS

        # current % of thermal view
        self._state = self._observation[4, 0, 0]
        self._episode_ended = False
        self._pl_hist = PipelineHistorian()

        self._observation_space_size = t_dim[0]*t_dim[1]*(self._action_spec.maximum + 1)

        self.conn = Connector()
    
    def check_timeout(self):
        if time.time() > self._timout_after:
            self.conn.loop_stop()
            return True
        else:
            return False

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
        # TODO: uncomment  when figure out attribute not found error..
        self.check_timeout()
        lat = None

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

            # # busy wait til next Buffer comes in
            while lat == None:
                t_array, d_array, lat, long = self.conn.check_for_data()
                time.sleep(1)    
        else:
            raise ValueError('`action` should be 0 to 3.')

        if self._episode_ended or self._observation[4, 0, 0] > .85:
            return ts.termination(reset_observation_vector, reward=100)
        else:
            # Check if fatal action
            self._pl_hist.current_pl.check_if_fatal(next_action)
            # Valid action returned guaranteed, carry out next turn
            self.conn.send_turn(next_action)
            mock_t = t_array
            mock_d = d_array

            # TODO - remove;
            # Convert to usable arrays NUMPY arrs
            # mock_t = np.array([float(i) for i in range(t_dim[0]*t_dim[1])], dtype=np.int32).reshape(t_dim)
            # mock_d = np.array([100, 100, 25, 25], dtype=np.int32)

            # Generate pipeline for next step
            new_pipeline = Pipeline(mock_t, mock_d, prev_t_percent=self._observation[4, 0, 0], prev_d_score=prev_d_score, width=t_dim[1], height=t_dim[0])

            # TODO - remove
            # new_pipeline.t_percent = self._observation[4, 0, 0]+.2

            self._pl_hist.add_step(new_pipeline)

            # Update observation vector to send to the next time step
            new_observation_vector = np.zeros((6,t_dim[0],t_dim[1]), dtype=np.int32)
            new_observation_vector[0:4, 0, 0] = [i for i in new_pipeline.d_array]
            new_observation_vector[4, 0, 0] = new_pipeline.t_percent
            new_observation_vector[5] = new_pipeline.t_array
            self._observation = new_observation_vector

            return ts.transition(
                np.array(new_observation_vector, dtype=np.int32), reward=new_pipeline.reward, discount=1.0)

if __name__ == "__main__":
    test_env = RCCarEnv()
    utils.validate_py_environment(test_env, episodes=5)

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
