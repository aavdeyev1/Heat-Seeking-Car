"""The environment setup to run the heat-seeking car in.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pipeline import Pipeline
from pipeline_historian import PipelineHistorian
from connector import Connector

import tensorflow as tf
import numpy as np
import time

from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

TIMOUT_MINS = 15
# t_dim = (24, 32)
t_dim = (12, 16)

reset_observation_vector = np.zeros((6,t_dim[0],t_dim[1]), dtype=np.int32)
reset_observation_vector[:, 0, 0] = [500.0, 500.0, 500.0, 500.0, 0.1, 0]

stop_percent = 25


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

    @property
    def observation_space_size(self):
        return self._observation_space_size

    @property
    def action_spec(self):
        return self._action_spec.maximum

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self._pl_hist.reset()
        self._observation = reset_observation_vector
        return ts.restart(reset_observation_vector)

    def _step(self, next_action):
        # TODO: uncomment  when figure out attribute not found error..
        lat = None

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        if self._observation[4, 0, 0] > stop_percent:
            # End episode if target found, or T% is > stop precent
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
            while lat is None:
                t_array, d_array, lat, long = self.conn.check_for_data()
                time.sleep(1)   
        else:
            raise ValueError('`action` should be 0 to 3.')

        # Prevent the extra step after target finding.
        if self._pl_hist.current_pl.get_t_percent(t_array) > stop_percent:
            # End episode if target found, or T% is > stop precent
            print("Target Found, ending episode...")
            final_step = 4  # stop command
            self.conn.send_turn(final_step)
            # Send Stop command to car
            self._episode_ended = True

        if self._episode_ended or self._observation[4, 0, 0] > stop_percent:
            return ts.termination(reset_observation_vector, reward=100)
        else:
            # Check if fatal action
            next_action = self._pl_hist.current_pl.check_if_fatal(next_action)
            # Valid action returned guaranteed, carry out next turn
            self.conn.send_turn(next_action)

            print(t_array)
            print(d_array)
            print(f"Location: lat={lat}, long={long}")

            # TODO - remove;
            # Convert to usable arrays NUMPY arrs
            # t_array = np.array([float(i) for i in range(t_dim[0]*t_dim[1])],
            #                   dtype=np.int32).reshape(t_dim)
            # d_array = np.array([100, 100, 25, 25], dtype=np.int32)

            # Generate pipeline for next step
            new_pipeline = Pipeline(t_array, d_array, prev_t_percent=self._observation[4, 0, 0],
            prev_d_score=prev_d_score, width=t_dim[1], height=t_dim[0])

            # TODO - remove
            # new_pipeline.t_percent = self._observation[4, 0, 0]+.2

            self._pl_hist.add_step(new_pipeline)

            # Update observation vector to send to the next time step
            new_observation_vector = np.zeros((6,t_dim[0],t_dim[1]), dtype=np.int32)
            new_observation_vector[0:4, 0, 0] = [i for i in new_pipeline.d_array]
            new_observation_vector[4, 0, 0] = new_pipeline.t_percent
            new_observation_vector[5] = new_pipeline.t_array
            self._observation = new_observation_vector
            print(self._observation)

            return ts.transition(
                np.array(new_observation_vector, dtype=np.int32),
                reward=new_pipeline.reward, discount=1.0)

if __name__ == "__main__":
    test_env = RCCarEnv()
    utils.validate_py_environment(test_env, episodes=5)

    tf_env = tf_py_environment.TFPyEnvironment(test_env)
    # reset() creates the initial time_step after resetting the environment.
    time_step = tf_env.reset()
    NUM_STEPS = 4
    transitions = []
    REWARD = 0
    for i in range(NUM_STEPS):
        action = tf.constant(i)
        # applies the action and returns the new TimeStep.
        next_time_step = tf_env.step(action)
        transitions.append([time_step, action, next_time_step])
        REWARD += next_time_step.reward
        time_step = next_time_step
        print(time_step)

    np_transitions = tf.nest.map_structure(lambda x: x.numpy(), transitions)
    print('\n'.join(map(str, np_transitions)))
    print('Total reward:', REWARD.numpy())
