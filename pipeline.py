
import numpy as np
import helper_funcs as hf
import math
import enum

t_perc_rate = .50
min_reward = -100
max_reward = 100
min_t_percent = .1
max_t_percent = .85
min_d = 20

dir_names = ["fwd", "right", "bwd", "lft"]
opposite_enum = {"fwd": 2,
                 "right": 3,
                 "bwd": 0,
                 "lft": 1}

class Pipeline():
    """Pipeline object for processing data from RC car."""
    def __init__(self, t_array, d_array, width=32, height=24, prev_t_percent=0, prev_d_score=0):
        self.img_width = width
        self.img_height = height
        self.t_array = t_array
        self.d_array = d_array
        self.gps_lat = None
        self.gps_long = None
        self.t_percent = self.get_t_percent()
        self.d_score = self.get_d_score()
        self.reward = self.get_reward(prev_t_percent, prev_d_score)

    def __str__(self):
        return f"Pipeline: \n> distance:{self.d_array}, thermal size: {self.t_array.shape}\n"

    def get_t_percent(self, any_t_array = None):
        if any_t_array:
            array_to_process = any_t_array
        else:
            array_to_process = self.t_array
        t_max = np.amax(array_to_process)
        t_min = np.amin(array_to_process)

        t_threshold = (t_max - t_min)*t_perc_rate + t_min

        t_in_range_cnt = 0
        for element in np.nditer(array_to_process):
            if element > t_max:
                t_max = element
                t_in_range_cnt += 1
            elif element > t_threshold:
                t_in_range_cnt += 1

        t_perc_in_range = t_in_range_cnt / float(self.img_width*self.img_height)
        if any_t_array:
            return t_perc_in_range * 100
            
        self.t_percent = t_perc_in_range * 100
        return self.t_percent

    def get_d_score(self):
        num_bad_elements = 0
        for element in self.d_array:
            if element < min_d:
                num_bad_elements += 1
        
        # Below function gives very bad score of .04 if all 4 directions occupied, if none, returns 1
        score = math.exp(-1.0*num_bad_elements/1.25)
        self.d_score = score
        return score

    def get_reward(self, prev_t_percent, prev_d_score):
        reward = self.t_percent / 100 * self.d_score

        if (self.t_percent / 100) <= min_t_percent:
            if prev_t_percent > self.t_percent:
                # worst case of going out of range of the target
                reward = min_reward
        elif (self.t_percent / 100) >= max_t_percent:
            if prev_t_percent > self.t_percent:
                # best case of coming close enough to the target to send coords
                reward = max_reward
        elif np.amin(self.d_array) < min_d:
            # worst case of crashing into obstacle
            if prev_d_score > self.d_score:
                reward = min_reward

        return reward
    
    def check_if_fatal(self, next_action):
        """Change action to move out of way of obstacle if dangerously close."""
        if np.amin(self.d_array) <= min_d:
            too_close_dir = np.argmin(self.d_array)
            # Dangerously close, reroute manually to dir opposite the closest
            opposite_dir = opposite_enum[dir_names[too_close_dir]]
            if self.d_array[opposite_dir] > 2 * min_d:
                # ok to reroute through opposite dir
                next_action = opposite_dir
            else:
                # Opposite dir not enough room, reroute through dir with greatest headroom
                next_action = np.argmax(self.d_array)

        return next_action

if __name__ == "__main__":

    test_t = np.array( [[27.0, 27.0, 31.0, 31.0],
                    [27.0, 27.0, 32.0, 32.0],
                    [27.0, 27.0, 32.0, 32.0],
                    [27.0, 27.0, 31.0, 31.0]])

    # mock_msg = [float(i) for i in range(775)]
    mock_width = 4
    mock_height = 4
    mock_msg = [float(i) for i in range(mock_width*mock_height + 6)]

    # test case 1: obstacle avoidance, bigger reward
    prev_t_percent = .5
    prev_d_score = .091 # 3 directions
    

    d_array = np.array( [101, 100, 19, 19])
    t_array = np.array([[27.0, 27.0, 31.0, 31.0],
                        [27.0, 27.0, 32.0, 32.0],
                        [27.0, 27.0, 32.0, 32.0],
                        [27.0, 27.0, 31.0, 31.0]])

    mock_pl = Pipeline(t_array, d_array, mock_width, mock_height, prev_t_percent, prev_d_score)

    # Test check_if_fatal function
    dec = mock_pl.check_if_fatal(2)
    print(dec)

    test_percent = mock_pl.get_t_percent()
    test_score = mock_pl.get_d_score()
    test_reward = mock_pl.get_reward(prev_t_percent, prev_d_score)
    print(f"{mock_pl}\nT%: {test_percent}, D score: {test_score}\nREWARD: {test_reward}")

    # test case 2: reverse avoidance, smaller reward
    prev_t_percent = .5
    prev_d_score = .202 # 2 directions

    test_percent = mock_pl.get_t_percent()
    test_score = mock_pl.get_d_score()
    test_reward = mock_pl.get_reward(prev_t_percent, prev_d_score)
    print(f"{mock_pl}\nT%: {test_percent}, D score: {test_score}\nREWARD: {test_reward}")

    new_observation_vector = np.append(mock_pl.d_array, mock_pl.t_percent)
    print(new_observation_vector)
