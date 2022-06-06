# Heat-Seeking Car Reinforcement Learning Algorithm
## EENG Senior Capstone Project for Eastern Washington University

This is the repo for the reinforcement learning algorithm for our Heat-Seeking Car.
The Qtable is in Google Drive under the following link:

[Qtable Files](https://drive.google.com/drive/folders/1IwjKeiST-YtoJIAkEPGNqHGjPYd8h5Nw?usp=sharing)

Download the Qtable and place it in the same directory as the rest of the code.

There are too many dependencies to list here so the best option is trying to run some file, for example, `pipeline.py`, and running `pip install` on the requirements that are missing.

To train the car, run the `train_qtable` script. This will change your local copy of the Qtable.
To test the car, which will also change the local copy of the Qtable, run the `test_qtable` script.

Special thanks to Thomas Simonini's tutorial which I followed for the test/train qtable scripts:

[Q Learning with OpenAI Taxi-v2](https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/Taxi-v2/Q%20Learning%20with%20OpenAI%20Taxi-v2%20video%20version.ipynb)

And the Tensorflow Tutorial which I followed to create the tensorflow environment:

[Introduction to RL and Deep Q Networks](https://www.tensorflow.org/agents/tutorials/0_intro_rl)


### Project Members:
Mattias Tyni: Raspberry Pi code, sensors

Amelya Avdeyev: RL Algorithm

Artyom Gurdyumov: Motors and RC assembly
### Questions about this code? Contact me at aavdeyev1@ewu.edu