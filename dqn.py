import numpy as np
import argparse
import skimage as skimage
import random
import json
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, Flatten, Dense
from keras.initializations import normal
from keras.optimizers import Adam
from collections import deque
from skimage import transform, color, exposure


# preprocessed image info
img_height = 80
img_width = 80
img_frames = 4 # stack 4 frames to infer the velocity information of the bird
cnn_input_shape = (img_height, img_width, img_frames)

# Hyper Parameters
NUM_ACTION = 2  # bird only has 2 actions: fly, or not
INITIAL_EPSILON = 0.1  # starting value of epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
BATCH_SIZE = 32
GAMMA = 0.99
FRAME_PER_ACTION = 1
OBSERVATION = 2000
EXPLORE = 3000000
REPLAY_MEMORY_SIZE = 50000


class BrainDQN:
    def __init__(self):
        # init replay memory
        self.replayMemory = deque()
        # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        # init Q network
        self.create_q_net()
        self.loss = 0

    def create_q_net(self):
        """
        A five-layer convolutional network with the following architecture:

        (conv - relu) - (conv - relu) - (conv - relu) - fc - softmax

        The network operates on stack of images that have shape (H, W, N)
        consisting of N images, each with height H and width W.
        """
        model = Sequential()

        # first conv layer
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), init=lambda shape, name: normal(shape, scale=0.01, name=name),
                                dim_ordering='tf', border_mode='same', input_shape=cnn_input_shape))
        model.add(Activation('relu'))

        # second conv layer
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), init=lambda shape, name: normal(shape, scale=0.01, name=name),
                                dim_ordering='tf', border_mode='same'))
        model.add(Activation('relu'))

        # third conv layer
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init=lambda shape, name: normal(shape, scale=0.01, name=name),
                                dim_ordering='tf', border_mode='same'))
        model.add(Activation('relu'))

        # full-connected layer
        model.add(Flatten())
        model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
        model.add(Activation('relu'))

        # output layer
        model.add(Dense(2, init=lambda shape, name: normal(shape, scale=0.01, name=name)))

        # compile model
        adam = Adam(lr=1e-6)
        model.compile(optimizer=adam, loss='mean_squared_error')
        return model

    def run_dqn(self, model, args):
        game_state = game.GameState()

        # get the initial state
        curr_state = self.get_init_state(game_state)

        if args['mode'] == 'Run':
            OBSERVE = 999999999  # We keep observe, never train
            epsilon = FINAL_EPSILON
            print("Now we load weight")
            model.load_weights("model_params.h5")
            adam = Adam(lr=1e-6)
            model.compile(loss='mse', optimizer=adam)
            print("Weight load successfully")
        else:  # We go to training mode
            OBSERVE = OBSERVATION
            epsilon = INITIAL_EPSILON

        t = 0
        while True:
            self.loss = 0
            # initialize params for Q-net
            action_index = 0
            curr_reward = 0
            curr_action = np.zeros([NUM_ACTION])
            # epsilon greedy strategy
            if t % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(NUM_ACTION)
                    curr_action[action_index] = 1
                else:
                    predicted_q = model.predict(curr_state)
                    action_index = np.argmax(predicted_q)
                    curr_action[action_index] = 1

            # reduced the epsilon gradually
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            next_state, terminal = self.get_next_state(game_state, curr_state, curr_action)

            # store the transition in replay memory pool
            self.replayMemory.append((curr_state, action_index, curr_reward, next_state, terminal))
            if len(self.replayMemory) > REPLAY_MEMORY_SIZE:
                self.replayMemory.popleft()

            if t > OBSERVE:
                print("now we train the model")
                self.train_q_network(model, curr_state)

            curr_state = next_state
            t += 1

            # save progress every 10000 iterations
            if t % 100 == 0:
                print("Now we save model")
                model.save_weights("model_params.h5", overwrite=True)
                with open("model_params.json", "w") as outfile:
                    json.dump(model.to_json(), outfile)

            # print info
            if t <= OBSERVE:
                state = "observe"
            elif OBSERVE < t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"

            print("TIMESTEP", t, "/ STATE", state,
                  "/ EPSILON", epsilon, "/ ACTION", action_index, "/ Loss ", self.loss)

    def get_next_state(self, game_state, curr_state, curr_action):
        # run the selected action and observed next state and reward
        next_frame, curr_reward, terminal = game_state.frame_step(curr_action)
        next_frame = self.pre_process_frame(next_frame)  # pre-process image
        next_frame = next_frame.reshape(1, next_frame.shape[0], next_frame.shape[1], 1)
        next_state = np.append(next_frame, curr_state[:, :, :, :3], axis=3)
        return next_state, terminal

    def pre_process_frame(self, init_frame):
        """
        pre-process image
        :param init_frame:
        :return:
        """
        # compute luminance of an RGB image
        pre_processed_frame = skimage.color.rgb2gray(init_frame)
        # resize image to match a certain size
        pre_processed_frame = skimage.transform.resize(pre_processed_frame, (80, 80))
        # return image after stretching or shrinking its intensity levels
        pre_processed_frame = skimage.exposure.rescale_intensity(pre_processed_frame, out_range=(0, 255))
        return pre_processed_frame

    def get_init_state(self, game_state):
        """
        1. Set the init action to "not fly"
        2. Get the initial image from wrapped_flappy_bird.py.
        3. Stack the inital image four times to form the inital state.
        :param game_state: game state to communicate with emulator
        :return: init_state = (1, H, W, N) = (1, 80, 80, 4)
        """
        init_action = np.zeros(NUM_ACTION)
        # actions are defined in wrapped_flappy_bird.py
        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        init_action[0] = 1  # do nothing
        init_frame, init_reward, terminal = game_state.frame_step(init_action)
        init_frame = self.pre_process_frame(init_frame)  # pre-process image
        # stack the images to form a state, the init state consists of four same images
        init_state = np.stack((init_frame, init_frame, init_frame, init_frame), axis=0)
        # if dim_ordering='tf', input_shape = (samples, rows, cols, channels)
        init_state = init_state.reshape(1, init_state.shape[1], init_state.shape[2], init_state.shape[0])
        return init_state

    def train_q_network(self, model, curr_state):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)

        inputs = np.zeros((BATCH_SIZE, curr_state.shape[1], curr_state.shape[2], curr_state.shape[3]))  # 32, 80, 80, 4
        estimated_q_batch = np.zeros((BATCH_SIZE, NUM_ACTION))  # 32, 2

        for i in range(0, len(minibatch)):
            curr_state = minibatch[i][0]
            action_index = minibatch[i][1]
            reward = minibatch[i][2]
            next_state = minibatch[i][3]
            terminal = minibatch[i][4]

            inputs[i:i + 1] = curr_state  # saved down s_t

            # Step 2: calculate y or estimated Q_value
            estimated_q_batch[i] = model.predict(curr_state)
            predicted_q_batch = model.predict(next_state)
            if terminal:
                estimated_q_batch[i, action_index] = reward
            else:
                estimated_q_batch[i, action_index] = reward + GAMMA * np.max(predicted_q_batch)

            # targets2 = normalize(targets)
            self.loss += model.train_on_batch(inputs, estimated_q_batch)
            # save network every 100000 iteration


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    brain = BrainDQN()
    model = brain.create_q_net()
    brain.run_dqn(model, args)

if __name__ == "__main__":
    main()
