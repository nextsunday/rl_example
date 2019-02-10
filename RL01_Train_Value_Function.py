import gym
import random
import numpy as np
import keras
import keras.backend as K
from sklearn.preprocessing import OneHotEncoder
from collections import deque
import matplotlib.pyplot as plt

K.set_image_dim_ordering('th')


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img):
    return img[::2, ::2]  # 2칸씩 건너뛰어 Return (크기 1/2)


def preprocess(img):
    return to_grayscale(downsample(img))


def load_model(name='./atari_keras9999.h'):
    model = keras.models.load_model(name)
    return model


def atari_model(n_actions, ATARI_SHAPE):
    # We assume a theano backend here, so the "channels" are first.
    ATARI_SHAPE = ATARI_SHAPE
    # ATARI_SHAPE = (4, 84, 84) # last 4 frame


    # With the functional API we need to define the inputs.
    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    print(frames_input)
    actions_input = keras.layers.Input((n_actions,), name='mask')
    print(actions_input)

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)

    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = keras.layers.Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(normalized)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = keras.layers.Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv_1)
    # Flattening the second convolutional layer.
    conv_flattened = keras.layers.Flatten()(conv_2)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = keras.layers.Dense(n_actions)(hidden)
    # Finally, we multiply the output by the mask!
    filtered_output = keras.layers.Multiply()([output, actions_input])

    model = keras.models.Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')
    return model


def fit_batch(model, gamma, mini_batch):
    """Do one deep Q learning iteration.

    Params:
    - model: The DQN
    - gamma: Discount factor (should be 0.99)
    - mini_batch: Contains below variables as deque
        - start_states: numpy array of starting states
        - actions: numpy array of one-hot encoded actions corresponding to the start states
        - rewards: numpy array of rewards corresponding to the start states and actions
        - next_states: numpy array of the resulting states corresponding to the start states and actions
        - is_terminal: numpy boolean array of whether the resulting state is terminal

    """
    start_states, next_states, actions, reward, is_terminal = [], [], [], [], []

    for i in range(BATCH_SIZE):
        # mini_batch(start_states, onehot_action, reward, next_states, start_live, is_terminal)
        start_states.append(mini_batch[i][0])
        actions.append(mini_batch[i][1])
        reward.append(mini_batch[i][2])
        next_states.append(mini_batch[i][3])
        is_terminal.append(mini_batch[i][5])

    start_states = np.array(start_states)
    actions = np.array(actions)
    reward = np.array(reward)
    next_states = np.array(next_states)

    # print(start_states.shape, actions.shape, reward.shape, next_states.shape)

    # First, predict the Q values of the next states. Note how we are passing ones as the mask.
    next_Q_values = model.predict([next_states, np.ones(actions.shape)])
    # The Q values of the terminal states is 0 by definition, so override them
    for i, trm in enumerate(is_terminal):
        if trm != 0:
            next_Q_values[i] = 0

    # The Q values of each start state is the reward + gamma * the max next state Q value
    Q_values = reward + gamma * np.max(next_Q_values, axis=1)
    # Fit the keras model. Note how we are passing the actions as the mask and multiplying
    # the targets by the actions.
    history = model.fit(
        [start_states, actions], actions * Q_values[:, None],
        epochs=1, batch_size=BATCH_SIZE, verbose=0
    )
    return history



# Create a breakout environment
# env = gym.make('BreakoutDeterministic-v4')   # 고정형
env = gym.make('Breakout-v0')  # 랜덤형
ATARI_SHAPE = (4, 105, 80)
# model = atari_model(n_actions=4, ATARI_SHAPE=ATARI_SHAPE)
model = load_model('./atari_keras128000.h5')

e = START_EXPLORATION = 0.5
N_ITERATION = 200000
MEMORY_SIZE = 100000
BATCH_SIZE = 32
FINAL_EXPLORATION = 0.1
TRAIN_START = 50000
EXPLORATION = 10000


gamma = 0.99
one_encoder = OneHotEncoder(handle_unknown='ignore')

X = np.array([[0], [1], [2], [3]])
one_encoder.fit(X)

# Memory
replay_memory = deque(maxlen=MEMORY_SIZE)

for i in range(N_ITERATION):

    # Reset it, returns the starting frame
    start_states = env.reset()
    # Render
    # env.render()

    start_states = preprocess(start_states)
    # states = np.zeros(ATARI_SHAPE, dtype=np.float32)
    start_states = np.stack((start_states, start_states, start_states, start_states), axis=0)

    rewards = 0
    is_terminal = 0
    max_steps = 0
    max_reward = 0
    episode_reward = 0
    is_random = 0
    is_done = False
    start_live = 5
    steps = 0

    while not is_done:

        # Perform a random action, returns the new frame, reward and whether the game is over
        if np.random.rand(1) < e:
            is_random += 1   # Count # of random action

            action = env.action_space.sample()

            onehot_action = one_encoder.transform(np.array([[action]])).toarray()
            onehot_action = onehot_action.reshape(4)
        else:

            if is_terminal == 0:  # Initial Step
                action = model.predict([np.expand_dims(start_states, axis=0), np.ones((1, 4))])
            else:
                start_states = next_states
                action = model.predict([np.expand_dims(start_states, axis=0), np.ones((1, 4))])

            action = np.argmax(action, axis=1)
            onehot_action = one_encoder.transform(action.reshape(1, 1)).toarray()
            onehot_action = onehot_action.reshape(4)
            action = int(action)

        next_states, reward, is_done, info = env.step(action)
        next_states = preprocess(next_states)
        next_states = np.append(np.expand_dims(next_states, axis=0), start_states[:3, :, :], axis=0)

        # Life Check, If die, negative reward & terminal True
        if start_live > info['ale.lives']:
            start_live = info['ale.lives']

            # save in replay_memory
            reward = -1
            replay_memory.append((start_states, onehot_action, reward, next_states, start_live, is_terminal))
            if max_steps < is_terminal:
                max_steps = is_terminal
            is_terminal = 0

            if rewards > 0:
                print("Life: {}, Life Reward: {}".format(start_live, rewards))
            if max_reward < rewards:
                max_reward = rewards
            episode_reward += rewards
            rewards = 0

        else:
            reward = np.clip(reward, -1., 1.)
            replay_memory.append((start_states, onehot_action, reward, next_states, start_live, 0))

            is_terminal += 1
            rewards += reward
        steps += 1

        if len(replay_memory) >= TRAIN_START:  # Train Start
            mini_batch = random.sample(replay_memory, BATCH_SIZE)
            history = fit_batch(model=model, gamma=gamma, mini_batch=mini_batch)
            if steps % 100 == 0:
                print("Training......", steps, "Loss: ", history.history['loss'])

    if is_done is True:
        # E-greedy algorithm
        if e > FINAL_EXPLORATION and len(replay_memory) >= TRAIN_START:
            e -= (START_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION


    print(i, "Episode Reward: {0:.2f}".format(episode_reward), "Memory Size: {}".format(len(replay_memory)))
    print("Max Steps: {}".format(max_steps), "Max Reward: {}".format(max_reward))
    print("Random: ", is_random,  "e: {0:.2f}".format(e))
    print("Model: ", steps - is_random)

    if i % 1000 == 0:
        continue
        # To-do 특정 타이밍에서 Network Copy

    if i % 500 == 0 and i > 0:
        print("saving Model")
        model.save('./atari_keras' + str(i) + '.h5')


print("saving Model")
model.save('./atari_keras' + str(i) + '.h5')


