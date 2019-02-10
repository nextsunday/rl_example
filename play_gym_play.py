import gym
import time
import numpy as np
import keras
import keras.backend as K
import random
K.set_image_dim_ordering('th')


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    return to_grayscale(downsample(img))


def load_model(name='./atari_keras9999.h'):
    model = keras.models.load_model(name)
    return model


def atari_model(n_actions):
    # We assume a theano backend here, so the "channels" are first.
    ATARI_SHAPE = (4, 105, 80)

    # With the functional API we need to define the inputs.
    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    print(frames_input)
    actions_input = keras.layers.Input((n_actions,), name='mask')
    print(actions_input)

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)

    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input
    #  image and applies a rectifier nonlinearity."
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


# Create a breakout environment
# env = gym.make('BreakoutNoFrameskip-v4')
# env = gym.make('BreakoutDeterministic-v4')
env = gym.make('Breakout-v0')

# model = atari_model(n_actions=4)
model = load_model('./atari_keras12500.h5')
PLAY_TIME = 100

for i in range(PLAY_TIME):

    # Reset it, returns the starting frame
    frame = env.reset()
    frame = preprocess(frame)
    next_frame = np.stack((frame, frame, frame, frame), axis=0)
    # Render
    env.render()

    is_done = False
    is_start = True
    start_live = 5
    while not is_done:
        # Perform a random action, returns the new frame, reward and whether the game is over
        action = model.predict([np.expand_dims(next_frame, axis=0), np.ones((1, 4))])

        # action = model.predict([start_frame.reshape(1, 4, 105, 80), np.ones((1, 4))])
        action = np.argmax(action, axis=1)
        action = int(action)

        if is_start is True:
            for _ in range(random.randint(1, 4)):  # 가끔식 발사 안하고 멈추는 경우가 있어서 추가
                frame, reward, is_done, info = env.step(1)
        else:
            frame, reward, is_done, info = env.step(action)
        frame = preprocess(frame)
        next_frame = np.append(np.expand_dims(frame, axis=0), next_frame[:3, :, :], axis=0)
        if start_live > info['ale.lives']:
            start_live = info['ale.lives']
            is_start = True
        else:
            is_start = False
        env.render()
        time.sleep(0.001)


