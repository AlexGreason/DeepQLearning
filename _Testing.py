from keras.layers import Flatten, Convolution2D, Reshape, UpSampling2D, MaxPooling2D, Dense
from keras.models import Sequential, model_from_json
from keras.optimizers import *

from qlearning4k.Training import *
from qlearning4k.games.tictactoe import TicTacToe
from qlearning4k.nnAgent import Agent
from qlearning4k.randagent import RandAgent
from qlearning4k.TreeSearchAgent import TreeSearchAgent

gridx, gridy = 6, 6
hidden_size = 512
nb_frames = 1
trainable=True
model = Sequential()
model.add(Flatten(input_shape=(1, gridx, gridy)))
model.add(Reshape((1, gridx, gridy)))
model.add(UpSampling2D(size=(4, 4)))
model.add(Convolution2D(64, 3, 3, border_mode="same", activation="relu", trainable=trainable))
model.add(Convolution2D(64, 3, 3, border_mode="same", activation="relu", trainable=trainable))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode="same", activation="relu", trainable=trainable))
model.add(Convolution2D(64, 3, 3, border_mode="same", activation="relu", trainable=trainable))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode="same", activation="relu", trainable=trainable))
model.add(Convolution2D(64, 3, 3, border_mode="same", activation="tanh", trainable=trainable))
model.add(Convolution2D(1, 3, 3, border_mode="same", trainable=trainable))
model.add(Flatten())
model.add(Reshape((gridx*gridy,)))

load_net = False
load_weights = True
save_net = True
save_weights = True

t3_1 = TicTacToe(gridx, gridy, values = (1,0,-1), twoplayer=False)
t3_2 = TicTacToe(gridx, gridy, values = (1,0,-1), twoplayer=True)
t3_3 = TicTacToe(gridx, gridy, values = (1,0,-1), twoplayer=True)
if load_net:
    model = model_from_json(open('t3-6-6-conv-4.json').read())
if load_weights:
    model.load_weights("'t3-6-6-conv-4.sav")


model.compile(sgd(lr=0.01), loss="mse")
agent = Agent(model=model)
rand = RandAgent(gridx*gridy)
treeagent = TreeSearchAgent(agent, t3_3, [1, .5], depth=1)
#while True:
#agent.train(t3_1, nb_epoch = 5000, epsilon = .1, batch_size=32)
#cycle_train(agent, [rand], [1, .5], t3_2, epsilons=[0.1, 0.1], nb_epoch=1000, batch_size=32)
if save_net:
    json_string = model.to_json()
    open('t3-6-6-linear-1.json', 'w').write(json_string)
if save_weights:
    model.save_weights("t3-6-6-linear-1.sav", overwrite=True)
#play_abstracted([agent, rand], [1, .5], t3_2, epsilons=[.1, .1], nb_epoch=100, visualize=False)
play_abstracted([treeagent, rand], [1, .5], t3_2, epsilons=[.1, .1], nb_epoch=1000, visualize=False)
