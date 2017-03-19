from qlearning4k.memory import Memory
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import pygame
import time
import copy

class DummyModel:
    def __init__(self):
        pass

    def train_on_batch(self, inputs, targets):
        return 0

    def predict(self, state):
        return np.random.uniform(-.1, .1, state.shape)

class RandAgent:
    def __init__(self, nb_actions):
        self.memory = Memory()
        self.model = DummyModel()
        self.nb_actions = nb_actions
        self.nb_frames = 1


    def update(self, batch_size, gamma):
        return 0

    def get_move(self, state, epsilon, nb_actions, return_cause=False):
        if return_cause:
            return int(np.random.randint(nb_actions)), 0
        else:
            return int(np.random.randint(nb_actions))

    def check_game_compatibility(self, game):
        game_output_shape = (1, None) + game.get_frame().shape
        if self.nb_actions != game.nb_actions:
            raise Exception('action number mismatch')

    def get_game_data(self, game, player=None):
        if player is None:
            frame = game.get_frame()
        else:
            frame = game.get_frame(player)
        if self.frames is None:
            self.frames = [frame] * self.nb_frames
        else:
            self.frames.append(frame)
            self.frames.pop(0)
        return np.expand_dims(self.frames, 0)

    def clear_frames(self):
        self.frames = None