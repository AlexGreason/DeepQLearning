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

class TreeSearchAgent:
    def __init__(self, baseagent, modelgame, playerids, depth=1):
        self.memory = baseagent.memory
        self.model = baseagent.model
        self.nb_actions = baseagent.nb_actions
        self.baseagent = baseagent
        self.nb_frames = baseagent.nb_frames
        self.depth = depth
        self.modelgame = modelgame
        self.playerids = playerids


    def update(self, batch_size, gamma):
        # Updating is possible for tree search agents, but not currently implemented
        return 0

    #@profile
    def evaluate(self, state, playerid):
        if playerid == 0:
            newstate = state.reshape(tuple([1,1] + list(state.shape)))
            q = self.model.predict(newstate)
            return np.max(q)
        else:
            #newstate = self.modelgame.changestate(self.playerids[0], self.playerids[playerid])
            newstate = state.reshape(tuple([1, 1] + list(state.shape)))
            q = self.model.predict(newstate)
            return -np.max(q)
    #@profile
    def treeEvaluate(self, state, nb_actions, depth, playerid):
        if self.modelgame.is_over():
            return self.modelgame.get_score(player=self.playerids[int(playerid)])
        if depth == 0:
            return self.evaluate(state, playerid)
        if depth == 1:
            states = []
            for move in range(nb_actions):
                self.modelgame.reset()
                self.modelgame.state = copy.copy(state)
                self.modelgame.play(move, player=self.playerids[int(playerid)])
                states.append(self.modelgame.state)
            states = np.array(states)
            qs = self.model.predict(states)
            evals = ((int(not playerid))*2-1)*np.amax(qs, axis=(1, 2))
            eval = ((int(not playerid))*2-1)*np.max(evals)
            return eval
        best_move = 0
        best_score = -float('inf')
        for move in range(nb_actions):
            self.modelgame.reset()
            self.modelgame.state = copy.copy(state)
            self.modelgame.play(move, player=self.playerids[int(playerid)])
            score = -self.treeEvaluate(self.modelgame.state, nb_actions, depth-1, not playerid)
            if score > best_score:
                best_move = move
                best_score = score
        return best_score
    #@profile
    def minimax(self, state, nb_actions, depth, playerid):
        if depth == 0:
            return np.argmax(self.model.predict(state.reshape(tuple([1,1] + list(state.shape)))))
        move = 0
        if depth == 1:
            states = []
            endedstates = []
            for move in range(nb_actions):
                self.modelgame.reset()
                self.modelgame.state = copy.copy(state)
                self.modelgame.play(move, player=self.playerids[int(playerid)])
                states.append(self.modelgame.state.reshape([1]+list(self.modelgame.state.shape)))
                if self.modelgame.is_over():
                    endedstates.append((move, self.modelgame.get_score(player=self.playerids[int(playerid)])))
            states = np.array(states)
            qs = self.model.predict(states)
            evals = ((int(not playerid)) * 2 - 1) * np.amax(qs, axis=1)
            for i, x in endedstates:
                evals[i] = x
            move = np.argmax(evals)
            return move
        best_move = 0
        best_score = float('-inf')
        for move in range(nb_actions):
            self.modelgame.reset()
            self.modelgame.state = copy.copy(state)
            self.modelgame.play(move, player=self.playerids[int(playerid)])
            score = -self.treeEvaluate(self.modelgame.state, nb_actions, depth-1, not playerid)
            if score > best_score:
              best_move = move
              best_score = score
        return best_move
    #@profile
    def get_move(self, state, epsilon, nb_actions, return_cause=False):
        random = 0
        if np.random.random() < epsilon:
            a = int(np.random.randint(nb_actions))
            random = 1
        else:
            a = self.minimax(state[0][0], nb_actions, self.depth, 0)
        if return_cause:
            return a, random
        else:
            return a

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