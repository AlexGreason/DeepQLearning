__author__ = "Exa"

import numpy as np
from qlearning4k.games.game import Game
import copy


class TicTacToe(Game):
    def __init__(self, gridx, gridy, values=(1,0,-1), twoplayer=True):
        self.won = False
        self.over = False
        self.gridx = gridx
        self.gridy = gridy
        self.values = values
        self.twoplayer=twoplayer
        self.reset()

    def reset(self):
        self.state = np.zeros((self.gridx, self.gridy))
        self.won = False
        self.over = False

    @property
    def name(self):
        return "TicTacToe"

    @property
    def nb_actions(self):
        return self.gridx*self.gridy
    #@profile
    def haswon(self, player):
        state = self.state
        bitmap = state == player
        if max(np.bitwise_and.reduce(bitmap, axis=0)):
            return True
        if max(np.bitwise_and.reduce(bitmap, axis=1)):
            return True
        if self.gridx == self.gridy:
            if all([state[i,i] == player for i in range(self.gridx)]):
                return True
            if all([state[(self.gridx-1)-i, i] == player for i in range(self.gridx)]):
                return True
        return False
    #@profile
    def play(self, action, player=1):
        state = self.state
        if state[action // self.gridy, action % self.gridy] == 0:
            state[action // self.gridy, action % self.gridy] = player
        if not self.twoplayer:
            if np.count_nonzero(state) != self.gridx*self.gridy:
                moved = False
                while not moved:
                    location = (np.random.randint(0, self.gridx), np.random.randint(0, self.gridy))
                    if state[location] == 0:
                        moved = True
                        state[location] = .5
        self.state = state


    def get_state(self, player=1):
        if player == 1:
            return self.state
        else:
            state = copy.deepcopy(self.state)
            state[state == 1] = 2
            state[state == .5] = 1
            state[state == 2] = .5
            return state


    def get_score(self, player=1):
        if self.haswon(player):
            self.won = player == 1
            self.over=True
            return self.values[0]
        if self.haswon([1, .5][int(player*2-1)]):
            self.won = player != 1
            self.over = True
            return self.values[2]
        else:
            return self.values[1]
    #@profile
    def is_over(self):
        if self.haswon(1):
            self.over = True
            self.won = True
            return True
        if self.haswon(.5):
            self.over = True
            return True
        if np.count_nonzero(self.state) == self.gridx * self.gridy:
            self.over = True
            return True
        return False

    def is_won(self, player=1):
        if player==1:
            return self.won
        else:
            return not self.won

    def changestate(self, player1, player2):
        newstate = copy.copy(self.state)
        newstate[newstate == player1] = 2
        newstate[newstate == player2] = player1
        newstate[newstate == 2] = player2
        return newstate

    def islegal(self, player, action):
        return self.state[action // self.gridy, action % self.gridy] == 0
