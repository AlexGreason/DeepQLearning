from .memory import ExperienceReplay
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import pygame
import time
import copy


class Agent:
    def __init__(self, model, memory=None, memory_size=1000, nb_frames=None):
        assert len(model.output_shape) == 2, "Model's output shape should be (nb_samples, nb_actions)."
        if memory:
            self.memory = memory
        else:
            self.memory = ExperienceReplay(memory_size)
        if not nb_frames and not model.input_shape[1]:
            raise Exception("Missing argument : nb_frames not provided")
        elif not nb_frames:
            nb_frames = model.input_shape[1]
        elif model.input_shape[1] and nb_frames and model.input_shape[1] != nb_frames:
            raise Exception("Dimension mismatch : time dimension of model should be equal to nb_frames.")
        self.model = model
        self.nb_frames = nb_frames
        self.frames = None
        self.nb_actions = model.output_shape[1]

    @property
    def memory_size(self):
        return self.memory.memory_size

    @memory_size.setter
    def memory_size(self, value):
        self.memory.memory_size = value

    def reset_memory(self):
        self.exp_replay.reset_memory()

    def update(self, batch_size, gamma):
        batch = self.memory.get_batch(model=self.model, batch_size=batch_size, gamma=gamma)
        loss = 0
        if batch:
            inputs, targets = batch
            loss =self.model.train_on_batch(inputs, targets)
        return loss

    def get_move(self, state, epsilon, nb_actions, return_cause=False):
        random = 0
        if np.random.random() < epsilon:
            a = int(np.random.randint(nb_actions))
            random=1
        else:
            q = self.model.predict(state)
            a = int(np.argmax(q[0]))
        if return_cause:
            return a, random
        else:
            return a

    def check_game_compatibility(self, game):
        game_output_shape = (1, None) + game.get_frame().shape
        if len(game_output_shape) != len(self.model.input_shape):
            raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
        else:
            for i in range(len(self.model.input_shape)):
                if self.model.input_shape[i] and game_output_shape[i] and self.model.input_shape[i] != \
                        game_output_shape[i]:
                    raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
        if len(self.model.output_shape) != 2 or self.model.output_shape[1] != game.nb_actions:
            raise Exception('Output shape of model should be (nb_samples, nb_actions).')

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

    def train(self, game, nb_epoch=1000, batch_size=50, gamma=0.9, epsilon=[1., .1], epsilon_rate=0.5,
              reset_memory=False):
        self.check_game_compatibility(game)
        if type(epsilon) in {tuple, list}:
            delta = ((epsilon[0] - epsilon[1]) / (nb_epoch * epsilon_rate))
            final_epsilon = epsilon[1]
            epsilon = epsilon[0]
        else:
            final_epsilon = epsilon
        model = self.model
        nb_actions = model.output_shape[-1]
        win_count = 0
        sumscore = 0
        for epoch in range(nb_epoch):
            loss = 0.
            game.reset()
            self.clear_frames()
            if reset_memory:
                self.reset_memory()
            game_over = False
            S = self.get_game_data(game)
            while not game_over:
                if np.random.random() < epsilon:
                    a = int(np.random.randint(game.nb_actions))
                else:
                    q = model.predict(S)
                    a = int(np.argmax(q[0]))
                game.play(a)
                r = game.get_score()
                S_prime = self.get_game_data(game)
                game_over = game.is_over()
                transition = [S, a, r, S_prime, game_over]
                self.memory.remember(*transition)
                S = S_prime
                batch = self.memory.get_batch(model=model, batch_size=batch_size, gamma=gamma)
                if batch:
                    inputs, targets = batch
                    loss += float(model.train_on_batch(inputs, targets))
            score = game.get_score(1)
            sumscore += score
            if game.is_won():
                win_count += 1
            if epsilon > final_epsilon:
                epsilon -= delta
            print("Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.2f} | Win count {} | Win rate {:.3f} | Score {} | Average Score {:.4f}".format(epoch + 1, nb_epoch, loss,
                                                                                             epsilon, win_count, win_count/(epoch+1), score, sumscore/(epoch+1)))

    def play(self, game, nb_epoch=10, epsilon=0., visualize=True):
        self.check_game_compatibility(game)
        model = self.model
        win_count = 0
        frames = []
        for epoch in range(nb_epoch):
            game.reset()
            self.clear_frames()
            S = self.get_game_data(game)
            if visualize:
                frames.append(copy.deepcopy(game.draw()))
            game_over = False
            while not game_over:
                if np.random.rand() < epsilon:
                    print("random")
                    action = int(np.random.randint(0, game.nb_actions))
                else:
                    q = model.predict(S)
                    action = int(np.argmax(q[0]))
                game.play(action)
                S = self.get_game_data(game)
                if visualize:
                    frames.append(copy.deepcopy(game.draw()))
                game_over = game.is_over()
            if game.is_won():
                win_count += 1
            score = game.get_score()
            print("Epoch {:03d}/{:03d}| Epsilon {:.2f} | Win count {} | Win rate {:.3f} | Score {}".format(int(epoch + 1),
                                                                                                           int(nb_epoch), epsilon, win_count, win_count/(epoch+1), score))
        print("Accuracy {} %".format(100. * win_count / nb_epoch))
        if visualize:
            #if 'images' not in os.listdir('.'):
                #os.mkdir('images')
            pygame.init()
            print(frames[0].shape)
            screen = pygame.display.set_mode((600, 600))
            screen.fill((0, 0, 0))
            for i in range(len(frames)):
                base = frames[i] * 128
                base = base[..., None].repeat(3, -1).astype("uint8")

                surface = pygame.surfarray.make_surface(base)
                newscreen = pygame.transform.scale(surface, (600, 600))
                screen.blit(newscreen, (0,0))
                pygame.display.flip()
                time.sleep(.3)