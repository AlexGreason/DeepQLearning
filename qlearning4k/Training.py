from .memory import ExperienceReplay
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import pygame
import time
import copy



def train_nplayer(players, playerids, game, nb_epoch=1000, batch_size=50, gamma=0.9, epsilon=[1., .1], epsilon_rate=0.5,
          reset_memory=False):
    for player in players:
        player.check_game_compatibility(game)
    if type(epsilon) in {tuple, list}:
        delta = ((epsilon[0] - epsilon[1]) / (nb_epoch * epsilon_rate))
        final_epsilon = epsilon[1]
        epsilon = epsilon[0]
    else:
        final_epsilon = epsilon
    models = [player.model for player in players]
    nb_actions = models[0].output_shape[-1]
    win_counts = [0 for player in players]
    draw_count = 0
    sumlosses = [0 for player in players]
    for epoch in range(nb_epoch):
        losses = [0 for player in players]
        game.reset()
        for player in players:
            player.clear_frames()
        if reset_memory:
            for player in players:
                player.reset_memory()
        game_over = False
        states = [player.get_game_data(game, playerid) for player,playerid in zip(players, playerids)]
        while not game_over:
            for i, (player, playerid) in enumerate(zip(players, playerids)):
                if np.random.random() < epsilon:
                    a = int(np.random.randint(game.nb_actions))
                else:
                    q = player.model.predict(states[i])
                    a = int(np.argmax(q[0]))
                game.play(a, player=playerid, twoplayer=True)
                r = game.get_score(playerid)
                S_prime = player.get_game_data(game, playerid)
                transition = [states[i], a, r, S_prime, game_over]
                player.memory.remember(*transition)
                states[i] = S_prime
                batch = player.memory.get_batch(model=player.model, batch_size=batch_size, gamma=gamma)
                if batch:
                    inputs, targets = batch
                    losses[i] += float(player.model.train_on_batch(inputs, targets))
            game_over = game.is_over()
        drawn = True
        for i, playerid in enumerate(playerids):
            if game.is_won(playerid):
                win_counts[i] += 1
                drawn = False
        if drawn:
            draw_count += 1
        if epsilon > final_epsilon:
            epsilon -= delta
        sumlosses = [sum(x) for x in zip(sumlosses, losses)]
        print("Epoch {:03d}/{:03d} | Loss {:.4f}/{:.4f} | Average Loss {:.4f}/{:.4f} | Epsilon {:.2f} | Win count {}/{}".format(epoch + 1,
               nb_epoch, losses[0], losses[1], sumlosses[0] / (epoch + 1), sumlosses[1] / (epoch + 1), epsilon, win_counts[0], win_counts[1]))


def play_2player(self, game, nb_epoch=10, epsilon=0., visualize=True):
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
        print("Epoch {:03d}/{:03d}| Epsilon {:.2f} | Win count {}".format(int(epoch + 1), int(nb_epoch), epsilon,
                                                                          win_count))
    print("Accuracy {} %".format(100. * win_count / nb_epoch))
    if visualize:
        # if 'images' not in os.listdir('.'):
        # os.mkdir('images')
        pygame.init()
        print(frames[0].shape)
        screen = pygame.display.set_mode((frames[0].shape[0] * 200, frames[0].shape[1] * 200))
        screen.fill((0, 0, 0))
        for i in range(len(frames)):
            plt.imshow(frames[i], interpolation='none')
            if i == 0:
                cbar = plt.colorbar()
            else:
                cbar.set_clim(frames[i].min(), frames[i].max())
            plt.savefig("images/" + game.name + ".png")
            myimage = pygame.image.load("images/" + game.name + ".png")
            newimage = pygame.transform.scale(myimage, (frames[0].shape[0] * 200, frames[0].shape[1] * 200))
            imagerect = newimage.get_rect()
            # screen.fill(0,0,0)
            screen.blit(newimage, imagerect)
            pygame.display.flip()
            time.sleep(.25)

def train_abstracted(players, playerids, game, nb_epoch=1000, batch_size=50, gamma=0.9, epsilons=[.1, .1]):
    for player in players:
        player.check_game_compatibility(game)
    win_counts = [0 for player in players]
    scores = [0 for player in players]
    sumscores = [0 for player in players]
    losses = [0 for player in players]
    sumlosses = [0 for player in players]
    drawcount = 0
    starttime = time.time()
    tempepsilons = copy.copy(epsilons)
    for epoch in range(nb_epoch):
        newtime = time.time()
        game.reset()
        for player in players:
            player.clear_frames()
        game_over = False
        states = [player.get_game_data(game, playerid) for player, playerid in zip(players, playerids)]
        scores = [0 for player in players]
        losses = [0 for player in players]
        while not game_over:
            initialstate = players[0].get_game_data(game, playerids[0])
            for i, (player, playerid) in enumerate(zip(players, playerids)):
                a = player.get_move(states[i], epsilons[i], game.nb_actions)
                r = game.get_score(playerid)
                game.play(a, playerid)
                S_prime = player.get_game_data(game, playerid)
                game_over = game.is_over()
                transition = [states[i], a, r, S_prime, game_over]
                player.memory.remember(*transition)
                states[i] = S_prime
                batch = player.memory.get_batch(model=player.model, batch_size=batch_size, gamma=gamma)
                loss = 0
                if batch:
                    inputs, targets = batch
                    loss = float(player.model.train_on_batch(inputs, targets))
                losses[i] += loss
                sumlosses[i] += loss
                if game_over:
                    break
            newstate = players[0].get_game_data(game, playerids[0])
            if (newstate == initialstate).all():
                epsilons = [.2, .2]
            else:
                epsilons = copy.copy(tempepsilons)
            for i, (player, playerid) in enumerate(zip(players, playerids)):
                scores[i] += game.get_score(playerid)
                sumscores[i] += game.get_score(playerid)
        drawn = True
        for i, playerid in enumerate(playerids):
            if game.haswon(playerid):
                win_counts[i] += 1
                drawn = False
        drawcount += int(drawn)
        print("Epoch {:03d}/{:03d} | Epsilons {:.2f}/{:.2f} | Win counts {}/{} | Win rates {:.3f}/{:.3f} | Scores {}/{} | Average Scores {:.4f}/{:.4f} | Losses {:.5f}/{:.5f} | Average Losses {:.5f}/{:.5f} | Epoch Time {:.3f} | Avgtime {:.4f}".format(
                epoch + 1, nb_epoch,
                epsilons[0], epsilons[1], win_counts[0], win_counts[1], win_counts[0] / (epoch + 1), win_counts[1] / (epoch + 1), scores[0], scores[1], sumscores[0] / (epoch + 1), sumscores[1] / (epoch + 1),
                losses[0], losses[1], sumlosses[0]/(epoch+1), sumlosses[1]/(epoch+1), (time.time()-newtime), (time.time()-starttime)/(epoch+1)))

def play_abstracted(players, playerids, game, nb_epoch=1000, epsilons=[.1, .1], visualize = True):
    for player in players:
        player.check_game_compatibility(game)
    win_counts = [0 for player in players]
    scores = [0 for player in players]
    sumscores = [0 for player in players]
    drawcount = 0
    frames = []
    starttime = time.time()
    for epoch in range(nb_epoch):
        newtime = time.time()
        game.reset()
        for player in players:
            player.clear_frames()
        game_over = False
        states = [player.get_game_data(game, playerid) for player, playerid in zip(players, playerids)]
        if visualize:
            frames.append((copy.deepcopy(game.draw()), 0))
        scores = [0 for player in players]
        while not game_over:
            for i, (player, playerid) in enumerate(zip(players, playerids)):
                state = players[0].get_game_data(game, playerids[0])
                states[i] = player.get_game_data(game, player=playerid)
                a, random = player.get_move(states[i], epsilons[i], game.nb_actions, return_cause=True)
                game.play(a, playerid)
                newstate = players[0].get_game_data(game, playerids[0])
                if visualize and (state != newstate).any():
                    frames.append((copy.deepcopy(game.draw()), random))
                game_over = game.is_over()
                states[i] = player.get_game_data(game, playerid)
                if game_over:
                    break
        for i, (player, playerid) in enumerate(zip(players, playerids)):
            scores[i] += game.get_score(playerid)
            sumscores[i] += game.get_score(playerid)
        drawn = True
        for i, playerid in enumerate(playerids):
            if game.haswon(playerid):
                win_counts[i] += 1
                drawn = False
        drawcount += int(drawn)
        print(
            "Epoch {:03d}/{:03d} | Epsilons {:.2f}/{:.2f} | Win counts {}/{} | Win rates {:.3f}/{:.3f} | Scores {}/{} | Average Scores {:.4f}/{:.4f} | Epoch Time {:.3f} | Avgtime {:.4f}".format(
                epoch + 1, nb_epoch,
                epsilons[0], epsilons[1], win_counts[0], win_counts[1], win_counts[0] / (epoch + 1), win_counts[1] / (epoch + 1), scores[0], scores[1], sumscores[0] / (epoch + 1), sumscores[1] / (epoch + 1), (time.time()-newtime), (time.time()-starttime)/(epoch+1)))
    if visualize:
        # if 'images' not in os.listdir('.'):
        # os.mkdir('images')
        pygame.init()
        print(frames[0][0].shape)
        screen = pygame.display.set_mode((600, 800))
        screen.fill((0, 0, 0))
        for i in range(len(frames)):
            if (frames[i][0] != frames[abs(max(0, i-1))][0]).any():
                base = frames[i][0] * 128
                base = base[..., None].repeat(3, -1).astype("uint8")
                cause = np.array([[frames[i][1]]])*128
                cause = cause[..., None].repeat(3, -1).astype("uint8")
                surface = pygame.surfarray.make_surface(base)
                newscreen = pygame.transform.scale(surface, (600, 600))
                causesurf = pygame.surfarray.make_surface(cause)
                newcause = pygame.transform.scale(causesurf, (600, 200))
                screen.blit(newscreen, (0, 0))
                screen.blit(newcause, (0, 600))
                pygame.display.flip()
                time.sleep(.3)
                
def cycle_train(player, opponents, playerids, game, nb_epoch=1000, batch_size=50, gamma=0.9, epsilons=[.1, .1]):
    players = [player]+opponents
    for player in players:
        player.check_game_compatibility(game)
    win_counts = [0 for player in players]
    scores = [0 for player in players]
    sumscores = [0 for player in players]
    losses = [0 for player in players]
    sumlosses = [0 for player in players]
    drawcount = 0
    starttime = time.time()
    tempepsilons = copy.copy(epsilons)
    for epoch in range(nb_epoch):
        newtime = time.time()
        game.reset()
        for player in players:
            player.clear_frames()
        game_over = False
        states = [player.get_game_data(game, playerid) for player, playerid in zip(players, playerids)]
        scores = [0 for player in players]
        losses = [0 for player in players]
        currplayers = [player] + [opponents[epoch%len(opponents)]]
        while not game_over:
            initialstate = currplayers[0].get_game_data(game, playerids[0])
            for i, (player, playerid) in enumerate(zip(currplayers, playerids)):
                a = player.get_move(states[i], epsilons[i], game.nb_actions)
                r = game.get_score(playerid)
                game.play(a, playerid)
                S_prime = player.get_game_data(game, playerid)
                game_over = game.is_over()
                transition = [states[i], a, r, S_prime, game_over]
                player.memory.remember(*transition)
                states[i] = S_prime
                batch = player.memory.get_batch(model=player.model, batch_size=batch_size, gamma=gamma)
                loss = 0
                if batch:
                    inputs, targets = batch
                    loss = float(player.model.train_on_batch(inputs, targets))
                losses[i] += loss
                sumlosses[i] += loss
            newstate = currplayers[0].get_game_data(game, playerids[0])
            if (newstate == initialstate).all():
                epsilons = [.2, .2]
            else:
                epsilons = copy.copy(tempepsilons)
            for i, (player, playerid) in enumerate(zip(players, playerids)):
                scores[i] += game.get_score(playerid)
                sumscores[i] += game.get_score(playerid)
        drawn = True
        for i, playerid in enumerate(playerids):
            if game.haswon(playerid):
                win_counts[i] += 1
                drawn = False
        drawcount += int(drawn)
        print("Epoch {:03d}/{:03d} | Epsilons {:.2f}/{:.2f} | Win counts {}/{} | Win rates {:.3f}/{:.3f} | Scores {}/{} | Average Scores {:.4f}/{:.4f} | Losses {:.5f}/{:.5f} | Average Losses {:.5f}/{:.5f} | Epoch Time {:.3f} | Avgtime {:.4f}".format(
                epoch + 1, nb_epoch,
                epsilons[0], epsilons[1], win_counts[0], win_counts[1], win_counts[0] / (epoch + 1), win_counts[1] / (epoch + 1), scores[0], scores[1], sumscores[0] / (epoch + 1), sumscores[1] / (epoch + 1),
                losses[0], losses[1], sumlosses[0]/(epoch+1), sumlosses[1]/(epoch+1), (time.time()-newtime), (time.time()-starttime)/(epoch+1)))