from qlearning4k import TreeSearchAgent, randagent, Training
from qlearning4k.games import tictactoe

game = tictactoe.TicTacToe(3, 3, twoplayer=True)
modelgame = tictactoe.TicTacToe(3, 3, twoplayer=True)
baseagent = randagent.RandAgent(9)
randagent2 = randagent.RandAgent(9)
agent = TreeSearchAgent.TreeSearchAgent(baseagent, modelgame, [1,0.5], depth=4)
Training.play_abstracted([agent, randagent2], [1, .5], game, nb_epoch = 50)