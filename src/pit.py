from src.arena import Arena
from src.net import NeuralNet
import src.MCTS as MCTS
from src.game import Game
from src.players import *

import numpy as np

human_vs_bot = True

game = Game(8)

random_player = RandomPlayer('Random')
greedy_player = GreedyPlayer('Greedy')
human_player = HumanPlayer('Human')

net1 = NeuralNet(game)
net1.load_checkpoint('./','6x6_100checkpoints_best.pth.tar')

mcts1 = MCTS(game, net1)
net1p = lambda x: np.argmax(mcts1.get_action_probs(x, temp=0))

if human_vs_bot:
    player2 = human_player
else:
    net2 = NeuralNet(game)
    net2.load_checkpoint('./','6x6_100checkpoints_best.pth.tar')

    mcts2 = MCTS(game, net2)
    net2p = lambda x: np.argmax(mcts2.get_action_probs(x, temp=0))

    player2 = net2p

arena = Arena(net1p, player2, game, display=Game.display)

print(arena.play_games(2, verbose=True))

    



