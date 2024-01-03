from src.game.game import Game
from src.model.model import OthelloModel
from MCTS.mcts import MCTS
from src.train.arena import Arena
from src.train.players import *

class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

args_dict = {
    'size': 6, # size of the board
    'opponent': 'mcts', # opponent to play against
    # TODO: add arguments
}

args = Args(**args_dict)

def main():
    player1 = HumanPlayer()
    player2 = None

    game = Game(args.size)

    if args.opponent == 'mcts':
        model = OthelloModel(game)
        # TODO: change file and folder names
        model.loadCheckpoint('./temp/', 'best.pth.tar')
        mcts = MCTS(game, model, args)
        player2 = MCTSPlayer(mcts)

    elif args.opponent == 'random':
        player2 = RandomPlayer()

    elif args.opponent == 'greedy':
        player2 = GreedyPlayer()

    else:
        raise ValueError('Invalid opponent')
    
    arena = Arena(player1, player2, game)

    arena.playGames(args.num_games)

if __name__ == "__main__":
    main()
