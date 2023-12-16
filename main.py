import logging
import coloredlogs

from coach import Coach
from game import Game
from net import NeuralNet
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'num_iters': 100,
    'num_eps': 10,
    'maxlen_of_queue': 2000,
    'num_mcts_sims': 15,
    'arena_batch_size': 32,
    'num_iters_for_train_examples_history': 20,
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp','checkpoint_1.pth.tar'),
})

def main():
    log.info('Loading %s...', Game.__name__)
    game = Game(6)

    log.info('Loading %s...', NeuralNet.__name__)
    nnet = NeuralNet(game)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    coach = Coach(game, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        coach.load_train_examples()

    log.info('Starting the learning process...')
    coach.learn()

if __name__ == "__main__":
    main()