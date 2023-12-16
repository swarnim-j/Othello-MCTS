import logging
import os
import numpy as np
from tqdm import tqdm
from collections import deque
from random import shuffle
from pickle import Pickler, Unpickler
import sys

from src.arena import Arena
from src.MCTS import MCTS

log = logging.getLogger(__name__)


class Coach:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game) # new instance of the same class
        self.args = args
        self.mcts = MCTS(self.game, self.nnet)
        self.train_examples_history = []
        self.skip_first_self_play = False

    def execute_episode(self):
        train_examples = []
        board = self.game.get_init_board()
        current_player = 1
        episode_step = 0

        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_board(board, current_player)
            temp = int(episode_step < self.args.temp_threshold)

            policy = self.mcts.get_action_probs(canonical_board, temp=temp)
            symmetries = self.game.get_symmetries(canonical_board, policy) # data augmentation

            for sym_board, sym_policy in symmetries: # self-play
                train_examples.append([sym_board, current_player, sym_policy, None])

            action = np.random.choice(len(policy), p=policy) # sample action from policy
            board, current_player = self.game.get_next_state(board, current_player, action)

            game_result = self.game.get_game_ended(board, current_player)

            if game_result != 0:
                return [(x[0], x[2], game_result * ((-1) ** (x[1] != current_player))) for x in train_examples]

    def learn(self):
        for iteration in range(1, self.args.num_iters + 1):
            log.info(f'Starting Iter #{iteration} ...')

            if not self.skip_first_self_play or iteration > 1:
                iteration_train_examples = deque([], maxlen=self.args.maxlen_of_queue)

                for _ in tqdm(range(self.args.num_eps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)
                    iteration_train_examples += self.execute_episode()

                self.train_examples_history.append(iteration_train_examples)

            if len(self.train_examples_history) > self.args.num_iters_for_train_examples_history:
                log.warning(f"Removing the oldest entry in train_examples_history. "
                            f"len(train_examples_history) = {len(self.train_examples_history)}")
                self.train_examples_history.pop(0)

            self.save_train_examples(iteration - 1)

            train_examples = [example for sublist in self.train_examples_history for example in sublist]
            shuffle(train_examples)

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(train_examples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.get_action_prob(x, temp=0)),
                          lambda x: np.argmax(nmcts.get_action_prob(x, temp=0)), self.game)
            pwins, nwins, draws = arena.play_games(self.args.arena_compare)

            log.info(f'NEW/PREV WINS : {nwins} / {pwins} ; DRAWS : {draws}')

            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.update_threshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.get_checkpoint_file(iteration))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def get_checkpoint_file(self, iteration):
        return f'checkpoint_{iteration}.pth.tar'

    def save_train_examples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.get_checkpoint_file(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.train_examples_history)
        f.closed

    def load_train_examples(self):
        model_file = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examples_file = model_file + ".examples"

        if not os.path.isfile(examples_file):
            log.warning(f'File "{examples_file}" with train_examples not found!')
            user_input = input("Continue? [y|n]")
            if user_input.lower() != "y":
                sys.exit()
        else:
            log.info("File with train_examples found. Loading it...")
            with open(examples_file, "rb") as f:
                self.train_examples_history = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skip_first_self_play = True

