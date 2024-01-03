from src.game.game import Game
from src.model.model import OthelloModel
from src.game.board import Board
from src.MCTS import MCTS
from src.train.arena import Arena
from train.players import *

import os
from random import shuffle
import sys
import numpy as np
from tqdm import tqdm
from collections import deque
from pickle import Pickler, Unpickler

class Trainer():
    def __init__(self, game: Game, model: OthelloModel, args) -> None:
        """
        Initializes a Trainer object.

        Args:
            game (Game): The game object.
            model (OthelloModel): The model object.
            args: Additional arguments.

        Returns:
            None
        """
        self.game = game
        self.player1_net = model
        self.player2_net = self.nnet.__class__(self.game)
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.training_examples_history = []

    def runEpisode(self) -> list[(Board, list[float], float)]:
        """
        Executes one episode of a game.

        Returns:
            training_examples (list[(Board, list[float], float)]): A list of examples of the form (canonical_board, probabilities, value).
        """
        training_examples = []
        board = self.game.getInitialBoard()
        self.current_player = 1
        episode_step = 0

        while True:
            episode_step += 1
            canonical_board = self.game.getCannonicalForm(board, self.current_player)

            pi = self.mcts.simulate(canonical_board)

            symmetries = self.game.getSymmetries(canonical_board, pi)

            for board, pi in symmetries:
                training_examples.append([board, self.current_player, pi, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.current_player = self.game.nextState(board, action, self.current_player)

            result = self.game.hasGameEnded(board, self.current_player)

            if result != 0:
                return [(x[0], x[2], result * ((-1) ** (x[1] != self.current_player))) for x in training_examples]
            
    def learn(self) -> None:
        """
        Performs num_iters iterations with num_eps episodes of self-play in each iteration.

        Returns:
            None
        """
        for i in range(1, self.args.num_iters + 1):
            if i > 1:
                train_examples = deque([], maxlen=self.args.maxlen_queue)
                
                for _ in range(tqdm(self.args.num_eps, desc="SelfPlay.learn")):
                    self.mcts = MCTS(self.game, self.player1_net, self.args)
                    train_examples += self.runEpisode()

                self.training_examples_history.append(train_examples)

            if len(self.training_examples_history) > self.args.num_iters_history:
                print("Clearing training examples history")
                self.training_examples_history.pop(0)
            
            self.saveExamples(i - 1)

            train_examples = []
            for e in self.training_examples_history:
                train_examples.extend(e)
            
            shuffle(train_examples)

            # TODO: change file and folder name
            self.player1_net.saveCheckpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.player2_net.loadCheckpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            player2_mcts = MCTS(self.game, self.player2_net, self.args)

            self.player1_net.train(train_examples)
            player1_mcts = MCTS(self.game, self.player1_net, self.args)

            arena = Arena(MCTSPlayer(player1_mcts), MCTSPlayer(player2_mcts), self.game)

            p1_wins, p2_wins, draws = arena.playGames(self.args.arena_eps)

            print("P1 Wins:", p1_wins)
            print("P2 Wins:", p2_wins)
            print("Draws:", draws)

            if p1_wins + p2_wins == 0 or float(p1_wins) / (p1_wins + p2_wins) < self.args.update_threshold:
                print("Rejecting new model")
                # TODO: change file and folder name
                self.player1_net.loadCheckpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print("Accepting new model")
                self.player1_net.saveCheckpoint(folder=self.args.checkpoint, filename='best.pth.tar')



    def saveExamples(self, iter: int) -> None:
        """
        Saves the training examples to a file.

        Returns:
            None
        """
        # TODO: change file name and folder name
        folder = './temp/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, "checkpoint_" + str(iter) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.training_examples_history)
        f.closed

    def loadExamples(self) -> None:
        """
        Loads the training examples from a file.

        Returns:
            None
        """
        # TODO: change file name and folder name
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with training examples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with training examples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.training_examples_history = Unpickler(f).load()
            f.closed
        