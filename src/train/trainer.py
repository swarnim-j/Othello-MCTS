from src.game.game import Game
from src.model.model import OthelloModel
from src.game.board import Board
from src.MCTS import MCTS

import numpy as np
from tqdm import tqdm
from collections import deque

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
        self.nnet = model
        self.pnet = self.nnet.__class__(self.game)
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.training_examples_history = []

    def playEpisode(self) -> list[(Board, list[float], float)]:
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
                    self.mcts = MCTS(self.game, self.nnet, self.args)
                    train_examples += self.playEpisode()

                self.training_examples_history.append(train_examples)

            if len(self.training_examples_history) > self.args.num_iters_history:
                print("Clearing training examples history")
                self.training_examples_history.pop(0)
            
            self.nnet.train(self.training_examples_history)
        