from src.game.board import Board
from src.MCTS import MCTS

from abc import ABC, abstractmethod

import numpy as np

class OthelloPlayer(ABC):
    """
    Abstract class for a player.
    """
    @abstractmethod
    def getAction(self, board: Board) -> int:
        """
        Returns the action to play in a game.
        """
        pass

class RandomPlayer(OthelloPlayer):
    """
    Player that plays a random action.
    """
    def getAction(self, board: Board) -> int:
        """
        Returns a random action.
        """
        return np.random.choice(board.getLegalMoves())
    
class HumanPlayer(OthelloPlayer):
    """
    Player that asks for input to play an action.
    """
    def getAction(self, board: Board) -> int:
        """
        Returns the action selected by the user.
        """
        valids = board.getLegalMoves()
        while True:
            action = input()
            action = int(action)
            x, y = action // board.getBoardSize(), action % board.getBoardSize()
            if (x, y) in valids:
                break
            else:
                print('Invalid')
        return action
    
class GreedyPlayer(OthelloPlayer):
    """
    Player that plays the action with the highest value.
    """
    def getAction(self, board: Board) -> int:
        """
        Returns the action with the highest value.
        """
        valids = board.getLegalMoves()
        actions = []
        for action in range(board.getBoardSize() ** 2 + 1):
            action = (action // board.getBoardSize(), action % board.getBoardSize())
            if action in valids:
                next_pieces = board.playMove(action)
                next_board = Board(board.getBoardSize())
                next_board.pieces = next_pieces
                score = next_board.diff(1)
                actions.append((-score, action))
        actions.sort()
        best_action = board.getBoardSize() ** 2 if len(actions) == 0 else actions[0][1]
        return best_action
    
class MCTSPlayer(OthelloPlayer):
    """
    Player that uses the MCTS algorithm to play an action.
    """
    def __init__(self, mcts: MCTS) -> None:
        """
        Initializes the MCTSPlayer class.

        Parameters:
        - mcts (MCTS): The MCTS algorithm instance used by the player.
        """
        self.mcts = mcts
        
        
    def getAction(self, board: Board) -> int:
        """
        Returns the action selected by the MCTS algorithm.

        Parameters:
        - board (Board): The current game board.

        Returns:
        - int: The selected action.
        """
        pi = self.mcts.simulate(board)
        action = np.random.choice(len(pi), p=pi)
        return action