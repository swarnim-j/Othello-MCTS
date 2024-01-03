from board import Board

import numpy as np

class Game:
    """
    Represents the game logic for Othello.
    """

    def __init__(self, size: int) -> None:
        """
        Initializes a new instance of the Game class.

        Args:
            size (int): The size of the game board.
        """
        self.n = size

    def getInitialBoard(self) -> Board:
        """
        Gets the initial game board.

        Returns:
            Board: The initial game board.
        """
        return Board(self.n)

    def hasGameEnded(self, board: Board, player: int) -> int:
        """
        Checks if the game has ended.

        Args:
            board (Board): The current game board.
            player (int): The current player.

        Returns:
            int: 0 if the game has not ended, 1 if the current player has won, -1 if the opponent has won.
        """
        if (len(board.getLegalMoves(1)) or len(board.getLegalMoves(-1))):
            return 0
        diff = board.diff(player)
        if diff > 0:
            return 1
        return -1

    def getValidMoves(self, board: Board, player: int) -> list[int]:
        """
        Gets the valid moves for the current player.

        Args:
            board (Board): The current game board.
            player (int): The current player.

        Returns:
            list[int]: A list of valid moves represented as integers.
        """
        moves = board.getLegalMoves(player)
        valids = [0] * self.getActionSize()
        if len(moves) == 0:
            valids[-1] = 1
        for x, y in moves:
            valids[self.n * x + y] = 1
        return valids
    
    def getActionSize(self) -> int:
        """
        Gets the total number of possible actions.

        Returns:
            int: The total number of possible actions.
        """
        return self.n * self.n + 1
    
    def nextState(self, board: Board, player: int, move: int) -> (Board, int):
        """
        Gets the next state of the game after a move is played.

        Args:
            board (Board): The current game board.
            player (int): The current player.
            move (int): The move to be played.

        Returns:
            (Board, int): The next game board and the next player.
        """
        if move == self.n * self.n:
            return board, -player
        board.playMove(move, player)
        return board, player
    
    def getBoardSize(self) -> (int, int):
        """
        Gets the size of the game board.

        Returns:
            (int, int): The size of the game board as a tuple (rows, columns).
        """
        return (self.n, self.n)

    def getCanonicalForm(self, board: Board, player: int) -> Board:
        """
        Gets the canonical form of the game board for the specified player.

        Args:
            board (Board): The current game board.
            player (int): The player for whom to get the canonical form.

        Returns:
            Board: The canonical form of the game board.
        """
        if player == 1:
            return board
        new_board = Board(self.n)
        new_board.pieces = [[-p for p in row] for row in board.pieces]
        return new_board
    
    def getSymmetries(self, canonical_board: Board, pi: list[float]) -> list[(Board, list[float])]:
        """
        Gets the symmetries of the game board and policy vector.

        Args:
            board (Board): The current game board.
            pi (list[float]): The policy vector.

        Returns:
            list[(Board, list[float])]: A list of tuples containing the symmetries of the game board and policy vector.
        """
        board = canonical_board.pieces
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []
        for i in range(1, 5):
            for j in [True, False]:
                new_b = np.rot90(board, i)
                new_pi = np.rot90(pi_board, i)
                if j:
                    new_b = np.fliplr(new_b)
                    new_pi = np.fliplr(new_pi)
                new_board = Board(self.n)
                new_board.pieces = new_b
                l += [(new_board, list(new_pi.ravel()) + [pi[-1]])]
        return l