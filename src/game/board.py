class Board:
    """
    Represents the game board for Othello.

    Attributes:
    - n (int): The size of the board.
    - pieces (list[list[int]]): The matrix representing the board state.

    Methods:
    - __init__(self, size: int) -> None: Initializes the board with the given size.
    - __getitem__(self, key: (int, int)) -> int: Returns the value at the specified position on the board.
    - __str__(self) -> str: Returns a string representation of the board.
    - getBoardSize(self) -> int: Returns the size of the board.
    - diff(self, player: int) -> int: Returns the difference in the number of pieces between the specified color and its opponent.
    - isValidMove(self, x: int, y: int, dx: int, dy: int, player: int) -> bool: Checks if a move is valid for the specified color at the given position.
    - getLegalMoves(self, player: int) -> list[(int, int)]: Returns a list of legal moves for the specified color.
    - playMove(self, move: int) -> list[list[int]]: Plays the specified move on the board and returns the updated board state.
    - printBoard(self) -> None: Prints the current board state.
    """

    DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]

    def __init__(self, size: int) -> None:
        """
        Initializes a new instance of the Board class.

        Args:
            size (int): The size of the board.

        Returns:
            None
        """
        self.n = size

        self.pieces = [[0] * size for _ in range(size)]

        self.pieces[size // 2 - 1][size // 2 - 1] = 1
        self.pieces[size // 2][size // 2] = 1
        self.pieces[size // 2 - 1][size // 2] = -1
        self.pieces[size // 2][size // 2 - 1] = -1

    def __getitem__(self, key: (int, int)) -> int:
        """
        Get the value at the specified position on the board.

        Args:
            key: A tuple representing the position on the board (x, y).

        Returns:
            The value at the specified position on the board.

        """
        x, y = key
        return self.pieces[x][y]

    def __str__(self) -> str:
        """
        Returns a string representation of the board.

        Returns:
            str: The string representation of the board.
        """
        return str(self.pieces)

    def getBoardSize(self) -> int:
        """
        Returns the size of the board.

        Returns:
            int: The size of the board.
        """
        return self.n

    def diff(self, player: int) -> int:
        """
        Calculates the difference between the count of pieces of the specified player and the count of pieces of the opposite player on the board.

        Args:
            player (int): The player of the pieces to calculate the difference for.

        Returns:
            int: The difference between the count of pieces of the specified player and the count of pieces of the opposite player.
        """
        return sum([row.count(player) - row.count(-player) for row in self.pieces])

    def isValidMove(self, x: int, y: int, dx: int, dy: int, player: int) -> bool:
        """
        Check if a move is valid for a given player at a specific position on the board.

        Args:
            x (int): The x-coordinate of the position.
            y (int): The y-coordinate of the position.
            dx (int): The change in x-coordinate for each step.
            dy (int): The change in y-coordinate for each step.
            player (int): The player making the move.

        Returns:
            bool: True if the move is valid, False otherwise.
        """
        if x + dx >= self.n or x + dx < 0 or y + dy >= self.n or y + dy < 0:
            return False
        if self.pieces[x + dx][y + dy] == player:
            return False
        for i in range(2, self.n):
            if x + i * dx >= self.n or x + i * dx < 0 or y + i * dy >= self.n or y + i * dy < 0:
                return False
            if self.pieces[x + i * dx][y + i * dy] == 0:
                return False
            if self.pieces[x + i * dx][y + i * dy] == player:
                return True
        return False

    def getLegalMoves(self, player: int) -> list[(int, int)]:
        """
        Returns a list of legal moves for the specified player.

        Args:
            player (int): The player for whom to find legal moves.

        Returns:
            list[(int, int)]: A list of legal moves represented as (x, y) coordinates.
        """
        moves = []
        for x in range(self.n):
            for y in range(self.n):
                if self.pieces[x][y] != 0:
                    continue
                for dx, dy in self.DIRECTIONS:
                    if self.isValidMove(x, y, dx, dy, player):
                        moves.append((x, y))
                        break
        return moves

    def playMove(self, move: int) -> list[list[int]]:
        """
        Plays a move on the game board.

        Args:
            move (int): The move to be played.

        Returns:
            list[list[int]]: The updated game board after playing the move.
        """
        if move == self.n * self.n:
            return self.pieces
        x, y = move // self.n, move % self.n
        self.pieces[x][y] = 1
        for dx, dy in self.DIRECTIONS:
            if self.isValidMove(x, y, dx, dy, 1):
                for i in range(1, self.n):
                    if x + i * dx >= self.n or x + i * dx < 0 or y + i * dy >= self.n or y + i * dy < 0:
                        break
                    if self.pieces[x + i * dx][y + i * dy] == 1:
                        break
                    self.pieces[x + i * dx][y + i * dy] = 1
        return self.pieces

    def printBoard(self) -> None:
        """
        Prints the current state of the game board.
        X represents player 1's pieces.
        O represents player 2's pieces.
        _ represents empty spaces on the board.
        """
        print("   ", end="")
        for y in range(self.n):
            print(y, end=" ")
        print()
        for x in range(self.n):
            print(x, end=": ")
            for y in range(self.n):
                if self.pieces[x][y] == 1:
                    print("X", end=" ")
                elif self.pieces[x][y] == -1:
                    print("O", end=" ")
                else:
                    print("_", end=" ")
            print()
