from board import Board

class Game:
    def __init__(self, size: int) -> None:
        self.n = size

    def getInitialBoard(self) -> Board:
        return Board(self.n)

    def hasGameEnded(self, board: Board, player: int) -> int:
        if (len(board.getLegalMoves(1)) or len(board.getLegalMoves(-1))):
            return 0
        diff = board.diff(player)
        if diff > 0:
            return 1
        return -1

    def getValidMoves(self, board: Board, player: int) -> list[int]:
        moves = board.getLegalMoves(player)
        valids = [0] * self.getActionSize()
        if len(moves) == 0:
            valids[-1] = 1
        for x, y in moves:
            valids[self.n * x + y] = 1
        return valids
    
    def getActionSize(self) -> int:
        return self.n * self.n + 1
    
    def nextState(self, board: Board, player: int, move: int) -> (Board, int):
        if move == self.n * self.n:
            return board, -player
        board.playMove(move, player)
        return board, player
    
    def getBoardSize(self) -> (int, int):
        return (self.n, self.n)

    def getCannonicalForm(self, board: Board, player: int) -> Board:
        if player == 1:
            return board
        new_board = Board(self.n)
        new_board.pieces = [[-p for p in row] for row in board.pieces]
        return new_board