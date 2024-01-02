from board import Board

class Game:
    def __init__(self, size: int) -> None:
        self.n = size

    def hasGameEnded(self, board: Board, player: int) -> bool:
        if len(self.getValidMoves(board, player)) == 0:
            return True
        return False

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
    
    def nextState(self, board: Board, move: int) -> Board:
        new_board = Board(self.n)
        new_board.pieces = board.playMove(move)
        return new_board
    
    def getBoardSize(self) -> (int, int):
        return (self.n, self.n)

