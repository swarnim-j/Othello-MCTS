from board import Board

class Othello:
    def __init__(self, players, size):
        self.players = players
        self.board = Board(size)

    def hasGameEnded(self, board : Board, player : int):
        if board.getValidMoves(player) == []:
            return True
        return False
    

