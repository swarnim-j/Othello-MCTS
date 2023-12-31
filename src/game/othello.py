from board import Board

class Game:
    def __init__(self, players, size):
        self.players = players
        self.board = Board(size)
        