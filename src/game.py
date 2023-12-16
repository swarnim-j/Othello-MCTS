import numpy as np

from src.othello import Board

class Game:
    def __init__(self, n):
        self.n = n

    def get_init_board(self):
        board = Board(self.n)
        return board.pieces
    
    def get_board_size(self):
        return self.n
    
    def get_action_size(self):
        return self.n * self.n + 1
    
    def get_next_state(self, board, player, action):
        if action == self.n * self.n:
            return (board, -player)
        else:
            row = action // self.n
            col = action % self.n
            new_board = Board(self.n)
            new_board.pieces = [x[:] for x in board]
            new_board.move((row, col), player)
            return (new_board.pieces, -player)
        
    def get_valid_moves(self, board, player):
        valid_moves = [0] * self.get_action_size()
        legal_moves = board.get_legal_moves(player)
        if legal_moves:
            for row, col in legal_moves:
                valid_moves[row * self.n + col] = 1
        valid_moves[self.n * self.n] = 1
        return valid_moves
    
    def get_symmetries(self, board, pi):
        assert(len(pi) == self.n ** 2 + 1)
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []
        
        for i in range(1, 5):
            for j in [True, False]:
                new_b = np.rot90(board, i)
                new_pi = np.rot90(pi_board, i)
                if j:
                    new_b = np.fliplr(new_b)
                    new_pi = np.fliplr(new_pi)
                l += [(new_b, list(new_pi.ravel()) + [pi[-1]])]
        return l
    
    def has_game_ended(self, board, player):
        if board.has_legal_moves(player) or board.has_legal_moves(-player):
            return 0
        if board.count(player) > board.count(-player):
            return player
        return -player
    
    def get_canonical_board(self, board, player):
        return player * board
    
    def score(self, board, player):
        return board.count(player) - board.count(-player)

    def get_string(self, board):
        return str(board)