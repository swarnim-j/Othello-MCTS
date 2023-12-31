import numpy as np

EPS = 1e-8

class MCTS:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args

        self.Qsa = {}      # stores Q values for s, a
        self.Nsa = {}      # stores times edge s, a was visited
        self.Ns = {}       # stores times board s was visited
        self.Ps = {}       # stores policy (probabilities) returned by neural network

        self.Ended_s = {}  # stores if state is terminal
        self.Valids_s = {} # stores valid moves for each state

    def search(self, canonical_board):
        state = str(canonical_board)

        if state not in self.Ended_s:
            self.Ended_s[state] = self.game.hasGameEnded(canonical_board, 1)

        if self.Ended_s[state] != 0:
            # terminal node
            return -self.Ended_s[state]
        
        
            
