import numpy as np
import math

from game.board import Board
from src.game.game import Game
from src.model.model import OthelloModel

EPS = 1e-8

class MCTS:
    def __init__(self, game: Game, model: OthelloModel, args) -> None:
        self.game = game
        self.model = model
        self.args = args

        self.Q_sa = {}     # stores Q values for s, a
        self.N_sa = {}     # stores times edge s, a was visited
        self.N_s = {}      # stores times board s was visited
        self.P_s = {}      # stores policy (probabilities) returned by neural network

        self.Ended_s = {}  # stores if state is terminal
        self.Valids_s = {} # stores valid moves for each state

    def simulate(self, canonical_board: Board) -> list[float]:
        for _ in range(self.args.num_sims):
            self.search(canonical_board)

        state = str(canonical_board)

        counts = [self.N_sa[(state, action)] if (state, action) in self.N_sa else 0 for action in range(self.game.getActionSize())]
        
        counts_sum = float(sum(counts))

        probabilities = [count / counts_sum for count in counts]

        return probabilities

    def search(self, canonical_board: Board) -> float:
        state = str(canonical_board)

        if state not in self.Ended_s:
            self.Ended_s[state] = self.game.hasGameEnded(canonical_board, 1)

        if self.Ended_s[state] != 0:
            # terminal node
            return -self.Ended_s[state]
        
        if state not in self.P_s:
            # leaf node
            self.P_s[state], value = self.model.predict(canonical_board)

            self.Valids_s[state] = self.game.getValidMoves(canonical_board, 1)
            self.P_s = self.P_s * self.Valids_s[state]

            if np.sum(self.P_s[state]) > 0:
                self.P_s[state] /= np.sum(self.P_s[state])
            else:
                # some error
                self.P_s[state] += self.Valids_s[state]
                self.P_s[state] /= np.sum(self.P_s[state])

            self.N_s[state] = 0

            return -value
        
        action = self.bestMove(state)

        next_state_board = Board(self.game.n)
        next_state_board.pieces = self.game.nextState(canonical_board, action)

        # expand
        value = self.search(next_state_board)

        # backpropagate value from child nodes, i.e., update
        self.Q_sa[(state, action)] = (self.N_sa[(state, action)] * self.Q_sa[(state, action)] + value) / (self.N_sa[(state, action)] + 1)
        self.N_sa[(state, action)] += 1

        return -value
    
    def bestMove(self, state: str) -> int:
        valid_moves = self.Valids_s[state]
        best_u = -float('inf')
        best_action = -1 # no valid moves, by default

        # search
        for action in range(self.game.getActionSize()):
            if valid_moves[action]:
                # upper confidence bound
                if (state, action) in self.Q_sa:
                    u = self.Q_sa[(state, action)] + self.args.c_puct * self.P_s[state][action] * math.sqrt(self.N_s[state]) / (1 + self.N_sa[(state, action)])
                else:
                    u = self.args.c_puct * self.P_s[state][action] * math.sqrt(self.N_s[state] + EPS)
                if u > best_u:
                    best_u = u
                    best_action = action

        return best_action

