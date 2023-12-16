import numpy as np

class MCTS():
    def __init__(self, game, nnet, c_puct=1.0, n_sims=1000):
        self.game = game
        self.nnet = nnet

        self.c_puct = c_puct
        self.n_sims = n_sims

        self.Q_sa = {} # stores Q values for s,a
        self.N_sa = {} # number of times action a was taken from state s
        self.P_s = {} # prior probability of selecting action a in state s
        self.N_s = {} # number of times state s was visited

        self.E_s = {} # end state
        self.V_s = {} # valid moves

    def get_action_probs(self, board, temp=1):
        for _ in range(self.n_sims):
            self.simulate(board)

        state = self.game.get_string(board)

        visits = [self.N_sa[(state, action)] if (state, action) in self.N_sa else 0
                  for action in range(self.game.get_action_size())]
        
        if temp == 0:
            best_actions = np.argwhere(visits == np.max(visits)).flatten()
            best_action = np.random.choice(best_actions)
            probs = [0] * len(visits)
            probs[best_action] = 1
            return probs

        visits = [x ** (1. / self.c_puct) for x in visits]
        visits_sum = float(sum(visits))
        probs = [x / visits_sum for x in visits]
        return probs
    
    def simulate(self, board):
        state = self.game.get_string(board)

        if state not in self.E_s:
            self.E_s[state] = self.game.has_game_ended(board, 1)
        if self.E_s[state] != 0:
            return -self.E_s[state]
        
        if state not in self.P_s:
            self.P_s[state], value = self.nnet.predict(board)
            self.V_s[state] = self.game.get_valid_moves(board, 1)
            self.P_s[state] = self.P_s[state] * self.V_s[state]
        
            sum_P_s = np.sum(self.P_s[state])

            if sum_P_s > 0:
                self.P_s[state] /= sum_P_s
            else:
                self.P_s[state] = self.V_s[state] / np.sum(self.V_s[state])

            self.N_s[state] = 0
            return -value
        
        valid_moves = self.V_s[state]
        
        best_action = self.get_best_action(state, valid_moves)
        
        next_state, next_player = self.game.get_next_state(board, 1, best_action)
        next_state = self.game.get_canonical_board(next_state, next_player)

        value = self.simulate(next_state)

        if (state, best_action) in self.Q_sa:
            self.Q_sa[(state, best_action)] = (self.N_sa[(state, best_action)] * self.Q_sa[(state, best_action)] + value) / \
                (self.N_sa[(state, best_action)] + 1)
            self.N_sa[(state, best_action)] += 1
        else:
            self.Q_sa[(state, best_action)] = value
            self.N_sa[(state, best_action)] = 1

        self.N_s[state] += 1

        return -value
    
    def get_best_action(self, state, valid_moves):
        current_best, best_action = -float('inf'), -1

        for action in range(self.game.get_action_size()):
            if valid_moves[action]:
                u = self.calculate_action_value(state, action)
                if u > current_best:
                    current_best = u
                    best_action = action

        return best_action
    
    def calculate_action_value(self, state, action):
        if (state, action) in self.Q_sa:
            u = self.Q_sa[(state, action)] + self.c_puct * self.P_s[state][action] * \
                np.sqrt(self.N_s[state]) / (1 + self.N_sa[(state, action)])
        else:
            u = self.c_puct * self.P_s[state][action] * np.sqrt(self.N_s[state] + 1e-8)
        return u