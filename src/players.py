import numpy as np

class RandomPlayer():
    def __init__(self, identity):
        self.identity = identity

    def make_move(self, game_board, player):
        legal_moves = game_board.get_legal_moves(player)
        action = np.random.choice(legal_moves)

        while legal_moves[action] != 1:
            action = np.random.choice(legal_moves)

        return action

class HumanPlayer():
    def __init__(self, identity):
        self.identity = identity

    def make_move(self, game_board):
        valid_moves = game_board.get_valid_moves(self.identity)

        for i, is_valid in enumerate(valid_moves):
            if is_valid:
                print(f"[{i // game_board.n}, {i % game_board.n}] ", end="")
        
        while True:
            input_move = input()
            input_a = input_move.split(" ")

            if len(input_a) == 2:
                try:
                    x, y = [int(i) for i in input_a]

                    if (0 <= x < game_board.n and 0 <= y < game_board.n) or (x == game_board.n and y == 0):
                        a = game_board.n * x + y if x != -1 else game_board.n ** 2

                        if valid_moves[a]:
                            break
                except ValueError:
                    print('Invalid input. Please enter two integers.')
            print('Invalid move')

        return a

class GreedyPlayer():
    def __init__(self, identity):
        self.identity = identity

    def make_move(self, game_board):
        valid_moves = game_board.get_valid_moves(self.identity)
        candidates = []

        for action in range(game_board.get_action_size()):
            if valid_moves[action] == 0:
                continue

            next_board, _ = game_board.get_next_state(game_board, self.identity, action)
            score = game_board.score(next_board, self.identity)
            candidates.append((-score, action))

        candidates.sort()
        return candidates[0][1]
