import logging
from tqdm import tqdm

log = logging.getLogger(__name__)

class Arena:
    def __init__(self, player1, player2, game, display=None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def play_game(self, verbose=False):
        players = [self.player2, None, self.player1]
        current_player = 1
        board = self.game.get_init_board()
        turn = 0

        while self.game.has_game_ended(board, current_player) == 0:
            turn += 1

            if verbose:
                assert self.display
                print("Turn", str(turn), "Player", str(current_player))
                self.display(board)

            action = players[current_player + 1].play(board, current_player)
            valid_moves = self.game.get_valid_moves(board, current_player)
            assert valid_moves[action] == 1
            board, current_player = self.game.get_next_state(board, current_player, action)

        if verbose:
            assert self.display
            print("Game over:", str(self.game.has_game_ended(board, current_player)))
            self.display(board)

        return self.game.has_game_ended(board, current_player)
    
    def play_games(self, num_games, verbose=False):
        num_games //= 2
        player1_wins = 0
        player2_wins = 0
        draws = 0

        for _ in tqdm(range(num_games), desc="Arena.play_games (1)"):
            game_result = self.play_game(verbose=verbose)
            if game_result == 1:
                player1_wins += 1
            elif game_result == -1:
                player2_wins += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num_games), desc="Arena.play_games (2)"):
            game_result = self.play_game(verbose=verbose)
            if game_result == -1:
                player1_wins += 1
            elif game_result == 1:
                player2_wins += 1
            else:
                draws += 1
        
        return player1_wins, player2_wins, draws
