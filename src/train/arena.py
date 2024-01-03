from src.game.game import Game
from train.players import *
from tqdm import tqdm

class Arena:
    def __init__(self, player1: OthelloPlayer, player2: OthelloPlayer, game: Game):
            """
            Initializes an Arena object.

            Args:
                player1 (OthelloPlayer): The first player.
                player2 (OthelloPlayer): The second player.
                game (Game): The game object.
            """
            self.player1 = player1
            self.player2 = player2
            self.game = game

    def playGame(self, print_board: bool = False) -> int:
            """
            Executes one episode of a game.

            Args:
                print_board (bool): Flag to indicate whether to print the board during the game.

            Returns:
                int: Integer with the result of the game for player1.
            """
            board = self.game.getInitBoard()

            players = [self.player1, None, self.player2]

            current_player = 1

            i = 0

            # Play game
            while self.game.hasGameEnded(board, current_player) == 0:
                i += 1
                # Get move
                move = players[current_player + 1](self.game.getCanonicalForm(board, current_player))

                valids = self.game.getValidMoves(self.game.getCanonicalForm(board, current_player), 1)

                if valids[move] != 0:
                    board, current_player = self.game.nextState(board, move)

                # Print board
                if print_board:
                    print("Move ", i, "Player ", current_player)
                    board.printBoard()

            # Get result
            result = self.game.hasGameEnded(self.board, current_player)

            # Print result
            if print_board:
                print("Game over: Turn ", i, "Result ", result)

            return result * current_player
    
    def playGames(self, num_games: int, print_board: bool = False) -> (int, int, int):
        """
        Play a specified number of games between two players and return the results.

        Args:
            num_games (int): The number of games to be played.
            print_board (bool): Flag indicating whether to print the board after each move. Default is False.

        Returns:
            tuple: A tuple containing the number of wins for player 1, player 2, and draws, respectively.
        """
        num = num_games // 2
        wins = [0, 0]
        draws = 0

        for _ in tqdm(range(num), desc="Arena.playGames (P1 starts)"):
            result = self.playGame(print_board)
            wins[0] += result
            if result == 0:
                draws += 1
            elif result == -1:
                wins[1] += 1
            else:
                wins[0] += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (P2 starts)"):
            result = self.playGame(print_board)
            wins[0] += result
            if result == 0:
                draws += 1
            elif result == -1:
                wins[0] += 1
            else:
                wins[1] += 1

        return (*wins, draws)

