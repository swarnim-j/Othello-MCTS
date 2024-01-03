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

    def playGame(self) -> int:
            """
            Executes one episode of a game.
            Returns:
                Integer with the result of the game for player1.
            """
            board = self.game.getInitBoard()

            players = [self.player1, None, self.player2]

            current_player = 1

            i = 0

            # Play game
            while self.game.hasGameEnded(board, current_player) == 0:
                i += 1
                # Get move
                move = players[current_player + 1](self.game.getCannonicalForm(board, current_player))

                valids = self.game.getValidMoves(self.game.getCannonicalForm(board, current_player), 1)

                if valids[move] != 0:
                    board, current_player = self.game.nextState(board, move)

            # Get result
            result = self.game.hasGameEnded(self.board, current_player)

            return result * current_player
    
    def playGames(self, num_games: int) -> (int, int, int):
            """
            Play a specified number of games between two players and return the results.

            Args:
                num_games (int): The number of games to be played.

            Returns:
                tuple: A tuple containing the number of wins for player 1, player 2, and draws, respectively.
            """
            num = num_games // 2
            wins = [0, 0]
            draws = 0

            for _ in tqdm(range(num), desc="Arena.playGames (P1 starts)"):
                result = self.playGame()
                wins[0] += result
                if result == 0:
                    draws += 1
                elif result == -1:
                    wins[1] += 1
                else:
                    wins[0] += 1

            self.player1, self.player2 = self.player2, self.player1

            for _ in tqdm(range(num), desc="Arena.playGames (P2 starts)"):
                result = self.playGame()
                wins[0] += result
                if result == 0:
                    draws += 1
                elif result == -1:
                    wins[0] += 1
                else:
                    wins[1] += 1

            return (*wins, draws)

