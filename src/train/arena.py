class Arena:
    def __init__(self, player1: Player, player2: Player, game: Game, display: Display):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def play_game(self, verbose=False) -> int:
        """
        Executes one episode of a game.
        Returns:
            Integer with the result of the game for player1.
        """
        # Reset players
        self.player1.reset()
        self.player2.reset()

        # Reset game
        self.game.reset()

        # Play game
        while not self.game.is_over():
            # Get current player
            player = self.game.get_current_player()

            # Get action from player
            action = player.get_action(self.game)

            # Play action
            self.game.play_action(action)

            # Display game
            if verbose:
                self.display.display(self.game)

        # Get result
        result = self.game.get_result(self.player1)

        # Update players
        self.player1.update(result)
        self.player2.update(-result)

        # Return result
        return result