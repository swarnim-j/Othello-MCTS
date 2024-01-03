from src.model.OthelloNet import OthelloNet
from src.game.game import Game
from src.game.board import Board

import os
import numpy as np

import torch.optim as optim
import torch
from tqdm import tqdm

args = {}

class OthelloModel():
    def __init__(self, game: Game) -> None:
        """
        Initializes the OthelloModel class.

        Args:
            game (Game): The game instance.

        Returns:
            None
        """
        self.net = OthelloNet(game, args)
        self.x, self.y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self) -> None:
        """
        Trains the model using the examples stored in self.examples.

        Returns:
            None
        """
        optimizer = optim.Adam(self.net.parameters())

        for epoch in tqdm(range(args.epochs)):
            print('EPOCH ::: ' + str(epoch + 1))
            pi_losses = []
            v_losses = []
            self.net.train()

            batch_count = int(len(self.examples) / args.batch_size)

            for _ in tqdm(range(batch_count), desc="OthelloNet.train"):
                sample_ids = np.random.randint(len(self.examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[self.examples[i] for i in sample_ids]))
                boards = [board.pieces for board in boards]
                boards_tensor = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                pi, v = self.net(boards_tensor)

                # compute loss
                pi_loss = -torch.sum(target_pis * pi) / target_pis.size()[0]
                v_loss = torch.sum((target_vs - v.view(-1)) ** 2) / target_vs.size()[0]
                total_loss = pi_loss + v_loss

                # record loss
                pi_losses.append(pi_loss.item())
                v_losses.append(v_loss.item())

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board: Board) -> (list[float], float):
        """
        Predicts the policy and value for a given board state.

        Args:
            board (Board): The board state.

        Returns:
            tuple: A tuple containing the policy (list of probabilities) and the value (float).
        """
        board_tensor = torch.FloatTensor(board.pieces.astype(np.float64))
        board_tensor = board_tensor.view(1, self.x, self.y)
        self.net.eval()
        with torch.no_grad():
            pi, v = self.net(board_tensor)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
    
    def saveCheckpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        """
        Saves the model checkpoint.

        Args:
            folder (str): The folder to save the checkpoint in.
            filename (str): The filename of the checkpoint.

        Returns:
            None
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("making new directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("checkpoint directory exists")
        torch.save({'state_dict': self.net.state_dict()}, filepath)

    def loadCheckpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        """
        Loads the model checkpoint.

        Args:
            folder (str): The folder to load the checkpoint from.
            filename (str): The filename of the checkpoint.

        Returns:
            None
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("no model in path {}".format(filepath))
        checkpoint = torch.load(filepath)
        self.net.load_state_dict(checkpoint['state_dict'])
    
        
