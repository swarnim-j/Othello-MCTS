from model.OthelloNet import OthelloNet
from src.game.game import Game

import numpy as np
import torch.optim as optim
import torch
from tqdm import tqdm

args = {}

class OthelloModel():
    def __init__(self, game: Game) -> None:
        self.model = OthelloNet(game, args)
        self.x, self.y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train():
        pass

    def predict():
        pass
