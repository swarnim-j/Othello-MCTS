from model.OthelloNet import OthelloNet
from src.game.game import Game

import numpy as np
import torch.optim as optim
import torch
from tqdm import tqdm

args = {}

class OthelloModel():
    def __init__(self, game: Game) -> None:
        self.net = OthelloNet(game, args)
        self.x, self.y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self) -> None:
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
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                pi, v = self.net(boards)

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

    def predict():
        pass
