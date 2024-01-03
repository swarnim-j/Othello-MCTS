import torch
import torch.nn as nn
import torch.nn.functional as F

from src.game.game import Game

class OthelloNet(nn.Module):
    def __init__(self, game: Game, args):
        """
        Initializes the OthelloNet model.

        Args:
            game (Game): The game object.
            args: The arguments for the model.
        """
        super(OthelloNet, self).__init__()

        self.x, self.y = game.getBoardSize()
        self.args = args
        channels = args.num_channels

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, stride=1)
        self.conv4 = nn.Conv2d(channels, channels, 3, stride=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(channels)

        # Fully connected layers
        self.fc1 = nn.Linear(channels * (self.x - 4) * (self.y - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        # Output layers
        self.fc3 = nn.Linear(512, game.getActionSize()) # Policy head
        self.fc4 = nn.Linear(512, 1)                    # Value head

    def forward(self, s):
        """
        Performs forward pass through the model.

        Args:
            s: The input tensor.

        Returns:
            Tuple: The policy and value outputs.
        """
        # Input reshaping
        s = s.view(-1, 1, self.x, self.y)

        # Convolutional layers + batch normalization + ReLU
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))

        # Flatten
        s = s.view(-1, self.args.num_channels * (self.x - 4) * (self.y - 4))

        # Fully connected layers + batch normalization + ReLU + dropout
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)

        # Output layers
        pi = self.fc3(s)
        v = self.fc4(s)

        # Return policy and value
        return F.log_softmax(pi, dim=1), torch.tanh(v)