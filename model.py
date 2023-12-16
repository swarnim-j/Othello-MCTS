import torch
import torch.nn as nn
import torch.nn.functional as F

class OthelloNet(nn.Module):
    def __init__(self, game, args):
        super(OthelloNet, self).__init__()

        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args
        channels = args.num_channels

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, stride=1)
        self.conv4 = nn.Conv2d(channels, channels, 3, stride=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(channels)

        # Fully connected layers
        self.fc1 = nn.Linear(channels * (self.board_x - 4) * (self.board_y - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        # Output layers
        self.fc3 = nn.Linear(512, self.action_size)  # Policy head
        self.fc4 = nn.Linear(512, 1)  # Value head

    def forward(self, s):
        # Input reshaping
        s = s.view(-1, 1, self.board_x, self.board_y)

        # Convolutional layers with batch normalization and ReLU activation
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))

        # Flatten before fully connected layers
        s = s.view(-1, self.args.num_channels * (self.board_x - 4) * (self.board_y - 4))

        # Fully connected layers with dropout
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)

        # Policy and value heads
        pi = self.fc3(s)
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)