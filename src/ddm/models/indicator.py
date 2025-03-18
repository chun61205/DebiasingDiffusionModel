import torch
import torch.nn as nn

class Indicator(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(int((4 * size * size) / 64), 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x