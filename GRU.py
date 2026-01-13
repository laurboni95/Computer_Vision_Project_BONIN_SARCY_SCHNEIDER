import torch.nn as nn
import torch
from common_train import train_model

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        
        self.gru = nn.GRU(input_size=4, hidden_size=256, batch_first=True)
        self.linear = nn.Linear(256, 1)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.linear(hidden[0])
    
def train_GRU_model(x_train: list[list[list[float]]], y_train: list[list[float]]) -> GRU:
    Y_in = torch.tensor(y_train)
    X_in = torch.tensor(x_train)

    model = GRU()
    epochs = 100

    return train_model(model, epochs, X_in, Y_in)