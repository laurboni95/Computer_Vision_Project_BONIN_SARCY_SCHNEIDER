import torch.nn as nn
import torch
from common_train import train_model

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1),

            # First, i used a sigmoid here but it was problematic with BCE loss
        )

    def forward(self, x):
        return self.network(x)
    
def train_MLP_model(x_train: list[list[list[float]]], y_train: list[list[float]]) -> MLP:
    Y_in = torch.tensor(y_train)
    not_flatten_tensor = torch.tensor(x_train)
    X_in = not_flatten_tensor.view(len(x_train), -1)

    model = MLP()
    epochs = 200

    return train_model(model, epochs, X_in, Y_in, 10)