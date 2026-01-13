import torch.nn as nn
import torch
from torch import Tensor

def train_model(model: nn.Module, epochs: int, x_train: Tensor, y_train: Tensor, divide_lr: float=1):
    pos_weight = (y_train == 0).float().sum() / (y_train == 1).float().sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Not BCE because I have removed Sigmoid
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001/divide_lr)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        print(" Step {}, loss {}".format(_, loss.item()))
    print("final loss", loss.item())

    return model