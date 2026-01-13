from MLP import MLP
import torch

# Remplacez le nom du fichier .pth si nécessaire (ex: "MLP_JAAD.pth")
weight = torch.load("MLP_JAAD.pth", map_location="cpu")
model = MLP()
model.load_state_dict(weight)
model.eval()
print("Model loaded and set to eval")