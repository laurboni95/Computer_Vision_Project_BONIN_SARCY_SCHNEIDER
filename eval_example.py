from MLP import MLP
from GRU import GRU
import torch
from data_loader import load_data, Sample
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Select the model with these two constants
model_type = "GRU"
model_name = "JAAD"

my_loaded_model = MLP() if model_type == "MLP" else GRU()
weight = torch.load("{}_{}.pth".format(model_type, model_name))
my_loaded_model.load_state_dict(weight)
my_loaded_model.eval()

JAAD_Data = load_data("JAAD_obs16_ov0.8_H30.pkl")
PIE_Data = load_data("PIE_obs16_ov0.6_H30.pkl")

def get_val_data(samples: list[Sample], labels: list[np.int64]) -> tuple[list[list[float]], list[list[float]]]:
    inputs = [sample.bboxes for sample in samples if sample.split != "train"]
    outputs = [[float(value)] for i, value in enumerate(labels) if samples[i].split != "train"]
    return inputs, outputs

x_train, y_train = get_val_data(JAAD_Data.samples, JAAD_Data.labels)

if model_type == "MLP":
    Y_in = torch.tensor(y_train)
    not_flatten_tensor = torch.tensor(x_train)
    X_in = not_flatten_tensor.view(len(x_train), -1)
else:
    Y_in = torch.tensor(y_train)
    X_in = torch.tensor(x_train)

with torch.no_grad():
    probas = my_loaded_model(X_in)
    predictions_finales = (probas > 0.5).float()
    
y_vrai_np = Y_in.numpy()
y_pred_np = predictions_finales.numpy()

acc = accuracy_score(y_vrai_np, y_pred_np)
print(f"Précision globale (Accuracy) : {acc * 100:.2f}%")
    
cm = confusion_matrix(y_vrai_np, y_pred_np)
print("\nMatrice de confusion :")
print(cm)
    
print("\nRapport complet :")
print(classification_report(y_vrai_np, y_pred_np, target_names=['Ne traverse pas', 'Traverse']))