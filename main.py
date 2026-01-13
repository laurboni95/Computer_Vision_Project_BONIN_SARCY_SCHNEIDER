from data_loader import load_data, Sample
from MLP import train_MLP_model
from GRU import train_GRU_model
import numpy as np
import torch

JAAD_Data = load_data("JAAD_obs16_ov0.8_H30.pkl")
PIE_Data = load_data("PIE_obs16_ov0.6_H30.pkl")

def get_test_data(samples: list[Sample], labels: list[np.int64]) -> tuple[list[list[float]], list[list[float]]]:
    inputs = [sample.bboxes for sample in samples if sample.split == "train"]
    outputs = [[float(value)] for i, value in enumerate(labels) if samples[i].split == "train"]
    return inputs, outputs

all_samples = JAAD_Data.samples + PIE_Data.samples
all_labels = list(JAAD_Data.labels) + list(PIE_Data.labels)

MLP_jaad = train_MLP_model(*get_test_data(JAAD_Data.samples, list(JAAD_Data.labels)))
MLP_pie = train_MLP_model(*get_test_data(PIE_Data.samples, list(PIE_Data.labels)))

torch.save(MLP_jaad.state_dict(), "MLP_JAAD.pth")
torch.save(MLP_pie.state_dict(), "MLP_PIE.pth")

GRU_jaad = train_GRU_model(*get_test_data(JAAD_Data.samples, list(JAAD_Data.labels)))
GRU_pie = train_GRU_model(*get_test_data(PIE_Data.samples, list(PIE_Data.labels)))

torch.save(GRU_jaad.state_dict(), "GRU_JAAD.pth")
torch.save(GRU_pie.state_dict(), "GRU_PIE.pth")