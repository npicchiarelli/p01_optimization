import argparse
import os
import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from experimental_optimizers.soap_mods import SOAP
from ml_tools import multi_training
from models.models import LinearNN, VectorModel
from of_pybind11_system import of_pybind11_system

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--save", action="store_true", help="Flag to enable data saving")
args = parser.parse_args()

# ---------------------------
# Connect to pybind system
# ------------------------- --
a = of_pybind11_system(["."])

# Mesh coordinates
X = a.getX()[0::3, 0]
Y = a.getX()[1::3, 0]

# Get temperature and source arrays from OF
T = a.getT()
S = a.getS()
# ---------------------------
# TRAINING data setup
# ---------------------------
S_train = np.zeros_like(S)
for i in range(len(X)):
    S_train[i, 0] = np.sin(np.pi * X[i]) * np.sin(np.pi * Y[i])

# System for training case
A_mat_train = a.get_system_matrix(T, S_train).toarray()   # (N, N)
b_vec_train = a.get_rhs(T, S_train).reshape(-1, 1)        # (N, 1)
print("System matrix and RHS vector for training case obtained. Shapes:", A_mat_train.shape, b_vec_train.shape)
print("System matrix condition number:", np.linalg.cond(A_mat_train))

# "Data" target (e.g., high-fidelity solution for training)
T_train_true = np.linalg.solve(A_mat_train, b_vec_train)  # (N, 1)
print("Training data setup complete. System matrix shape:", A_mat_train.shape)

# Input for training [x, y, s_train]
Input_train = torch.zeros(len(X), 3)
Input_train[:, 0] = torch.from_numpy(X).float()
Input_train[:, 1] = torch.from_numpy(Y).float()
Input_train[:, 2] = torch.from_numpy(S_train[:, 0]).float()

T_train_true = torch.from_numpy(T_train_true).float()

# ---------------------------
# Define the physics-informed loss function
# ---------------------------
class PhysicsInformedLoss(nn.Module):
    def __init__(self, A_mat: np.ndarray, b_vec: np.ndarray, data_weight: float = 0.0, physics_weight: float = 1.0, data_points_indices: np.ndarray = None, device=None):
        super().__init__()
        self.A = torch.from_numpy(A_mat).float().to(device)
        self.b = torch.from_numpy(b_vec).float().to(device)
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.data_points_indices = data_points_indices
        self.mse_loss = nn.MSELoss()

    def forward(self, T_pred: torch.Tensor, T_true: torch.Tensor) -> torch.Tensor:
        # Physics loss: ||A * T_pred - b||^2
        physics_residual = self.A @ T_pred - self.b
        physics_loss = torch.mean(physics_residual ** 2)

        # Data loss: ||T_pred - T_true||^2
        if self.data_weight > 0.0:
            data_loss = torch.mean((T_pred[self.data_points_indices] - T_true[self.data_points_indices]) ** 2)

        else:
            data_loss = torch.tensor(0.0, device=T_pred.device)

        # Total loss with weighting
        total_loss = self.physics_weight * physics_loss + self.data_weight * data_loss
        return total_loss
    
# ---------------------------
# Set up training
# ---------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

loader = DataLoader(TensorDataset(Input_train.to(device), T_train_true.to(device)), batch_size=len(Input_train), shuffle=False)

n_data_points = 1
np.random.seed(69)  # For reproducibility of data point selection
data_points_indices = np.random.choice(len(Input_train), size=n_data_points, replace=False)
print("Data points indices for loss:", data_points_indices)

criterion = PhysicsInformedLoss(A_mat_train, b_vec_train, data_weight=1.0, physics_weight=1e5, data_points_indices=data_points_indices, device=device)
training_repetitions = 1
epochs = 10

optimizer_configs = {
    "SGD": lambda p: torch.optim.SGD(p, lr=0.03),
    # "SOAPW": lambda p: SOAP(p, lr=0.03, betas = (0.99, 0.999), precondition_1d=True, projection=True, precondition_frequency=5),
}

results = multi_training.train_opt(LinearNN(hidden_size=64), optimizer_configs, criterion, loader, training_repetitions, epochs, device, seed_offset=709)

if args.save:
    with open("outputs/opt_state/optimizer_results_test.pkl", "wb") as f:
        pkl.dump(results, f)

# ---------------------------
# TEST data setup (can differ from training)
# ---------------------------
S_test = np.zeros_like(S)
for i in range(len(X)):
    S_test[i, 0] = np.sin(np.pi * X[i]) * np.sin(np.pi * Y[i])

A_mat_test = a.get_system_matrix(T, S_test).toarray()
b_vec_test = a.get_rhs(T, S_test).reshape(-1, 1)

# Input for testing [x, y, s_test]
Input_test = torch.zeros(len(X), 3)
Input_test[:, 0] = torch.from_numpy(X).float()
Input_test[:, 1] = torch.from_numpy(Y).float()
Input_test[:, 2] = torch.from_numpy(S_test[:, 0]).float()

T_test_true = np.linalg.solve(A_mat_test, b_vec_test)

for opt_name in results:
    model = results[opt_name]["model"]

    model.eval()
    with torch.no_grad():
        T_test_pred = model(Input_test.to(device)).cpu().numpy()

    if args.save:
        results_dir = "nn_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        pred_base = f"predictions_errors_linearmodel_mix{opt_name}"
        pred_dir = os.path.join(results_dir, pred_base)

        # ---------------------------
        # Export results to OpenFOAM
        # ---------------------------
        print(T_test_pred)
        a.setT(T_test_pred.reshape(-1,))
        a.exportT(".", os.path.join(pred_dir, "1"), "T")  # predicted tests

        a.setT(T_test_true.reshape(-1,))
        a.exportT(".", os.path.join(pred_dir, "2"), "T")  # true test

        a.setT(np.abs(T_test_pred - T_test_true).reshape(-1,))
        a.exportT(".", os.path.join(pred_dir, "3"), "T")  # absolute error map
        
        a.setT((np.abs((T_test_pred - T_test_true) ) / (np.abs(T_test_pred) + 1e-10)).reshape(-1,))
        a.exportT(".", os.path.join(pred_dir, "4"), "T")  # relative error map

    print(f"Mean T {opt_name}: {(np.mean(np.abs(T_test_true))):.6f}")
    print(f"Mean T {opt_name}: {(np.mean(np.abs(T_test_pred))):.6f}")
    print(f"Mean absolute error for {opt_name}: {(np.mean(np.abs((T_test_pred - T_test_true) ))):.6f}")
    print(f"Mean relative error for {opt_name}: {(np.mean(np.abs((T_test_pred-T_test_true))/np.abs((T_test_pred)))):.6f}")
    print(min(abs(T_test_true)), max(abs(T_test_true)))
    print(min(abs(T_test_pred-T_test_true)), max(abs(T_test_pred-T_test_true)))
    print(max(np.abs((T_test_pred-T_test_true))/(T_test_pred)), np.argmax(np.abs((T_test_pred-T_test_true))/(T_test_pred)))
