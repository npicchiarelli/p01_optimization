import pickle as pkl
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

from of_pybind11_system import of_pybind11_system
from experimental_optimizers.soap_mods import SOAP
from ml_tools import multi_training

import matplotlib.pyplot as plt

import os
print(os.getcwd())
print(os.environ['LD_LIBRARY_PATH'])
# ---------------------------
# Connect to pybind system
# ------------------------- --
a = of_pybind11_system(["."])

# Mesh coordinates
X = a.getX()[0::3, 0]
Y = a.getX()[1::3, 0]
print(a.getX().shape)

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

# "Data" target (e.g., high-fidelity solution for training)
T_train_true = np.linalg.solve(A_mat_train, b_vec_train)  # (N, 1)
print("Training data setup complete. System matrix shape:", A_mat_train.shape)
print(T_train_true.shape)
print(T.shape)

# Input for training [x, y, s_train]
Input_train = torch.zeros(len(X), 3)
Input_train[:, 0] = torch.from_numpy(X).float()
Input_train[:, 1] = torch.from_numpy(Y).float()
Input_train[:, 2] = torch.from_numpy(S_train[:, 0]).float()

T_train_true = torch.from_numpy(T_train_true).float()

# ---------------------------
# Define the model
# ---------------------------

class LinearNN(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size//2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
# ---------------------------
# Define the physics-informed loss function
# ---------------------------
class PhysicsInformedLoss(nn.Module):
    def __init__(self, A_mat: np.ndarray, b_vec: np.ndarray, data_weight: float = 0.0, n_data_points: int = None, device=None):
        super().__init__()
        self.A = torch.from_numpy(A_mat).float().to(device)
        self.b = torch.from_numpy(b_vec).float().to(device)
        self.data_weight = data_weight
        self.n_data_points = n_data_points
        self.mse_loss = nn.MSELoss()

    def forward(self, T_pred: torch.Tensor, T_true: torch.Tensor) -> torch.Tensor:
        # Physics loss: ||A * T_pred - b||^2
        physics_residual = self.A @ T_pred - self.b
        physics_loss = torch.mean(physics_residual ** 2)

        # Data loss: ||T_pred - T_true||^2
        if self.data_weight > 0.0:
            n = T_pred.shape[0]
            k = self.n_data_points if self.n_data_points is not None else n

            idx = torch.randperm(n, device=T_pred.device)[:k]
            data_loss = torch.mean((T_pred[idx] - T_true[idx]) ** 2)
        else:
            data_loss = torch.tensor(0.0, device=T_pred.device)

        # Total loss with weighting
        total_loss = physics_loss + self.data_weight * data_loss
        return total_loss
    
# ---------------------------
# Set up training
# ---------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

loader = DataLoader(TensorDataset(Input_train.to(device), T_train_true.to(device)), batch_size=len(Input_train), shuffle=False)

criterion = PhysicsInformedLoss(A_mat_train, b_vec_train, data_weight=1e-5, n_data_points=1, device=device)
training_repetitions = 2
epochs = 1000

# optimizer_configs = {
#     "SOAP_no_projection":   lambda p: SOAP(p, lr=0.003, betas = (0.99, 0.999), precondition_1d=False, projection=False, precondition_frequency=100, weight_decay=0.0, shampoo_beta=0, normalize_grads=False), # For speed
#     "Adam":                 lambda p: Adam(p, lr=0.003, betas = (0.99, 0.999), amsgrad=False)
# }
# optimizer_configs = {
#     "SOAP_no_projection":   lambda p: SOAP(p, lr=0.003, betas = (0.99, 0.999), precondition_1d=False, projection=False, precondition_frequency=100, weight_decay=0.0, shampoo_beta=0, normalize_grads=False), # For speed
#     "Adam":                 lambda p: Adam(p, lr=0.003, betas = (0.99, 0.999), amsgrad=False),
#     "SOAP_with_projection":   lambda p: SOAP(p, lr=0.03, betas = (0.99, 0.999), precondition_1d=True, projection=True, precondition_frequency=1, weight_decay=0.0,), # For speed
# }

# optimizer_configs = {
#     "SOAP_with_projection_100":   lambda p: SOAP(p, lr=0.001, betas = (0.99, 0.999), precondition_1d=True, projection=True, precondition_frequency=100, weight_decay=0.),
#     "SOAP_with_projection_10":   lambda p: SOAP(p, lr=0.001, betas = (0.99, 0.999), precondition_1d=True, projection=True, precondition_frequency=10, weight_decay=0.),
# }

optimizer_configs = {
    # "SOAP":   lambda p: SOAP(p, lr=0.003, betas = (0.99, 0.999), precondition_1d=False, projection=True, precondition_frequency=10, weight_decay=0.0,), # For speed
    "SOAPW": lambda p: SOAP(p, lr=0.03, betas = (0.99, 0.999), precondition_1d=True, projection=True, precondition_frequency=5),
}

# print(Input_train.shape, T_train_true.shape)

results = multi_training.train_opt(LinearNN, optimizer_configs, criterion, loader, training_repetitions, epochs, device, seed_offset=611)

with open("outputs/opt_state/optimizer_results_pinn_weight_decay.pkl", "wb") as f:
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

    results_dir = "nn_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    pred_base = f"predictions_errors_pinn_{opt_name}"
    pred_dir = os.path.join(results_dir, pred_base)

    # ---------------------------
    # Export results to OpenFOAM
    # ---------------------------
    a.setT(T_test_pred.reshape(-1,))
    a.exportT(".", os.path.join(pred_dir, "1"), "T")  # predicted test

    a.setT(T_test_true.reshape(-1,))
    a.exportT(".", os.path.join(pred_dir, "2"), "T")  # true test

    a.setT(np.abs(T_test_pred - T_test_true).reshape(-1,))
    a.exportT(".", os.path.join(pred_dir, "3"), "T")  # absolute error map
    
    a.setT(np.abs((T_test_pred - T_test_true) / (T_test_true + 1e-30)).reshape(-1,))
    a.exportT(".", os.path.join(pred_dir, "4"), "T")  # relative error map
