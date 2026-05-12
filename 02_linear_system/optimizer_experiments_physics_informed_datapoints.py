import argparse
import os
import pickle as pkl
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from smithers.io.openfoam import FoamMesh
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, TensorDataset

from experimental_optimizers.soap_mods import SOAP
from ml_tools import multi_training
from models.models import LinearNN, VectorModel, MatrixModel, BiasOnly
from of_pybind11_system import of_pybind11_system

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--save",
                    action="store_true",
                    help="Flag to enable data saving")

parser.add_argument("-e", "--exp_name",
                    type=str,
                    help="Experiment name")

args = parser.parse_args()

if args.save:
    if not args.exp_name:
        parser.error("--exp_name is required when --save is used")

print("Starting physics-informed training experiment:", args.exp_name if args.exp_name else "No name provided")

# ---------------------------
# Saving directories setup
# ---------------------------

# os.chdir("thickerCase")
if args.save:
    outputs_dir = "outputs"
    base_name = f"{args.exp_name}"
    res_path = os.path.join(outputs_dir, base_name)

    if os.path.exists(res_path):
        i = 1
        while True:
            new_path = os.path.join(outputs_dir, f"{base_name}_{i:02d}")
            if not os.path.exists(new_path):
                res_path = new_path
                break
            i += 1

    os.makedirs(res_path, exist_ok=True)
    print("Data will be saved to:", res_path)

# ---------------------------
# Import OpenFOAM mesh data
# ---------------------------
mesh = FoamMesh("./thickerCase")
print("Mesh loaded. Number of cells:", mesh.num_cell+1)
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
    def __init__(self, A_mat: np.ndarray, b_vec: np.ndarray, data_weight: float = 0.0, physics_weight: float = 1.0, data_points_indices: np.ndarray = None, norm: str = "l2", device="cpu"):
        super().__init__()
        self.A = torch.from_numpy(A_mat).float().to(device)
        self.b = torch.from_numpy(b_vec).float().to(device)
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.data_points_indices = data_points_indices
        self.norm = norm
        if self.norm == "l2":
            self.base_loss = nn.MSELoss()
        elif self.norm == "l1":
            self.base_loss = nn.L1Loss()
        else:
            raise ValueError("Unsupported norm type. Use 'l1' or 'l2'.")

    def forward(self, T_pred: torch.Tensor, T_true: torch.Tensor) -> torch.Tensor:
        if self.physics_weight > 0.0:
            # Physics residual: A * T_pred - b
            physics_residual = self.A @ T_pred - self.b
            if self.norm == "l2":
                physics_loss = torch.mean(physics_residual ** 2)
            elif self.norm == "l1":
                physics_loss = torch.mean(torch.abs(physics_residual))

        else:
            physics_loss = torch.tensor(0.0, device=T_pred.device)

        if self.data_weight > 0.0:
            if self.norm == "l2":
                data_loss = torch.mean((T_pred.reshape(-1, 1)[self.data_points_indices] - T_true.reshape(-1, 1)[self.data_points_indices]) ** 2)
            elif self.norm == "l1":
                data_loss = torch.mean(torch.abs(T_pred.reshape(-1, 1)[self.data_points_indices] - T_true.reshape(-1, 1)[self.data_points_indices]))

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
# print(torch.unsqueeze(Input_train[:,2]).shape)
# loader = DataLoader(TensorDataset(Input_train[:,2].unsqueeze(0).to(device), T_train_true.unsqueeze(0).to(device)), batch_size=1, shuffle=False)

dp_repetitions = 10
n_data_points = 5
np.random.seed(22)  # For reproducibility of data point selection
for _ in range(dp_repetitions):
    data_points_indices = np.random.choice(len(Input_train), size=n_data_points, replace=False)
    mask = ([mesh.is_cell_on_boundary(j,b'top') or mesh.is_cell_on_boundary(j,b'bottom') or mesh.is_cell_on_boundary(j,b'left') or mesh.is_cell_on_boundary(j,b'right') for j in data_points_indices])

    # data_points_indices = np.array([0])
    print("Data points indices for loss:", data_points_indices)

    criterion = PhysicsInformedLoss(A_mat_train, b_vec_train, data_weight=1.0, physics_weight=1e5, data_points_indices=data_points_indices, norm="l2", device=device)
    loss_dict = {
        "data_weight": criterion.data_weight,
        "physics_weight": criterion.physics_weight,
        "data_points_indices": criterion.data_points_indices,
        "on_boundary": mask,
        "norm": criterion.norm
    }

    training_repetitions = 3
    epochs = 2000

    optimizer_configs = {
        "SOAPW": lambda p: SOAP(p, lr=0.03, betas = (0.99, 0.999), precondition_1d=True, projection=True, precondition_frequency=5),
        "AdamW": lambda p: AdamW(p, lr=0.03, betas=(0.99, 0.999)),
    }

    model = LinearNN(hidden_size=64)
    print("Model initialized with parameters:", sum(p.numel() for p in model.parameters()))
    print("Model architecture:", model)

    results = multi_training.train_opt(model, optimizer_configs, criterion, loader, training_repetitions, epochs, device, seed_offset=709)

    if args.save:
        dp_path = os.path.join(res_path, f"data_points_{n_data_points}_{data_points_indices[0]}")
        opt_results_path = os.path.join(dp_path, "opt_state")
        os.makedirs(opt_results_path)
        with open(os.path.join(opt_results_path, "opt_results.pkl"), "wb") as f:
            pkl.dump(results, f)
        with open(os.path.join(opt_results_path, "loss_dict.pkl"), "wb") as f:
            pkl.dump(loss_dict, f)

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
        model.load_state_dict(results[opt_name]["best_model_state"])

        model.eval()
        with torch.no_grad():
            # T_test_pred = model(Input_test[:,2].unsqueeze(0).to(device)).cpu().numpy()
            T_test_pred = model(Input_test.to(device)).cpu().numpy()

        # ---------------------------
        # Export performance metrics
        # ---------------------------

        # Reshape for compatibility
        T_test_true = T_test_true.reshape(np.size(T_test_true))
        T_test_pred = T_test_pred.reshape(np.size(T_test_pred))

        abs_err = np.abs(T_test_pred - T_test_true) # absolute error
        mae = np.mean(abs_err)
        relative_error = np.abs((T_test_pred - T_test_true)) / (np.abs(T_test_pred) + 1e-10) # relative error with small epsilon to avoid division by zero
        mre = np.mean(relative_error)
        normalized_error = abs_err / (np.mean(np.abs(T_test_pred)) + 1e-10) # normalized error
        mne = np.mean(normalized_error)

        if args.save:
            nn_results_dir = os.path.join(dp_path, "nn_results", f"{opt_name}")
            os.makedirs(nn_results_dir)

            perform_dict = {
                "T_true": T_test_true,
                "T_pred": T_test_pred,
                "abs_error": abs_err,
                "mae": mae,
                "relative_error": relative_error,
                "mre": mre,
                "normalized_error": normalized_error,
                "mne" : mne,
                "datapoints_indices": data_points_indices
            }
            
            pkl.dump(perform_dict, open(os.path.join(nn_results_dir, "performance_metrics.pkl"), "wb"))

            # -------------------------------------------
            # Export OpenFOAM fields for visualization
            # -------------------------------------------

            for of_dir in ["system", "constant", "0"]:
                shutil.copytree(os.path.join(of_dir), os.path.join(nn_results_dir, of_dir))

            a.setT(T_test_pred.reshape(-1,))
            a.exportT(".", os.path.join(nn_results_dir, "1"), "T")  # predicted tests

            a.setT(T_test_true.reshape(-1,))
            a.exportT(".", os.path.join(nn_results_dir, "2"), "T")  # true test

            a.setT(np.abs(T_test_pred - T_test_true).reshape(-1,))
            a.exportT(".", os.path.join(nn_results_dir, "3"), "T")  # absolute error map
            
            a.setT((np.abs((T_test_pred - T_test_true) ) / (np.abs(T_test_pred) + 1e-10)).reshape(-1,))
            a.exportT(".", os.path.join(nn_results_dir, "4"), "T")  # relative error map

        print(f"Mean T true {opt_name}: {(np.mean(np.abs(T_test_true))):.6f}")
        print(f"Mean T predicted {opt_name}: {(np.mean(np.abs(T_test_pred))):.6f}")
        print(f"Mean absolute error for {opt_name}: {mae:.6f}")
        print(f"Mean relative error for {opt_name}: {mre:.6f}")
        print(f"Mean normalized error for {opt_name}: {mne:.6f}")

        print(f"Min absolute T_true for {opt_name}: {np.min(np.abs(T_test_true)):.6f}")
        print(f"Max absolute T_true for {opt_name}: {np.max(np.abs(T_test_true)):.6f}")
        print(f"Min absolute T_pred-T_true for {opt_name}: {np.min(np.abs(T_test_pred-T_test_true)):.6f}")
        print(f"Max absolute T_pred-T_true for {opt_name}: {np.max(np.abs(T_test_pred-T_test_true)):.6f}")
        print(f"Shape relative error for {opt_name}: {relative_error.shape}")
        print(f"Max relative error for {opt_name}: {np.max(relative_error):.6f}")
        print(f"Argmax relative error for {opt_name}: {np.argmax(relative_error)}")
