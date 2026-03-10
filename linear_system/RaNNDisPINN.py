import numpy as np
import torch
from of_pybind11_system import of_pybind11_system

# ---------------------------
# Connect to pybind system
# ---------------------------
a = of_pybind11_system(["."])

# Mesh coordinates
X = a.getX()[0:576, 0]
Y = a.getX()[576:1152, 0]

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
# ---------------------------
# Randomized Neural Network features (ELM-style)
# ---------------------------
hidden_size = 128
ridge = 1e-6
activation = np.tanh
rng = np.random.default_rng(seed=12345)

# Input for training [x, y, s_train]
Input_train = torch.zeros(len(X), 3)
Input_train[:, 0] = torch.from_numpy(X).float()
Input_train[:, 1] = torch.from_numpy(Y).float()
Input_train[:, 2] = torch.from_numpy(S_train[:, 0]).float()

X_train_np = Input_train.detach().cpu().numpy()  # (N, 3)
N, D = X_train_np.shape

# Random weights & biases
R = rng.normal(size=(hidden_size, D))  # (H, D)
b = rng.normal(size=(hidden_size,))    # (H,)

# Hidden-layer matrix for training (all N points)
H_train = activation(X_train_np @ R.T + b)       # (N, H)

# Targets
Y_train = T_train_true                            # (N, 1)

# ---------------------------
# Hybrid loss with selective data term
#   Physics term: all N points
#   Data term: only 2 randomly chosen points
# ---------------------------
lambda_data = 1.0   # set to 0.0 to disable data term entirely
lambda_phys = 25

# Physics pieces over ALL points
AH_train = A_mat_train @ H_train                  # (N, H)

# Select exactly 2 data points for the data term
rng_sub = np.random.default_rng(seed=2024)        # set seed for reproducibility
idx2 = rng_sub.choice(N, size=5, replace=False)   # (2,)

H_data = H_train[idx2, :]                         # (2, H)
Y_data = Y_train[idx2, :]                         # (2, 1)

# Normal equations:
# (λd H_d^T H_d + λp (AH)^T (AH) + ridge I) W = λd H_d^T Y_d + λp (AH)^T b
HTH_d  = H_data.T @ H_data                        # (H, H)
AT_A   = AH_train.T @ AH_train                    # (H, H)
HTY_d  = H_data.T @ Y_data                        # (H, 1)
AT_b   = AH_train.T @ b_vec_train                 # (H, 1)

G = (lambda_data * HTH_d) + (lambda_phys * AT_A) + ridge * np.eye(hidden_size)
c = (lambda_data * HTY_d) + (lambda_phys * AT_b)
W = np.linalg.solve(G, c)                         # (H, 1)

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

# Hidden layer for testing and prediction
H_test = activation(Input_test.detach().cpu().numpy() @ R.T + b)  # (N, H)
Y_test_pred = H_test @ W                                          # (N, 1)

# ---------------------------
# Diagnostics
# ---------------------------
T_test_true = np.linalg.solve(A_mat_test, b_vec_test)

data_residual = np.linalg.norm(Y_test_pred - T_test_true) / np.linalg.norm(T_test_true)
phys_residual = np.linalg.norm(A_mat_test @ Y_test_pred - b_vec_test) / (np.linalg.norm(b_vec_test) + 1e-12)

print("lambda_data =", lambda_data, "| lambda_phys =", lambda_phys)
print("Picked indices for data term:", idx2.tolist())
print("Relative data residual (test)  =", data_residual)
print("Relative physics residual (test) =", phys_residual)

# ---------------------------
# Export results to OpenFOAM
# ---------------------------
a.setT(Y_test_pred.reshape(-1,))
a.exportT(".", "1", "T")  # predicted test

a.setT(T_test_true.reshape(-1,))
a.exportT(".", "2", "T")  # true test

a.setT(np.abs(Y_test_pred - T_test_true).reshape(-1,))
a.exportT(".", "3", "T")  # error map
