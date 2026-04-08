import copy
import sys

import numpy as np
import torch
from tqdm import tqdm


def reset_weights(m, seed):
    if hasattr(m, 'reset_parameters'):
        torch.manual_seed(seed)
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def train_opt(model_fun, opt_configs, criterion, loader, training_repetitions, epochs, device, seed_offset):
    results = {}

    for opt_name in opt_configs:
        model = model_fun.to(device)
        opt_fun = opt_configs[opt_name]
        optimizer = opt_fun(model.parameters())
        print(f"Training with optimizer {opt_name}...")
        loss_history = np.zeros((epochs, training_repetitions))
        best_loss = float('inf')
        best_model_state = None

        for rep in range(training_repetitions):
            model.apply(lambda m: reset_weights(m, seed=rep + seed_offset))

            pbar = tqdm(range(epochs), desc="Training")
            for epoch in pbar:
                epoch_loss = 0.0
                for x_batch, y_batch in loader:
                    optimizer.zero_grad()
                    y_pred = model(x_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * len(x_batch)
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model_state = copy.deepcopy(model.state_dict())

                epoch_loss /= len(loader.dataset)
                loss_history[epoch, rep] = epoch_loss
                if epoch % 10 == 0:
                    pbar.set_postfix(loss=f"{epoch_loss:.6f}")

        results[opt_name] = {
            "best_model_state": best_model_state,
            "opt_state": optimizer.state_dict(),
            "loss_histories": loss_history,
        }

    return results


def save_results(results, path):
    """Save results dict — contains only tensors/arrays, no class objects."""
    torch.save(results, path)


def load_and_reconstruct(path, model_fun, opt_configs, device, weights_only=True):
    """
    Load saved results and reconstruct ready-to-use models.

    Parameters
    ----------
    path       : path passed to torch.save() earlier
    model_fun  : the same callable used during training
    opt_configs: the same optimizer config dict used during training
    device     : device to map tensors onto
    weights_only: True (default) = safe mode, no class unpickling
    """
    # weights_only=True is the safe default in PyTorch >= 2.0
    raw = torch.load(path, map_location=device, weights_only=weights_only)

    reconstructed = {}
    for opt_name, entry in raw.items():
        model = model_fun.to(device)
        model.load_state_dict(entry["model_state"])
        model.eval()                          # ready for inference

        optimizer = opt_configs[opt_name](model.parameters())
        optimizer.load_state_dict(entry["opt_state"])

        reconstructed[opt_name] = {
            "model": model,
            "optimizer": optimizer,
            "loss_histories": entry["loss_histories"],
        }

    return reconstructed