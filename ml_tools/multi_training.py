import copy
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

    for i, opt_name in enumerate(opt_configs):
        model = model_fun().to(device)
        opt_fun = opt_configs[opt_name]
        optimizer = opt_fun(model.parameters())
        print(f"Training with optimizer {opt_name}...")
        loss_history = np.zeros((epochs,training_repetitions))
        best_loss = float('inf')
        
        for rep in range(training_repetitions):
            model.apply(lambda m: reset_weights(m, seed= rep + seed_offset))  # reset weights with different seed for each repetition

            pbar = tqdm(range(epochs), desc="Training",)
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

                epoch_loss /= len(loader.dataset)  # average loss over dataset
                loss_history[epoch, rep] = epoch_loss  # save loss for this epoch
                if (epoch) % 10 == 0:
                    pbar.set_postfix(loss=f"{epoch_loss:.6f}")
        model.load_state_dict(best_model_state)
        results[opt_name] = {"model": model, "opt_state": optimizer.state_dict(), "loss_histories": loss_history}
    return results