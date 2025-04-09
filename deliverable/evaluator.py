import torch
import yaml
import traceback
from pathlib import Path
from build_model import Model  # Your wrapper class
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def evaluate_model(config, model_type, config_path=None, generation=None, log_dir=None):
    try:
        # Save temp YAML if needed
        if config_path is None:
            temp_path = Path(f".temp_config_{model_type.lower()}.yaml")
            with open(temp_path, "w") as f:
                yaml.dump(config, f)
            config_path = temp_path

        # Build model
        if model_type == "CNN":
            model = Model(config_path)
        elif model_type == "PKAN":
            encoder = Model("model_config/base_cnn.yaml")  # Use pretrained encoder or same config for now
            model = Model(config_path, encoder.network)
        else:
            # TODO: Add XGBoost/MLP fallback
            return float("inf")

        network = model.network
        dataloader = model.eos_dataloader
        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(network.parameters(), lr=config['optimizer']['learning_rate'])
        network.train()

        for _ in range(config["training"]["epochs"]):
            for x_batch, y_batch in dataloader.train_loader:
                x_batch, y_batch = x_batch.to(network.device), y_batch.to(network.device)
                optimizer.zero_grad()
                preds = network(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

        # Evaluate on validation/test set
        network.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in dataloader.test_loader:
                x_batch, y_batch = x_batch.to(network.device), y_batch.to(network.device)
                preds = network(x_batch)
                loss = criterion(preds, y_batch)
                val_losses.append(loss.item())

        avg_loss = np.mean(val_losses)

        # Optional: log to TensorBoard
        if log_dir:
            writer = SummaryWriter(log_dir=str(log_dir))
            writer.add_scalar(f"{model_type}/val_loss", avg_loss, generation)
            writer.close()

        # Optional: log to CSV
        if log_dir:
            with open(Path(log_dir) / "fitness_log.csv", "a") as f:
                f.write(f"{model_type},{generation},{avg_loss:.6f}\n")

        return avg_loss

    except Exception as e:
        print("⚠️ Evaluation failed:")
        traceback.print_exc()
        return float("inf")
