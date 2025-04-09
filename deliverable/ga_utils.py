import yaml
import random
import copy
from pathlib import Path
from hyperparam_space import HYPERPARAM_SPACES

CONFIG_DIR = Path("model_config")

def load_base_config(model_type):
    base_file = CONFIG_DIR / f"base_{model_type.lower()}.yaml"
    with open(base_file, "r") as f:
        return yaml.safe_load(f)

def save_config(config, filepath):
    with open(filepath, "w") as f:
        yaml.dump(config, f)

def sample_from_space(space):
    return random.choice(space) if isinstance(space, list) else space

def generate_random_config(model_type):
    base = load_base_config(model_type)
    search_space = HYPERPARAM_SPACES[model_type]

    # CNN Example
    if model_type == "CNN":
        for layer in base.get("conv_layers", []):
            layer["out_channels"] = sample_from_space(search_space["conv_out_channels"])
            layer["kernel_size"] = sample_from_space(search_space["conv_kernel_size"])
            layer["activation"] = sample_from_space(search_space["activation"])

        for layer in base.get("fc_layers", []):
            layer["units"] = sample_from_space(search_space["fc_units"])
            layer["dropout"] = sample_from_space(search_space["fc_dropout"])
            layer["activation"] = sample_from_space(search_space["activation"])

    elif model_type == "PKAN" or model_type == "MLP":
        for layer in base.get("layers", []):
            if layer["type"] == "FC":
                layer["units"] = sample_from_space(search_space["fc_units"])
                layer["dropout"] = sample_from_space(search_space["fc_dropout"])
                layer["activation"] = sample_from_space(search_space["activation"])
            elif layer["type"] == "KAN":
                layer["num_kernels"] = sample_from_space(search_space["num_kernels"])

    elif model_type == "XGBoost":
        for param, options in search_space.items():
            base["xgboost_params"][param] = sample_from_space(options)

    # Global values
    if "training" in base:
        base["training"]["batch_size"] = sample_from_space(search_space["batch_size"])
        base["training"]["epochs"] = sample_from_space(search_space["epochs"])
    if "optimizer" in base:
        base["optimizer"]["type"] = sample_from_space(search_space["optimizer"])
        base["optimizer"]["learning_rate"] = sample_from_space(search_space["learning_rate"])

    return base

def mutate_config(config, model_type, mutation_rate=0.2):
    search_space = HYPERPARAM_SPACES[model_type]
    new_config = copy.deepcopy(config)

    def maybe_mutate(key, options):
        if random.random() < mutation_rate:
            return sample_from_space(options)
        return None

    if model_type == "CNN":
        for layer in new_config.get("conv_layers", []):
            if (val := maybe_mutate("out_channels", search_space["conv_out_channels"])) is not None:
                layer["out_channels"] = val
        for layer in new_config.get("fc_layers", []):
            if (val := maybe_mutate("units", search_space["fc_units"])) is not None:
                layer["units"] = val

    elif model_type in {"PKAN", "MLP"}:
        for layer in new_config.get("layers", []):
            if layer["type"] == "FC":
                if (val := maybe_mutate("units", search_space["fc_units"])) is not None:
                    layer["units"] = val
                if (val := maybe_mutate("dropout", search_space["fc_dropout"])) is not None:
                    layer["dropout"] = val

    elif model_type == "XGBoost":
        for param in search_space:
            if (val := maybe_mutate(param, search_space[param])) is not None:
                new_config["xgboost_params"][param] = val

    return new_config

def crossover_configs(config1, config2, model_type):
    child = copy.deepcopy(config1)
    other = config2

    def choose(a, b):
        return random.choice([a, b])

    if model_type == "CNN":
        for i, layer in enumerate(child.get("conv_layers", [])):
            if i < len(other.get("conv_layers", [])):
                layer["out_channels"] = choose(layer["out_channels"], other["conv_layers"][i]["out_channels"])

    elif model_type in {"PKAN", "MLP"}:
        for i, layer in enumerate(child.get("layers", [])):
            if i < len(other.get("layers", [])) and layer["type"] == "FC":
                layer["units"] = choose(layer["units"], other["layers"][i]["units"])

    elif model_type == "XGBoost":
        for param in child["xgboost_params"]:
            child["xgboost_params"][param] = choose(child["xgboost_params"][param], other["xgboost_params"].get(param, child["xgboost_params"][param]))

    return child
