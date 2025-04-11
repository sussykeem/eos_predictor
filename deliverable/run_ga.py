import random
from pathlib import Path
from ga_utils import (
    generate_random_config,
    mutate_config,
    crossover_configs,
    save_config
)

from multiprocessing import Pool, cpu_count
from evaluator import evaluate_model
import torch
import torch.nn as nn


# === Settings ===
MODEL_TYPES = ["CNN"]  # "PKAN", "MLP", "XGBoost"]
POPULATION_SIZE = 10
NUM_GENERATIONS = 5
ELITE_RATIO = 0.2
MUTATION_RATE = 0.3
SAVE_DIR = Path("ga_runs")
SAVE_DIR.mkdir(exist_ok=True)

def validate_model(config, model_type):
    """
    Validates the model architecture by ensuring that layer dimensions match correctly.
    """

    if model_type == "CNN":
        # Check Conv Layers
        in_channels = 3  # Assume input is RGB images (3 channels)
        for layer in config.get("conv_layers", []):
            out_channels = layer["out_channels"]
            kernel_size = layer["kernel_size"]
            stride = layer.get("stride", 1)
            padding = layer.get("padding", 0)
            
            # Dummy input to test dimensions
            dummy_input = torch.randn(1, in_channels, 224, 224)  # Example image size
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
            
            # Forward pass through the convolution layer
            try:
                dummy_output = conv_layer(dummy_input)
                in_channels = out_channels  # Set output channels as the next input
            except Exception as e:
                raise ValueError(f"Invalid Conv Layer configuration: {str(e)}")

        # Check Fully Connected Layers
        # For the fully connected layers, you need to flatten the output from the conv layers
        flattened_size = in_channels * (224 // (2**len(config["conv_layers"])))**2  # Adjust based on pooling layers
        for layer in config.get("fc_layers", []):
            units = layer["units"]
            dropout = layer.get("dropout", 0.0)
            activation = layer["activation"]
            
            # Dummy input for FC layer
            dummy_fc_input = torch.randn(1, flattened_size)
            fc_layer = nn.Linear(flattened_size, units)
            
            try:
                dummy_output = fc_layer(dummy_fc_input)
                flattened_size = units  # Update flattened size for subsequent layers
            except Exception as e:
                raise ValueError(f"Invalid FC Layer configuration: {str(e)}")

    # Add checks for other model types like "PKAN", "MLP", etc., as needed
    return True

# === Genetic Algorithm ===
def parallel_evaluate(configs):
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(evaluate_model, configs)
    return results

def save_population(population, generation_dir):
    for i, (config, model_type, score) in enumerate(population):
        filepath = generation_dir / f"{model_type.lower()}_{i}_score_{score:.4f}.yaml"
        save_config(config, filepath)

# === Main GA Loop ===
def run_genetic_algorithm():
    population = []

    print("🚀 Initializing population...")
    for _ in range(POPULATION_SIZE):
        model_type = random.choice(MODEL_TYPES)
        config = generate_random_config(model_type)

         # Validate the generated model configuration
        try:
            if validate_model(config, model_type):
                score = evaluate_model(config, model_type)
                population.append((config, model_type, score))
            else:
                print(f"Invalid configuration for {model_type}, skipping...")
        except ValueError as e:
            print(f"Error validating model: {e}")

    for gen in range(NUM_GENERATIONS):
        print(f"\n🧬 Generation {gen+1}")

        # Sort by score (lower is better)
        population.sort(key=lambda x: x[2])
        generation_dir = SAVE_DIR / f"generation_{gen+1}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        save_population(population, generation_dir)

        # Select elites
        num_elites = int(ELITE_RATIO * POPULATION_SIZE)
        elites = population[:num_elites]

        # Generate new population
        new_population = elites[:]
        while len(new_population) < POPULATION_SIZE:
            parent1, model_type1, _ = random.choice(elites)
            parent2, model_type2, _ = random.choice(elites)

            # Ensure same model type
            if model_type1 != model_type2:
                continue

            child = crossover_configs(parent1, parent2, model_type1)
            child = mutate_config(child, model_type1, mutation_rate=MUTATION_RATE)

            # Create tuple for parallel evaluation (config, model_type, config_path, generation, log_dir)
            evaluate_job = (child, model_type1, None, gen+1, SAVE_DIR)
            new_population.append((child, model_type1, evaluate_model(evaluate_job)))

        population = new_population

    print("✅ Genetic algorithm complete.")

if __name__ == "__main__":
    run_genetic_algorithm()
