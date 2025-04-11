# ga/hyperparam_space.py

HYPERPARAM_SPACES = {
    "CNN": {
        "optimizer": ["Adam", "SGD"],
        "learning_rate": [1e-4, 1e-3, 1e-2],
        "batch_size": [16, 32, 64],
        "epochs": [10, 20],
        "conv_out_channels": [32, 64, 128],
        "conv_kernel_size": [3, 5],
        "fc_units": [64, 128, 256],
        "fc_dropout": [0.0, 0.1, 0.2],
        "activation": ["ReLU", "LeakyReLU"]
    },
    "PKAN": {
        "optimizer": ["Adam", "SGD"],
        "learning_rate": [1e-4, 1e-3, 1e-2],
        "batch_size": [64, 128, 256],
        "epochs": [10, 20],
        "fc_units": [64, 128, 256],
        "fc_dropout": [0.1, 0.2, 0.3],
        "activation": ["ReLU", "Sigmoid"],
        "num_kernels": [8, 10, 12]
    },
    "MLP": {
        "optimizer": ["Adam", "SGD"],
        "learning_rate": [1e-4, 1e-3, 1e-2],
        "batch_size": [16, 32, 64],
        "epochs": [10, 20],
        "fc_units": [64, 128, 256],
        "fc_dropout": [0.0, 0.1, 0.2],
        "activation": ["ReLU", "Tanh", "Sigmoid"]
    },
    "XGBoost": {
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 6, 9],
        "n_estimators": [100, 200, 300],
        "subsample": [0.5, 0.7, 1.0],
        "colsample_bytree": [0.5, 0.7, 1.0]
    }
}
