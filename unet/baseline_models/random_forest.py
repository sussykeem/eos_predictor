from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import torch
from eos_features import EOS_Features_Dataloader
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42, target='a'):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.feature_selector = None
        self.best_params = None
        self.original_feature_names = ['Molecular Weight', 'LogP', 'TPSA', 'Rotatable Bonds', 
                                     'H Bond Donors', 'H Bond Acceptors', 'Aromatic Rings', 
                                     'Num Rings', 'Atom Count']
        self.selected_feature_names = None
        self.target = target  # 'a' or 'b'
        self.target_index = 0 if target == 'a' else 1

    def train(self, dataloader, tune_hyperparams=False, select_features=False):
        # Extract data (single target)
        X_train, y_train = self._extract_data(dataloader)
        
        # Feature Selection (First step)
        if select_features:
            X_train = self._perform_feature_selection(X_train, y_train)
            
        # Hyperparameter Tuning (Uses selected features only)
        if tune_hyperparams:
            self._tune_hyperparameters(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)

    def _extract_data(self, dataloader):
        X, y = [], []
        for inputs, targets in dataloader:
            X.append(inputs.numpy())
            # Ensure y is always 2D (batch_size, 1)
            y.append(targets.numpy()[:, self.target_index:self.target_index+1])
        return np.vstack(X), np.vstack(y)

    def _perform_feature_selection(self, X, y):
        print(f"\n=== Feature Selection (Target: {self.target}) ===")
        # First fit to get importances
        selector_model = RandomForestRegressor(n_estimators=50, random_state=42)
        selector_model.fit(X, y.ravel())  # Use ravel() for single target
        
        # Plot all features' importance before selection
        self._plot_feature_importance(selector_model, self.original_feature_names, 
                                    f"Feature Importances for {self.target}")
        
        # Now perform actual selection
        self.feature_selector = SelectFromModel(selector_model, threshold='median')
        self.feature_selector.fit(X, y.ravel())
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_feature_names = [self.original_feature_names[i] for i in selected_indices]
        
        print(f"Selected features for {self.target}: {self.selected_feature_names}")
        return self.feature_selector.transform(X)

    def _plot_feature_importance(self, model, feature_names, title):
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title(title)
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

    def _tune_hyperparameters(self, X, y):
        print(f"\n=== Hyperparameter Tuning for {self.target} ===")
        param_grid = {
            'n_estimators': [100, 200, 400],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42),
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X, y.ravel())  # Use ravel() for single target
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        print(f"Best params for {self.target}: {self.best_params}")

    def evaluate(self, dataloader):
        X_test, y_test = self._extract_data(dataloader)
        
        if self.feature_selector is not None:
            X_test = self.feature_selector.transform(X_test)
        
        score = self.model.score(X_test, y_test.ravel())  # Use ravel() for single target
        print(f"\nTest R² Score for {self.target}: {score:.4f}")
        return score

    def predict(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if self.feature_selector is not None:
            x = self.feature_selector.transform(x.reshape(1, -1))
        return self.model.predict(x)

# Usage example
if __name__ == "__main__":
    features_data = EOS_Features_Dataloader(scale=True)

    # Initialize the model
    a_model = RandomForestModel(target='a')
    a_model.train(features_data.train, tune_hyperparams=True, select_features=True)
    a_model.evaluate(features_data.test)

    b_model = RandomForestModel(target='b')
    b_model.train(features_data.train, tune_hyperparams=True, select_features=True)
    b_model.evaluate(features_data.test)

    atom_name = 'Thiophene'
    features = np.array([84.143, 1.7481, 0.0, 0, 0, 1, 1, 1, 5])
    const = [17.21, 0.1058]

    input_scaled = features_data.in_scaler.transform(features.reshape(1,-1))

    a_pred = a_model.predict(input_scaled)
    b_pred = b_model.predict(input_scaled)

    a_pred = features_data.t_scaler.inverse_transform(np.array([a_pred[0], 0]).reshape(1,-1))
    b_pred = features_data.t_scaler.inverse_transform(np.array([0, b_pred[0]]).reshape(1,-1))

    print(f'{atom_name} Prediction: a: {a_pred[0][0]}, b: {b_pred[0][1]}')
    print(f'{atom_name} Actual: a: {const[0]:.4f}, b: {const[1]:.4f}')