from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import torch
import os
import sys
from eos_features import EOS_Features_Dataloader
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from joblib import dump, load

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(sys.prefix, "Library", "plugins", "platforms")

class RandomForestModel:
    def __init__(self, data=None, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.data = data
        self.feature_selector = None
        self.best_params = None
        self.original_feature_names = ['Molecular Weight', 'LogP', 'TPSA', 'Rotatable Bonds', 
                                     'H Bond Donors', 'H Bond Acceptors', 'Aromatic Rings', 
                                     'Num Rings', 'Atom Count']
        self.selected_feature_names = None

    def train(self, tune_hyperparams=False, select_features=False):
        # Extract data (single target)
        X_train, y_train = self._extract_data(self.data.train)
        
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
            y.append(targets.numpy())
        return np.vstack(X), np.vstack(y)

    def _perform_feature_selection(self, X, y):
        print(f"\n=== Feature Selection ===")
        # First fit to get importances
        selector_model = RandomForestRegressor(n_estimators=50, random_state=42)
        selector_model.fit(X, y)
        
        # Plot all features' importance before selection
        self._plot_feature_importance(selector_model, self.original_feature_names, 
                                    f"Feature Importances")
        
        # Now perform actual selection
        self.feature_selector = SelectFromModel(selector_model, threshold='median')
        self.feature_selector.fit(X, y)
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_feature_names = [self.original_feature_names[i] for i in selected_indices]
        
        print(f"Selected features: {self.selected_feature_names}")
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
        print(f"\n=== Hyperparameter Tuning ===")
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
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        print(f"Best params: {self.best_params}")

    def evaluate(self):
        X_test, y_test = self._extract_data(self.data.test)
        
        if self.feature_selector is not None:
            X_test = self.feature_selector.transform(X_test)
        
        score = self.model.score(X_test, y_test)
        print(f"\nTest R² Score: {score:.4f}")
        return score

    def predict(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if self.feature_selector is not None:
            x = self.feature_selector.transform(x)
        return self.data.t_scaler.inverse_transform(self.model.predict(x))
    
    def save_model(self, file_path="base_model_weights/random_forest.pth"):
        """ Save the model state dictionary to a file """
        dump(self, file_path)
        print(f"Model saved to {file_path}")

    def test_model(self):

        inputs, targets = self._extract_data(self.data.test)
        preds = self.predict(inputs)
        outputs = self.data.t_scaler.inverse_transform(preds)
        targets = self.data.t_scaler.inverse_transform(targets)

        a_pred = outputs[:,0]
        b_pred = outputs[:,1]

        a_stack = np.stack([targets[:,0], a_pred])
        b_stack = np.stack([targets[:,1], b_pred])

        return a_stack, b_stack

# # Usage example
# if __name__ == "__main__":
#     features_data = EOS_Features_Dataloader(scale=True)

#     # Initialize the model
#     model = RandomForestModel(features_data)
#     model.train(tune_hyperparams=True, select_features=True)
#     model.evaluate()

#     model.test_model()

#     model.save_model()