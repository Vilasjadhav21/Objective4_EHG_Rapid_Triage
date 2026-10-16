"""
Machine learning classifier module for EHG rapid triage.

Provides model training, hyperparameter tuning, Stratified K-Fold CV,
and model persistence wrappers.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

from src.config import ModelConfig, MODELS_DIR


class EHGClassifier:
    """Classifier wrapper for EHG preterm rapid triage."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.scaler = StandardScaler()
        self.model = self._build_model(self.config.model_type)
        self.best_params_: Optional[Dict] = None

    def _build_model(self, model_type: str):
        """Instantiate classifier based on model type."""
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.config.random_state
            )
        elif model_type == 'svm':
            return SVC(
                kernel='rbf',
                probability=True,
                random_state=self.config.random_state,
                class_weight='balanced'
            )
        elif model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=self.config.random_state,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> "EHGClassifier":
        """Fit scaler and classifier model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict binary triage labels (1=Preterm, 0=Term)."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities for ROC-AUC evaluation."""
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)
        else:
            # Fallback decision function scaling to [0, 1]
            df = self.model.decision_function(X_scaled)
            probs = 1 / (1 + np.exp(-df))
            return np.vstack([1 - probs, probs]).T

    def cross_validate(
        self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Perform Stratified K-Fold cross validation and compute metrics.
        """
        skf = StratifiedKFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.random_state
        )

        scores = {'accuracy': [], 'roc_auc': []}

        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        for train_idx, val_idx in skf.split(X_arr, y):
            X_train, X_val = X_arr[train_idx], X_arr[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit inner fold
            scaler_fold = StandardScaler()
            X_train_s = scaler_fold.fit_transform(X_train)
            X_val_s = scaler_fold.transform(X_val)

            fold_model = self._build_model(self.config.model_type)
            fold_model.fit(X_train_s, y_train)

            acc = fold_model.score(X_val_s, y_val)
            scores['accuracy'].append(acc)

        return scores

    def tune_hyperparameters(
        self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray
    ) -> Dict:
        """Grid search hyperparameter tuning."""
        X_scaled = self.scaler.fit_transform(X)

        if self.config.model_type == 'random_forest':
            param_grid = {
                'n_estimators': self.config.rf_n_estimators,
                'max_depth': self.config.rf_max_depth
            }
        elif self.config.model_type == 'svm':
            param_grid = {'C': self.config.svm_c}
        else:
            param_grid = {}

        if param_grid:
            grid = GridSearchCV(
                self.model, param_grid, cv=self.config.n_splits, scoring='roc_auc'
            )
            grid.fit(X_scaled, y)
            self.model = grid.best_estimator_
            self.best_params_ = grid.best_params_
            return grid.best_params_

        return {}

    def save(self, filepath: Union[str, Path]):
        """Save model and scaler checkpoints."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'best_params': self.best_params_
        }
        joblib.dump(checkpoint, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "EHGClassifier":
        """Load model checkpoint from file."""
        checkpoint = joblib.load(filepath)
        instance = cls(config=checkpoint['config'])
        instance.model = checkpoint['model']
        instance.scaler = checkpoint['scaler']
        instance.best_params_ = checkpoint.get('best_params')
        return instance
