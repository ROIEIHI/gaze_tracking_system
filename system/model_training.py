"""
Model Training module for the Gaze Tracking System
Handles feature engineering, outlier detection, and model training
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from typing import Tuple, Dict
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from config import *

class GazeModelTrainer:
    """Main model training class"""
    
    def __init__(self, calibration_file: str, output_dir: str = None):
        """Initialize trainer with calibration data file and output directory"""
        self.calibration_file = calibration_file
        self.output_dir = output_dir if output_dir is not None else MODELS_DIR
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.model = None
        self.training_results = {}
    
    def load_data(self) -> bool:
        """Load calibration data from CSV file"""
        try:
            self.df = pd.read_csv(self.calibration_file)
            print(f"Loaded calibration data: {len(self.df)} samples")
            print(f"Features: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def detect_outliers(self) -> pd.DataFrame:
        """Detect and remove outliers from the data"""
        print("Detecting outliers...")
        
        original_count = len(self.df)
        
        if OUTLIER_METHOD == 'iqr':
            # IQR method for outlier detection
            for feature in FEATURE_COLUMNS:
                if feature in self.df.columns:
                    Q1 = self.df[feature].quantile(0.25)
                    Q3 = self.df[feature].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - IQR_MULTIPLIER * IQR
                    upper_bound = Q3 + IQR_MULTIPLIER * IQR
                    
                    # Remove outliers
                    mask = (self.df[feature] >= lower_bound) & (self.df[feature] <= upper_bound)
                    self.df = self.df[mask]
        
        elif OUTLIER_METHOD == 'zscore':
            # Z-score method for outlier detection
            for feature in FEATURE_COLUMNS:
                if feature in self.df.columns:
                    z_scores = np.abs(stats.zscore(self.df[feature]))
                    mask = z_scores < ZSCORE_THRESHOLD
                    self.df = self.df[mask]
        
        removed_count = original_count - len(self.df)
        print(f"Removed {removed_count} outliers ({removed_count/original_count*100:.1f}%)")
        print(f"Remaining samples: {len(self.df)}")
        
        return self.df
    
    def feature_engineering(self) -> pd.DataFrame:
        """Apply feature engineering to improve model performance"""
        print("Applying feature engineering...")
        
        # Calculate average iris positions
        self.df['avg_iris_x'] = (self.df['norm_L_x'] + self.df['norm_R_x']) / 2
        self.df['avg_iris_y'] = (self.df['norm_L_y'] + self.df['norm_R_y']) / 2
        
        # Create interaction terms
        self.df['yaw_avg_x_inter'] = self.df['yaw'] * self.df['avg_iris_x']
        self.df['pitch_avg_y_inter'] = self.df['pitch'] * self.df['avg_iris_y']
        
        # Update feature columns to include new engineered features
        engineered_features = FEATURE_COLUMNS + ['yaw_avg_x_inter', 'pitch_avg_y_inter']

        print(f"Added engineered features: ['yaw_avg_x_inter', 'pitch_avg_y_inter']")
        print(f"Total features: {len(engineered_features)}")
        
        return self.df, engineered_features
    
    def prepare_data(self, feature_columns: list) -> bool:
        """Prepare data for training"""
        print("Preparing data for training...")
        
        # Extract features and targets
        X = self.df[feature_columns].values
        y = self.df[['target_x', 'target_y']].values
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        self.y_train = self.y_train.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)
        
        # Add AGWN noise to training data only
        noise = np.random.normal(0, NOISE_LEVEL, self.y_train.shape)
        self.y_train += noise
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"Added {NOISE_LEVEL}px AGWN to training data targets")
        
        return True
    
    def train_model(self) -> bool:
        """Train RandomForest model with GridSearchCV"""
        print("Training Multi-Output RandomForest model...")
        print("Optimizing for Euclidean distance minimization...")
        
        # Custom scorer for Euclidean distance
        def euclidean_distance_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            euclidean_distances = np.sqrt(np.sum((y - y_pred)**2, axis=1))
            return -np.mean(euclidean_distances)  # Negative because GridSearchCV maximizes
        
        # Create base model
        base_model = RandomForestRegressor(random_state=RANDOM_STATE)
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            RANDOM_FOREST_PARAMS,
            cv=CV_FOLDS,
            scoring=euclidean_distance_scorer,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(self.X_train, self.y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return True
    
    def evaluate_model(self) -> Dict:
        """Evaluate model performance"""
        print("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        # R² scores
        r2_x = r2_score(self.y_test[:, 0], y_pred[:, 0])
        r2_y = r2_score(self.y_test[:, 1], y_pred[:, 1])
        avg_r2 = (r2_x + r2_y) / 2
        
        # RMSE
        rmse_x = np.sqrt(mean_squared_error(self.y_test[:, 0], y_pred[:, 0]))
        rmse_y = np.sqrt(mean_squared_error(self.y_test[:, 1], y_pred[:, 1]))
        
        # Euclidean distance
        euclidean_distances = np.sqrt(np.sum((self.y_test - y_pred)**2, axis=1))
        euclidean_rmse = np.sqrt(np.mean(euclidean_distances**2))
        
        # Percentage error (relative to screen diagonal)
        screen_diagonal = np.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)
        percentage_error = (euclidean_rmse / screen_diagonal) * 100
        
        # Cross-validation score - USE THE SAME SCORER AS GRID SEARCH
        def euclidean_distance_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            euclidean_distances = np.sqrt(np.sum((y - y_pred)**2, axis=1))
            return -np.mean(euclidean_distances)  # Negative because GridSearchCV maximizes
        
        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train,
            cv=CV_FOLDS, scoring=euclidean_distance_scorer  # CHANGED THIS LINE
        )
        
        # Store results
        self.training_results = {
            'euclidean_rmse': euclidean_rmse,
            'percentage_error': percentage_error,
            'avg_r2': avg_r2,
            'r2_x': r2_x,
            'r2_y': r2_y,
            'rmse_x': rmse_x,
            'rmse_y': rmse_y,
            'cv_score_mean': cv_scores.mean(),
            'cv_score_std': cv_scores.std(),
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test)
        }
        
        return self.training_results
    
    def print_results(self):
        """Print formatted training results"""
        results = self.training_results
        
        print("\n" + "="*80)
        print("MULTI-OUTPUT RANDOM FOREST MODEL PERFORMANCE REPORT")
        print("="*80)
        print(f"EUCLIDEAN DISTANCE ERROR: {results['euclidean_rmse']:.2f} pixels")
        print(f"PERCENTAGE ERROR: {results['percentage_error']:.2f}% (relative to {SCREEN_WIDTH}x{SCREEN_HEIGHT} diagonal)")
        print(f"Average R² Score: {results['avg_r2']:.4f}")
        print(f"X-coordinate R²: {results['r2_x']:.4f}")
        print(f"Y-coordinate R²: {results['r2_y']:.4f}")
        print(f"X-coordinate RMSE: {results['rmse_x']:.2f} pixels")
        print(f"Y-coordinate RMSE: {results['rmse_y']:.2f} pixels")
        print(f"Cross-validation Score (Euclidean): {results['cv_score_mean']:.4f} ± {results['cv_score_std']:.4f}")
        print(f"Training samples: {results['train_samples']}")
        print(f"Test samples: {results['test_samples']}")
        print("="*80)
    
    def save_model(self) -> str:
        """Save trained model and scaler"""
        if self.model is None or self.scaler is None:
            print("No model to save!")
            return ""
        
        # Generate filename 
        model_filename = "gaze_model.pkl"
        scaler_filename = "scaler.pkl"

        model_path = os.path.join(self.output_dir, model_filename)
        scaler_path = os.path.join(self.output_dir, scaler_filename)
        
        # Save model and scaler
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        print(f"Model saved: {model_path}")
        print(f"Scaler saved: {scaler_path}")
        
        # Save model info
        info_filename = "model_info.txt"
        info_path = os.path.join(self.output_dir, info_filename)
        
        with open(info_path, 'w') as f:
            f.write("Gaze Tracking Model Information\n")
            f.write("="*40 + "\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Calibration File: {os.path.basename(self.calibration_file)}\n")
            f.write(f"Model File: {model_filename}\n")
            f.write(f"Scaler File: {scaler_filename}\n")
            f.write(f"Features Used: {len(FEATURE_COLUMNS) + 4}\n")  # +4 for engineered features
            f.write("\nPerformance Metrics:\n")
            for key, value in self.training_results.items():
                f.write(f"{key}: {value}\n")
        
        # Return absolute path to ensure consistency
        return os.path.abspath(model_path)
    
    def train_complete_model(self) -> str:
        """Run complete model training pipeline"""
        print("Starting Model Training Pipeline")
        print("="*50)
        
        # Load data
        if not self.load_data():
            return ""
        
        # Detect outliers
        self.detect_outliers()
        
        # Feature engineering
        _, feature_columns = self.feature_engineering()
        
        # Prepare data
        if not self.prepare_data(feature_columns):
            return ""
        
        # Train model
        if not self.train_model():
            return ""
        
        # Evaluate model
        self.evaluate_model()
        
        # Print results
        self.print_results()
        
        # Save model
        model_path = self.save_model()
        
        return model_path

def train_model_from_file(calibration_file: str) -> str:
    """Convenience function to train model from calibration file"""
    trainer = GazeModelTrainer(calibration_file)
    return trainer.train_complete_model()
