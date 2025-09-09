"""
Streamlined Multi-Output RandomForest Gaze Model Training
Optimized for Euclidean Distance Minimization
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import datetime
import os

def euclidean_distance_scorer(y_true, y_pred):
    """
    Custom scorer for calculating mean Euclidean distance between true and predicted (x, y) coordinates.
    
    Args:
        y_true: True (x, y) coordinates with shape (n_samples, 2)
        y_pred: Predicted (x, y) coordinates with shape (n_samples, 2)
        
    Returns:
        float: Negative mean Euclidean distance (negative because GridSearchCV maximizes)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate Euclidean distances for each sample
    euclidean_distances = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
    
    # Return negative mean distance (GridSearchCV maximizes, we want to minimize distance)
    return -np.mean(euclidean_distances)

def feature_engineer(df):
    """
    Apply feature engineering to the preprocessed calibration data.
    
    Args:
        df: DataFrame with preprocessed calibration data
        
    Returns:
        DataFrame with engineered features
    """
    print("Performing feature engineering...")
    
    # Create average normalized eye coordinates
    df['avg_norm_x'] = (df['norm_x_L'] + df['norm_x_R']) / 2
    df['avg_norm_y'] = (df['norm_y_L'] + df['norm_y_R']) / 2
    
    # Create interaction features
    df['x_yaw_interaction'] = df['avg_norm_x'] * df['yaw']
    df['y_pitch_interaction'] = df['avg_norm_y'] * df['pitch']
    
    print("Feature engineering complete. Features: ['avg_norm_x', 'avg_norm_y', 'yaw', 'pitch', 'roll', 'x_yaw_interaction', 'y_pitch_interaction']")
    
    return df

class GazeModelTrainer:
    """Streamlined trainer for multi-output XGBoost gaze tracking model."""
    
    def __init__(self):
        """Initialize the trainer."""
        self.model = None
        self.scaler = StandardScaler()
        self.training_history = {}
        
    def load_calibration_data(self, csv_file):
        """Load calibration data from CSV file."""
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded calibration data from {csv_file}")
            print(f"Data shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading calibration data: {str(e)}")
            return None
    
    def preprocess_data(self, df):
        """Preprocess the calibration data by removing outliers."""
        print("Preprocessing calibration data...")
        
        # Remove outliers using IQR method for each numeric column
        numeric_columns = ['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 'yaw', 'pitch', 'roll']
        
        for col in numeric_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    print(f"Found {len(outliers)} outliers in {col}")
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        print(f"Final preprocessed data shape: {df.shape}")
        return df
    
    def train_model(self, df, noise_level=5.0):
        """
        Train multi-output XGBoost model with Euclidean distance optimization.
        
        Args:
            df: Preprocessed DataFrame with calibration data
            noise_level: Standard deviation of Gaussian noise for data augmentation
            
        Returns:
            Trained model or None if training fails
        """
        print("--- Training Multi-Output XGBoost Model with Euclidean Distance Optimization ---")
        
        # Prepare features and targets
        feature_columns = ['avg_norm_x', 'avg_norm_y', 'yaw', 'pitch', 'roll', 'x_yaw_interaction', 'y_pitch_interaction']
        target_columns = ['target_x', 'target_y']
        
        X = df[feature_columns].values
        y = df[target_columns].values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target matrix shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Apply data augmentation to training set
        if noise_level > 0:
            print(f"Applying data augmentation to training targets...")
            noise = np.random.normal(0, noise_level, y_train.shape)
            y_train_augmented = y_train + noise
        else:
            y_train_augmented = y_train
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define parameter grid for RandomForestRegressor GridSearchCV
        param_grid = {
            'n_estimators': [300, 350, 400],
            'max_depth': [8, 10, 12],
            'min_samples_split': [1, 2, 3],
            'min_samples_leaf': [1, 2, 3],
        }
        
        # Create multi-output RandomForest regressor
        model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        )
        
        # Create custom scorer
        scorer = make_scorer(euclidean_distance_scorer, greater_is_better=True)
        
        # Perform GridSearchCV with custom scorer
        print("Training multi-output RandomForest model with GridSearchCV using Euclidean distance scorer...")
        grid_search = GridSearchCV(
            model,
            param_grid,
            scoring=scorer,
            cv=3,  # Reduced CV folds for faster training with RandomForest
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train_augmented)
        
        # Get best model
        self.model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"Best model parameters: {best_params}")
        print(f"Best CV score (negative mean Euclidean distance): {best_score:.4f}")
        
        # Evaluate on test set
        print("Evaluating model on clean test set...")
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate Euclidean distance error
        euclidean_distances = np.sqrt(np.sum((y_test - y_pred) ** 2, axis=1))
        mean_euclidean_error = np.mean(euclidean_distances)
        
        # Calculate R² scores
        r2_x = r2_score(y_test[:, 0], y_pred[:, 0])
        r2_y = r2_score(y_test[:, 1], y_pred[:, 1])
        avg_r2_score = (r2_x + r2_y) / 2
        
        # Calculate RMSE for each coordinate
        rmse_x = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
        rmse_y = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
        
        # Cross-validation with Euclidean scorer
        print("Performing cross-validation with Euclidean distance scorer...")
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            scoring=scorer, cv=5
        )
        
        # Store training history
        self.training_history = {
            'mean_euclidean_error': mean_euclidean_error,
            'avg_r2_score': avg_r2_score,
            'r2_x': r2_x,
            'r2_y': r2_y,
            'rmse_x': rmse_x,
            'rmse_y': rmse_y,
            'cv_euclidean_mean': np.mean(cv_scores),
            'cv_euclidean_std': np.std(cv_scores),
            'best_params': best_params,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'noise_level': noise_level
        }
        
        # Print performance report
        self._print_performance_report()
        
        return self.model
    
    def _print_performance_report(self):
        """Print detailed performance report."""
        # Calculate percentage error based on fixed resolution 1080x720
        screen_width = 1080
        screen_height = 720
        screen_diagonal = np.sqrt(screen_width**2 + screen_height**2)
        percentage_error = (self.training_history['mean_euclidean_error'] / screen_diagonal) * 100
        
        print("\n" + "=" * 80)
        print("MULTI-OUTPUT RANDOM FOREST MODEL PERFORMANCE REPORT")
        print("=" * 80)
        print(f"EUCLIDEAN DISTANCE ERROR: {self.training_history['mean_euclidean_error']:.2f} pixels")
        print(f"PERCENTAGE ERROR: {percentage_error:.2f}% (relative to {screen_width}x{screen_height} diagonal)")
        print(f"Average R² Score: {self.training_history['avg_r2_score']:.4f}")
        print(f"X-coordinate R²: {self.training_history['r2_x']:.4f}")
        print(f"Y-coordinate R²: {self.training_history['r2_y']:.4f}")
        print(f"X-coordinate RMSE: {self.training_history['rmse_x']:.2f} pixels")
        print(f"Y-coordinate RMSE: {self.training_history['rmse_y']:.2f} pixels")
        print(f"Cross-validation Score (Euclidean): {self.training_history['cv_euclidean_mean']:.4f} ± {self.training_history['cv_euclidean_std']:.4f}")
        print(f"Training samples: {self.training_history['training_samples']}")
        print(f"Test samples: {self.training_history['test_samples']}")
        print(f"Noise augmentation level: {self.training_history['noise_level']} pixels")
        print(f"Best parameters: {self.training_history['best_params']}")
        print("=" * 80)
    
    def save_model(self, custom_filename=None):
        """
        Save the trained model and associated data with fixed filename.
        Always saves as 'gaze_prediction_model.joblib' to overwrite previous models.
        
        Args:
            custom_filename: Optional custom filename for the model (ignored, kept for compatibility)
            
        Returns:
            str: Path to saved model file
        """
        if self.model is None:
            print("No trained model to save.")
            return None
        
        # Create models directory in parent directory if it doesn't exist
        # This ensures compatibility with the main project structure
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(parent_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Use fixed filename to overwrite previous models
        model_filename = "gaze_prediction_model.joblib"
        model_path = os.path.join(models_dir, model_filename)
        
        # Prepare model data
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'training_history': self.training_history,
            'feature_columns': ['avg_norm_x', 'avg_norm_y', 'yaw', 'pitch', 'roll', 'x_yaw_interaction', 'y_pitch_interaction'],
            'target_columns': ['target_x', 'target_y'],
            'model_type': 'randomforest_multi_output'
        }
        
        # Save model
        joblib.dump(model_data, model_path)
        print(f"Multi-output RandomForest model saved to: {model_path}")
        
        return model_path
    
    def train_from_csv(self, csv_file, save_model=True, noise_level=5.0):
        """
        Complete training pipeline from CSV file.
        
        Args:
            csv_file: Path to calibration CSV file
            save_model: Whether to save the trained model
            noise_level: Standard deviation for data augmentation
            
        Returns:
            str: Path to saved model file if save_model=True, else None
        """
        print("=== Streamlined Multi-Output XGBoost Gaze Model Training ===")
        
        # Load data
        df = self.load_calibration_data(csv_file)
        if df is None:
            return None
        
        # Preprocess data
        df = self.preprocess_data(df)
        if df is None or len(df) == 0:
            print("Error: No data available after preprocessing.")
            return None
        
        # Apply feature engineering
        print("Applying feature engineering...")
        engineered_df = feature_engineer(df)
        print(f"Engineered data shape: {engineered_df.shape}")
        
        # Train model
        model = self.train_model(engineered_df, noise_level=noise_level)
        if model is None:
            print("Model training failed.")
            return None
        
        # Save model
        model_path = None
        if save_model:
            model_path = self.save_model()
        
        print("=== XGBoost Training Complete ===")
        return model_path

def main():
    """Main function for standalone model training."""
    # Example usage
    csv_file = "calibration_data_20250828_110210.csv"
    
    if not os.path.exists(csv_file):
        print(f"Calibration file not found: {csv_file}")
        print("Please run calibration first or provide correct path.")
        return
    
    trainer = GazeModelTrainer()
    model_path = trainer.train_from_csv(csv_file)
    
    if model_path:
        print("Training completed successfully!")
        print(f"Model saved to: {model_path}")
    else:
        print("Training failed.")

if __name__ == "__main__":
    main()
