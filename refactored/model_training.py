import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import datetime
import os

# Optional imports for visualizations
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Using basic sklearn models.")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Visualizations disabled.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: Seaborn not available. Some visualizations disabled.")

class GazeModelTrainer:
    def __init__(self):
        self.model = None
        self.training_history = {}
        
    def load_calibration_data(self, csv_file):
        """Load calibration data from CSV file"""
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded calibration data from {csv_file}")
            print(f"Data shape: {df.shape}")
            
            # Validate required columns
            required_columns = ['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 'yaw', 'pitch', 'roll', 'target_x', 'target_y']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV file missing required columns. Expected: {required_columns}")
            
            return df
        except Exception as e:
            print(f"Error loading calibration data: {str(e)}")
            return None
    
    def create_visualizations(self, df):
        """Create visualizations of the calibration data"""
        if not MATPLOTLIB_AVAILABLE:
            print("Skipping visualizations (matplotlib not available)")
            return
        
        print("Creating data visualizations...")
        
        # Figure 1: Feature histograms
        plt.figure(figsize=(12, 8))
        features = ['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 'yaw', 'pitch', 'roll']
        df[features].hist(bins=20, alpha=0.7, figsize=(12, 8))
        plt.suptitle('Feature Histograms', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Figure 2: Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = df.corr()
        if SEABORN_AVAILABLE:
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        else:
            # Fallback visualization without seaborn
            plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
            plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
        plt.title('Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Figure 3: Feature vs target scatter plots
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Features vs Target Coordinates', fontsize=16)
        
        for i, feature in enumerate(features):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            ax.scatter(df[feature], df['target_x'], alpha=0.6, label='target_x', color='blue')
            ax.scatter(df[feature], df['target_y'], alpha=0.6, label='target_y', color='red')
            ax.set_xlabel(feature)
            ax.set_ylabel('Target Coordinates')
            ax.legend()
            ax.set_title(f'{feature} vs Targets')
        
        # Hide unused subplots
        for i in range(len(features), 9):
            row = i // 3
            col = i % 3
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_data(self, df):
        """Preprocess the calibration data"""
        print("Preprocessing calibration data...")
        
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            print("Warning: Missing values found in data")
            df = df.dropna()
            print(f"Data shape after removing missing values: {df.shape}")
        
        # Check for outliers (basic check)
        features = ['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 'yaw', 'pitch', 'roll']
        for feature in features:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
            if len(outliers) > 0:
                print(f"Found {len(outliers)} outliers in {feature}")
        
        print(f"Final preprocessed data shape: {df.shape}")
        return df
    
    def train_model(self, df, test_size=0.2, random_state=42):
        """Train the XGBoost model"""
        print("--- Training Model ---")
        
        # Prepare features and targets
        feature_columns = ['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 'yaw', 'pitch', 'roll']
        target_columns = ['target_x', 'target_y']
        
        X = df[feature_columns].values
        y = df[target_columns].values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target matrix shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Create and train model
        if XGBOOST_AVAILABLE:
            print("Using XGBoost regressor...")
            base_regressor = xgb.XGBRegressor(
                n_estimators=100,
                random_state=random_state,
                max_depth=6,
                learning_rate=0.1
            )
        else:
            print("Using Random Forest regressor (XGBoost not available)...")
            from sklearn.ensemble import RandomForestRegressor
            base_regressor = RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                max_depth=6
            )
        
        self.model = MultiOutputRegressor(base_regressor)
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_rmse_x = np.sqrt(mean_squared_error(y_train[:, 0], y_pred_train[:, 0]))
        train_rmse_y = np.sqrt(mean_squared_error(y_train[:, 1], y_pred_train[:, 1]))
        test_rmse_x = np.sqrt(mean_squared_error(y_test[:, 0], y_pred_test[:, 0]))
        test_rmse_y = np.sqrt(mean_squared_error(y_test[:, 1], y_pred_test[:, 1]))
        
        # Calculate mean prediction error
        train_distances = np.sqrt((y_train[:, 0] - y_pred_train[:, 0])**2 + (y_train[:, 1] - y_pred_train[:, 1])**2)
        test_distances = np.sqrt((y_test[:, 0] - y_pred_test[:, 0])**2 + (y_test[:, 1] - y_pred_test[:, 1])**2)
        
        train_mean_error = np.mean(train_distances)
        test_mean_error = np.mean(test_distances)
        
        # Store training history
        self.training_history = {
            'train_rmse_x': train_rmse_x,
            'train_rmse_y': train_rmse_y,
            'test_rmse_x': test_rmse_x,
            'test_rmse_y': test_rmse_y,
            'train_mean_error': train_mean_error,
            'test_mean_error': test_mean_error,
            'training_samples': X_train.shape[0],
            'test_samples': X_test.shape[0]
        }
        
        # Print results
        print(f"\nModel Evaluation Results:")
        print(f"Training RMSE X: {train_rmse_x:.2f} pixels")
        print(f"Training RMSE Y: {train_rmse_y:.2f} pixels")
        print(f"Training Mean Error: {train_mean_error:.2f} pixels")
        print(f"Test RMSE X: {test_rmse_x:.2f} pixels")
        print(f"Test RMSE Y: {test_rmse_y:.2f} pixels")
        print(f"Test Mean Error: {test_mean_error:.2f} pixels")
        
        # Check for overfitting
        if test_mean_error > train_mean_error * 1.5:
            print("Warning: Possible overfitting detected!")
        
        return self.model
    
    def save_model(self, model_filename=None):
        """Save the trained model to file"""
        if self.model is None:
            print("No model to save. Train a model first.")
            return None
        
        if model_filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"gaze_model_{timestamp}.joblib"
        
        try:
            # Create models directory if it doesn't exist
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            model_path = os.path.join(models_dir, model_filename)
            
            # Save model and training history
            model_data = {
                'model': self.model,
                'training_history': self.training_history,
                'feature_columns': ['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 'yaw', 'pitch', 'roll'],
                'target_columns': ['target_x', 'target_y']
            }
            
            joblib.dump(model_data, model_path)
            print(f"Model saved to: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return None
    
    def load_model(self, model_path):
        """Load a trained model from file"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.training_history = model_data.get('training_history', {})
            print(f"Model loaded from: {model_path}")
            
            # Print model info if available
            if self.training_history:
                print(f"Model test error: {self.training_history.get('test_mean_error', 'N/A'):.2f} pixels")
            
            return self.model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    
    def create_prediction_plots(self, df):
        """Create plots showing model predictions vs actual targets"""
        if not MATPLOTLIB_AVAILABLE:
            print("Skipping prediction plots (matplotlib not available)")
            return
        
        if self.model is None:
            print("No model available for prediction plots")
            return
        
        feature_columns = ['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 'yaw', 'pitch', 'roll']
        X = df[feature_columns].values
        y_actual = df[['target_x', 'target_y']].values
        y_pred = self.model.predict(X)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # X coordinate predictions
        ax1.scatter(y_actual[:, 0], y_pred[:, 0], alpha=0.6)
        ax1.plot([y_actual[:, 0].min(), y_actual[:, 0].max()], 
                [y_actual[:, 0].min(), y_actual[:, 0].max()], 'r--', lw=2)
        ax1.set_xlabel('Actual X')
        ax1.set_ylabel('Predicted X')
        ax1.set_title('X Coordinate Predictions')
        ax1.grid(True)
        
        # Y coordinate predictions
        ax2.scatter(y_actual[:, 1], y_pred[:, 1], alpha=0.6)
        ax2.plot([y_actual[:, 1].min(), y_actual[:, 1].max()], 
                [y_actual[:, 1].min(), y_actual[:, 1].max()], 'r--', lw=2)
        ax2.set_xlabel('Actual Y')
        ax2.set_ylabel('Predicted Y')
        ax2.set_title('Y Coordinate Predictions')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def train_from_csv(self, csv_file, create_visualizations=True, save_model=True):
        """Complete training pipeline from CSV file"""
        print("=== Gaze Model Training ===")
        
        # Load data
        df = self.load_calibration_data(csv_file)
        if df is None:
            return None
        
        # Create visualizations if requested
        if create_visualizations:
            self.create_visualizations(df)
        
        # Preprocess data
        df = self.preprocess_data(df)
        
        # Train model
        model = self.train_model(df)
        if model is None:
            return None
        
        # Create prediction plots
        if create_visualizations:
            self.create_prediction_plots(df)
        
        # Save model if requested
        model_path = None
        if save_model:
            model_path = self.save_model()
        
        print("=== Training Complete ===")
        return model_path

if __name__ == "__main__":
    # Example usage
    trainer = GazeModelTrainer()
    
    # Train from a CSV file
    csv_file = "calibration_data_20250821_120000.csv"  # Replace with actual file
    model_path = trainer.train_from_csv(csv_file)
    
    if model_path:
        print(f"Model successfully trained and saved to: {model_path}")
    else:
        print("Model training failed")
