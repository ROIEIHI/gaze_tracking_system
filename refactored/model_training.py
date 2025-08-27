import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import datetime
import os
import torch
import torch.nn as nn 
import numpy as np
import math
import copy
from torchvision.models import resnet18

class GazeModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.training_history = {}
        self.best_params = {}
        
    def load_calibration_data(self, csv_file):
        """Load calibration data from CSV file"""
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded calibration data from {csv_file}")
            print(f"Data shape: {df.shape}")
            
            # Validate required columns
            required_columns = ['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 
                            'yaw', 'pitch', 'roll', 'target_x', 'target_y']
            
            # Check for GazeTR features
            gazetr_columns = ['x_gaze_vect', 'y_gaze_vect', 'z_gaze_vect']
            gazetr_available = all(col in df.columns for col in gazetr_columns)

            if gazetr_available:
                print("GazeTR features found in calibration data")
                required_columns.extend(gazetr_columns)
            else:
                print("GazeTR features not found in calibration data")
                print("Using only traditional features")
                # Add default zero columns for backward compatibility
                for col in gazetr_columns:
                    df[col] = 0.0

            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV file missing required columns. Expected: {required_columns}")
            
            return df
        except Exception as e:
            print(f"Error loading calibration data: {str(e)}")
            return None
    
    def create_visualizations(self, df):
        """Create visualizations of the calibration data"""
        print("Creating data visualizations...")
        
        # Figure 1: Feature histograms
        plt.figure(figsize=(15, 10))
        features = ['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 
                    'yaw', 'pitch', 'roll'
                    'x_gaze_vect', 'y_gaze_vect', 'z_gaze_vect']
        df[features].hist(bins=20, alpha=0.7, figsize=(15, 10))
        plt.suptitle('Feature Histograms', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Figure 2: Correlation heatmap
        plt.figure(figsize=(15, 10))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
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

        # Add GazeTR-specific visualizations
        plt.figure(figsize=(15, 10))
        plt.title('GazeTR Gaze Vectors vs. Target Positions', fontsize=16)
        plt.scatter(df['x_gaze_vect'], df['target_x'], alpha=0.5, label='X Gaze vs Target X')
        plt.scatter(df['y_gaze_vect'], df['target_y'], alpha=0.5, label='Y Gaze vs Target Y')
        plt.xlabel('GazeTR Gaze Vector Component')
        plt.ylabel('Target Position')
        plt.legend()
        plt.grid(True)
        plt.show()
        
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
    
    def train_model(self, df, test_size=0.2, random_state=42, use_grid_search=True):
        """Train the RandomForest model with StandardScaler and GridSearchCV optimization"""
        print("--- Training Model with RandomForest + GridSearchCV ---")
        
        # Prepare features and targets
        feature_columns = ['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 
                           'yaw', 'pitch', 'roll',
                           'x_gaze_vect', 'y_gaze_vect', 'z_gaze_vect']
        
        target_columns = ['target_x', 'target_y']
        
        X = df[feature_columns].values
        y = df[target_columns].values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target matrix shape: {y.shape}")
        print(f"Using features: {feature_columns}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Apply StandardScaler normalization
        print("Applying StandardScaler normalization...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Print normalization statistics
        print(f"Feature scaling completed:")
        print(f"  Original range: [{X_train.min():.3f}, {X_train.max():.3f}]")
        print(f"  Scaled range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
        print(f"  Scaled mean: {X_train_scaled.mean():.3f}, std: {X_train_scaled.std():.3f}")
        
        if use_grid_search:
            print("Performing GridSearchCV with RandomForest...")
            
            # Define RandomForest base model
            rf_base = RandomForestRegressor(
                random_state=random_state,
                n_jobs=-1
            )
            
            # Define parameter grid for GridSearchCV (based on model comparison results)
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [2,3,4],
                'min_samples_leaf': [5, 10, 15],
                'min_samples_split': [10, 20, 30], # Prevent tiny splits
                'max_features': ['sqrt', 'log2']
            }
            
            # Create GridSearchCV with cross-validation
            print("Grid search parameters:")
            total_combinations = 1
            for param, values in param_grid.items():
                print(f"  {param}: {values}")
                total_combinations *= len(values)
            print(f"Total parameter combinations: {total_combinations}")
            print("This may take several minutes...")
            
            grid_search = GridSearchCV(
                estimator=rf_base,
                param_grid=param_grid,
                cv=3,  # 3-fold cross-validation
                scoring='neg_mean_squared_error',
                n_jobs=-1,  # Use all available cores
                verbose=1,
                return_train_score=True
            )
            
            # Fit GridSearchCV
            grid_search.fit(X_train_scaled, y_train)
            
            # Get best model
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"\nGridSearchCV completed!")
            print(f"Best parameters found:")
            for param, value in self.best_params.items():
                print(f"  {param}: {value}")
            print(f"Best cross-validation score: {-grid_search.best_score_:.2f}")
            
        else:
            print("Training with default RandomForest parameters...")
            # Fallback to default RandomForest model without grid search
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model on scaled data
        print("Evaluating model performance...")
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
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
            'test_samples': X_test.shape[0],
            'feature_scaling': True,
            'grid_search_used': use_grid_search,
            'best_params': self.best_params if use_grid_search else None
        }
        
        # Print results with clear formatting
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION RESULTS (RandomForest + StandardScaler + GridSearchCV)")
        print(f"{'='*60}")
        print(f"Training Performance:")
        print(f"  RMSE X: {train_rmse_x:.2f} pixels")
        print(f"  RMSE Y: {train_rmse_y:.2f} pixels")
        print(f"  Mean Error: {train_mean_error:.2f} pixels")
        print(f"\nTest Performance:")
        print(f"  RMSE X: {test_rmse_x:.2f} pixels")
        print(f"  RMSE Y: {test_rmse_y:.2f} pixels")
        print(f"  Mean Error: {test_mean_error:.2f} pixels")
        print(f"\nData Statistics:")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Feature normalization: ✅ StandardScaler applied")
        print(f"  Hyperparameter optimization: {'✅ GridSearchCV' if use_grid_search else '❌ Default params'}")
        
        # Enhanced overfitting detection
        overfitting_ratio = test_mean_error / train_mean_error if train_mean_error > 0 else float('inf')
        print(f"  Overfitting ratio: {overfitting_ratio:.2f}x")
        
        if overfitting_ratio > 2.0:
            print(f"⚠️  WARNING: Significant overfitting detected!")
            print(f"   Consider collecting more calibration data or adjusting model parameters.")
        elif overfitting_ratio > 1.5:
            print(f"⚠️  CAUTION: Moderate overfitting detected.")
        else:
            print(f"✅ Good generalization - minimal overfitting.")
        
        print(f"{'='*60}")
        
        return self.model
    
    def save_model(self, model_filename=None):
        """Save the trained model and scaler to file"""
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
            
            # Save model, scaler, and training history
            model_data = {
                'model': self.model,
                'scaler': self.scaler,  # Save the StandardScaler
                'training_history': self.training_history,
                'best_params': self.best_params,  # Save GridSearchCV results
                'feature_columns': ['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 
                                'yaw', 'pitch', 'roll',
                                'x_gaze_vect', 'y_gaze_vect', 'z_gaze_vect'],
                'target_columns': ['target_x', 'target_y'],
                'model_version': '3.0',  # Version with RandomForest + StandardScaler + GridSearchCV
                'timestamp': datetime.datetime.now().isoformat(),
                'gazetr_enabled': True
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


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos
        

    def forward(self, src, pos):
                # src_mask: Optional[Tensor] = None,
                # src_key_padding_mask: Optional[Tensor] = None):
                # pos: Optional[Tensor] = None):

        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        maps = 32  # Match the pre-trained weights dimension
        nhead = 8
        dim_feature = 7*7
        dim_feedforward=512
        dropout = 0.1
        num_layers=6

        # Modified ResNet initialization to extract spatial features
        self.base_model = resnet18(pretrained=False)
        # Modify first conv layer to maintain compatibility
        self.base_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final fully connected layer and global average pooling
        # to get spatial features instead of class logits
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])  # Remove avgpool and fc
        
        # Now the base_model will output [batch, 512, 7, 7] for 224x224 input
        
        # Add a projection layer to convert from 512 to 32 dimensions to match pre-trained weights
        self.feature_projection = nn.Linear(512, maps)

        encoder_layer = TransformerEncoderLayer(
                  maps, 
                  nhead, 
                  dim_feedforward, 
                  dropout)

        encoder_norm = nn.LayerNorm(maps) 
        # num_encoder_layer: deeps of layers 

        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))

        self.pos_embedding = nn.Embedding(dim_feature+1, maps)  # 7*7 + 1 = 50 positions

        self.feed = nn.Linear(maps, 2)  # maps = 32, so 32 -> 2
            
        self.loss_op = nn.L1Loss()


    def forward(self, x_in):
        # Extract spatial features from ResNet backbone
        feature = self.base_model(x_in["face"])  # [batch, 512, 7, 7]
        batch_size = feature.size(0)
        
        # Flatten spatial dimensions: [batch, 512, 7, 7] -> [batch, 512, 49]
        feature = feature.flatten(2)
        
        # Permute to prepare for projection: [batch, 512, 49] -> [batch, 49, 512]
        feature = feature.permute(0, 2, 1)
        
        # Project features to match pre-trained dimensions: [batch, 49, 512] -> [batch, 49, 32]
        feature = self.feature_projection(feature)
        
        # Permute to transformer format: [batch, 49, 32] -> [49, batch, 32]
        feature = feature.permute(1, 0, 2)
        
        # Add class token: [1, batch, 32]
        cls = self.cls_token.repeat( (1, batch_size, 1))
        # Concatenate: [49+1, batch, 32] = [50, batch, 32]
        feature = torch.cat([cls, feature], 0)
        
        # Get device from the input feature tensor to be device-agnostic
        device = feature.device
        # Position encoding for 50 positions (1 cls + 49 spatial)
        position = torch.from_numpy(np.arange(0, 50)).to(device)

        pos_feature = self.pos_embedding(position)

        # Apply transformer encoder: [50, batch, 32]
        feature = self.encoder(feature, pos_feature)
  
        # Permute back: [50, batch, 32] -> [batch, 32, 50]
        feature = feature.permute(1, 2, 0)

        # Extract class token features (index 0): [batch, 32, 50] -> [batch, 32]
        feature = feature[:,:,0]  # Take the class token (first position)

        # Final prediction layer: [batch, 32] -> [batch, 2]
        gaze = self.feed(feature)
        
        return gaze

    def loss(self, x_in, label):
        gaze = self.forward(x_in)
        loss = self.loss_op(gaze, label) 
        return loss


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
