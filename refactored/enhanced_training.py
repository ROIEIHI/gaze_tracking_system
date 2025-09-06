#!/usr/bin/env python3
"""
Enhanced multi-output model training with refined hyperparameters and better optimization.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
import datetime
import os

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available.")

def euclidean_distance_scorer(y_true, y_pred):
    """Enhanced Euclidean distance scorer with better numerical stability"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Ensure 2D arrays
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 2)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 2)
    
    # Calculate Euclidean distances
    euclidean_distances = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
    
    # Return negative mean distance (for maximization)
    return -np.mean(euclidean_distances)

# Create scorer object
euclidean_scorer = make_scorer(euclidean_distance_scorer, greater_is_better=True)

def enhanced_multi_output_training(csv_file):
    """Enhanced training with better hyperparameters and optimization strategy"""
    
    print("🚀 Enhanced Multi-Output XGBoost Training")
    print("=" * 60)
    
    # Load and prepare data
    df = pd.read_csv(csv_file)
    print(f"Loaded data shape: {df.shape}")
    
    # Feature engineering
    df['avg_norm_x'] = (df['norm_x_L'] + df['norm_x_R']) / 2
    df['avg_norm_y'] = (df['norm_y_L'] + df['norm_y_R']) / 2
    df['x_yaw_interaction'] = df['avg_norm_x'] * df['yaw']
    df['y_pitch_interaction'] = df['avg_norm_y'] * df['pitch']
    
    # Features and targets
    feature_columns = ['avg_norm_x', 'avg_norm_y', 'yaw', 'pitch', 'roll', 
                      'x_yaw_interaction', 'y_pitch_interaction']
    
    X = df[feature_columns].values
    y = np.column_stack([df['target_x'].values, df['target_y'].values])
    
    print(f"Feature matrix: {X.shape}, Target matrix: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Enhanced parameter grid with focus on accuracy
    print("🎯 Enhanced hyperparameter optimization...")
    
    # First, broader search
    broad_param_grid = {
        'n_estimators': [200, 400, 600],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 1.5]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    # Use both Euclidean and MSE scoring for comparison
    print("Training with Euclidean distance optimization...")
    grid_search_euclidean = GridSearchCV(
        xgb_model, broad_param_grid, cv=3, scoring=euclidean_scorer,
        n_jobs=-1, verbose=1
    )
    grid_search_euclidean.fit(X_train_scaled, y_train)
    
    print("Training with MSE optimization for comparison...")
    grid_search_mse = GridSearchCV(
        xgb_model, broad_param_grid, cv=3, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    grid_search_mse.fit(X_train_scaled, y_train)
    
    # Evaluate both models
    models = {
        'Euclidean Optimized': grid_search_euclidean.best_estimator_,
        'MSE Optimized': grid_search_mse.best_estimator_
    }
    
    best_model = None
    best_error = float('inf')
    best_name = ''
    
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    for name, model in models.items():
        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        
        # Calculate Euclidean error
        euclidean_errors = np.sqrt(np.sum((y_test - y_pred) ** 2, axis=1))
        mean_euclidean_error = np.mean(euclidean_errors)
        
        # Calculate R² scores
        r2_x = r2_score(y_test[:, 0], y_pred[:, 0])
        r2_y = r2_score(y_test[:, 1], y_pred[:, 1])
        avg_r2 = (r2_x + r2_y) / 2
        
        print(f"\n{name}:")
        print(f"  Euclidean Error: {mean_euclidean_error:.2f} pixels")
        print(f"  R² Score: {avg_r2:.4f}")
        print(f"  X R²: {r2_x:.4f}, Y R²: {r2_y:.4f}")
        
        if mean_euclidean_error < best_error:
            best_error = mean_euclidean_error
            best_model = model
            best_name = name
    
    print(f"\n🏆 BEST MODEL: {best_name} ({best_error:.2f} pixels)")
    
    # Calculate improvement over previous best
    previous_best = 79.98
    improvement = previous_best - best_error
    improvement_percent = (improvement / previous_best) * 100
    
    print("\n" + "🎯"*30)
    print("FINAL COMPARISON WITH PREVIOUS BEST")
    print("🎯"*30)
    print(f"🎯 NEW BEST ERROR: {best_error:.2f} pixels")
    print(f"📊 Previous Best: {previous_best:.2f} pixels")
    print(f"🚀 IMPROVEMENT: {improvement:+.2f} pixels ({improvement_percent:+.1f}%)")
    
    if improvement > 0:
        print(f"✅ SUCCESS: {improvement_percent:.1f}% improvement achieved!")
    else:
        print(f"⚠️  Performance: {abs(improvement_percent):.1f}% worse than previous")
    
    print("🎯"*30)
    
    # Save the best model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"enhanced_gaze_model_{timestamp}.joblib"
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_filename)
    
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'training_history': {
            'mean_euclidean_error': best_error,
            'avg_r2_score': avg_r2,
            'optimization_method': best_name,
            'training_samples': X_train.shape[0],
            'test_samples': X_test.shape[0]
        },
        'feature_columns': feature_columns,
        'target_columns': ['target_x', 'target_y'],
        'model_type': 'multi_output_xgboost'
    }
    
    joblib.dump(model_data, model_path)
    print(f"\n💾 Enhanced model saved to: {model_path}")
    
    return model_path

def main():
    csv_file = r"C:\Users\roie1\Desktop\collage\eye_tracker_git\gaze_tracking_system\refactored\calibration_data_20250828_110210.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ ERROR: File not found: {csv_file}")
        return
    
    try:
        model_path = enhanced_multi_output_training(csv_file)
        print(f"\n🎉 Enhanced training complete! Model: {model_path}")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
