# model_utils.py
# This file contains functions related to machine learning model training,
# evaluation, and data visualization. It separates the data science aspects
# of the project from the core application logic.

import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

def visualize_data(df):
    """
    Exports data to CSV and creates detailed visualizations to inspect the
    calibration data quality and feature relationships.

    Args:
        df (pd.DataFrame): The DataFrame containing the collected calibration data.
    """
    if df is None or df.empty:
        print("No data to visualize.")
        return

    print("--- Generating Data Visualizations ---")

    # --- Figure 1: Histograms of all features ---
    features_to_plot = ['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 'yaw', 'pitch', 'roll']
    df[features_to_plot].hist(bins=20, figsize=(15, 10))
    plt.suptitle('Feature Distributions', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- Figure 2: Correlation Heatmap ---
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt='.2f')
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.show()

    # --- Figure 3: Scatter plots of features vs. targets ---
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('Features vs. Target Coordinates', fontsize=20)
    axes = axes.flatten()

    for i, feature in enumerate(features_to_plot):
        ax = axes[i]
        # Plot feature vs. target_x
        ax.scatter(df[feature], df['target_x'], alpha=0.5, label='Target X', color='blue')
        # Plot feature vs. target_y
        ax.scatter(df[feature], df['target_y'], alpha=0.5, label='Target Y', color='red')
        ax.set_title(f'{feature} vs. Targets')
        ax.set_xlabel(feature)
        ax.set_ylabel('Screen Coordinates (pixels)')
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def train_model(df):
    """
    Trains a Multi-Output XGBoost Regressor model on the provided data.

    Args:
        df (pd.DataFrame): The DataFrame containing features and target coordinates.

    Returns:
        The trained machine learning model object, or None if training fails.
    """
    if df is None or df.empty:
        print("DataFrame is empty. Skipping model training.")
        return None

    print("--- Training Gaze Prediction Model ---")

    # Define features (X) and targets (y)
    features = ['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 'yaw', 'pitch', 'roll']
    targets = ['target_x', 'target_y']
    
    X = df[features]
    y = df[targets]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the XGBoost Regressor
    xgb_regressor = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )

    # Wrap the model with MultiOutputRegressor to predict both X and Y
    model = MultiOutputRegressor(xgb_regressor)

    # Train the model
    model.fit(X_train, y_train)
    print("Model training complete.")

    # --- Evaluate the model ---
    y_pred = model.predict(X_test)

    # Calculate RMSE for each coordinate
    rmse_x = np.sqrt(mean_squared_error(y_test['target_x'], y_pred[:, 0]))
    rmse_y = np.sqrt(mean_squared_error(y_test['target_y'], y_pred[:, 1]))

    # Calculate the mean prediction error in pixels (Euclidean distance)
    pixel_errors = np.sqrt((y_test['target_x'] - y_pred[:, 0])**2 + (y_test['target_y'] - y_pred[:, 1])**2)
    mean_pixel_error = np.mean(pixel_errors)

    print("\n--- Model Evaluation Results ---")
    print(f"RMSE for X coordinate: {rmse_x:.2f} pixels")
    print(f"RMSE for Y coordinate: {rmse_y:.2f} pixels")
    print(f"Mean Prediction Error: {mean_pixel_error:.2f} pixels")
    print("--------------------------------")

    return model
