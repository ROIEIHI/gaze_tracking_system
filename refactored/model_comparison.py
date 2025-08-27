"""
Model Comparison Script for Gaze Tracking
Tests multiple machine learning models on calibration data with proper validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
from datetime import datetime

# Optional seaborn import
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    print("Seaborn not available - using default matplotlib styling")
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

# Import different models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')

class ModelComparison:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.results = {}
        self.best_model = None
        self.best_score = float('inf')
        
    def load_and_preprocess_data(self):
        """Load and preprocess the calibration data"""
        print("=== Loading and Preprocessing Data ===")
        
        # Load data
        df = pd.read_csv(self.csv_file)
        print(f"Loaded data shape: {df.shape}")
        
        # Define features and targets
        feature_columns = ['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 
                          'yaw', 'pitch', 'roll']
        
        # Check for GazeTR features
        gazetr_columns = ['x_gaze_vect', 'y_gaze_vect', 'z_gaze_vect']
        if all(col in df.columns for col in gazetr_columns):
            feature_columns.extend(gazetr_columns)
            print("Including GazeTR features")
        else:
            print("GazeTR features not available")
            
        target_columns = ['target_x', 'target_y']
        
        # Extract features and targets
        X = df[feature_columns].copy()
        y = df[target_columns].copy()
        
        # Basic outlier removal using IQR method
        print("\nRemoving outliers...")
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (X[col] < lower_bound) | (X[col] > upper_bound)
            outlier_count = outliers.sum()
            if outlier_count > 0:
                print(f"  Found {outlier_count} outliers in {col}")
                # Replace outliers with median
                X.loc[outliers, col] = X[col].median()
        
        print(f"Final data shape: {X.shape}")
        print(f"Features: {list(X.columns)}")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Data preprocessing completed!")
        
    def define_models(self):
        """Define models and their parameter grids for GridSearchCV"""
        print("\n=== Defining Models and Parameter Grids ===")
        
        models = {
            'Ridge': {
                'model': MultiOutputRegressor(Ridge()),
                'params': {
                    'estimator__alpha': [0.1, 1.0, 10.0, 100.0],
                    'estimator__max_iter': [1000, 2000]
                }
            },
            
            'Lasso': {
                'model': MultiOutputRegressor(Lasso()),
                'params': {
                    'estimator__alpha': [0.01, 0.1, 1.0, 10.0],
                    'estimator__max_iter': [1000, 2000]
                }
            },
            
            'ElasticNet': {
                'model': MultiOutputRegressor(ElasticNet()),
                'params': {
                    'estimator__alpha': [0.1, 1.0, 10.0],
                    'estimator__l1_ratio': [0.1, 0.5, 0.9],
                    'estimator__max_iter': [1000, 2000]
                }
            },
            
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt', 'log2']
                }
            },
            
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 4, 5],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'subsample': [0.8, 0.9],
                    'min_samples_split': [2, 5]
                }
            },
            
            'XGBoost': {
                'model': MultiOutputRegressor(xgb.XGBRegressor(random_state=42)),
                'params': {
                    'estimator__n_estimators': [50, 100],
                    'estimator__max_depth': [2, 3, 4],
                    'estimator__learning_rate': [0.05, 0.1],
                    'estimator__subsample': [0.8, 0.9],
                    'estimator__reg_alpha': [1.0, 10.0],
                    'estimator__reg_lambda': [1.0, 10.0]
                }
            },
            
            'SVR': {
                'model': MultiOutputRegressor(SVR()),
                'params': {
                    'estimator__C': [0.1, 1.0, 10.0],
                    'estimator__gamma': ['scale', 'auto'],
                    'estimator__kernel': ['rbf', 'linear']
                }
            },
            
        }
        
        
        print(f"Defined {len(models)} models for comparison")
        return models
        
    def train_and_evaluate_model(self, name, model_config):
        """Train and evaluate a single model with GridSearchCV"""
        print(f"\n--- Training {name} ---")
        
        try:
            # Setup GridSearchCV
            grid_search = GridSearchCV(
                model_config['model'],
                model_config['params'],
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            # Use scaled data for all models
            X_train = self.X_train_scaled
            X_test = self.X_test_scaled
            
            # Train model
            print(f"  Running GridSearchCV with {len(model_config['params'])} parameter sets...")
            grid_search.fit(X_train, self.y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            # Calculate metrics
            train_rmse_x = np.sqrt(mean_squared_error(self.y_train.iloc[:, 0], y_train_pred[:, 0]))
            train_rmse_y = np.sqrt(mean_squared_error(self.y_train.iloc[:, 1], y_train_pred[:, 1]))
            train_rmse_mean = np.sqrt((train_rmse_x**2 + train_rmse_y**2) / 2)
            
            test_rmse_x = np.sqrt(mean_squared_error(self.y_test.iloc[:, 0], y_test_pred[:, 0]))
            test_rmse_y = np.sqrt(mean_squared_error(self.y_test.iloc[:, 1], y_test_pred[:, 1]))
            test_rmse_mean = np.sqrt((test_rmse_x**2 + test_rmse_y**2) / 2)
            
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            overfitting_ratio = test_rmse_mean / train_rmse_mean if train_rmse_mean > 0 else float('inf')
            
            # Store results
            results = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'best_cv_score': -grid_search.best_score_,
                'train_rmse_x': train_rmse_x,
                'train_rmse_y': train_rmse_y,
                'train_rmse_mean': train_rmse_mean,
                'test_rmse_x': test_rmse_x,
                'test_rmse_y': test_rmse_y,
                'test_rmse_mean': test_rmse_mean,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'overfitting_ratio': overfitting_ratio
            }
            
            # Print results
            print(f"  Best CV Score: {results['best_cv_score']:.2f}")
            print(f"  Test RMSE: {test_rmse_mean:.2f} pixels")
            print(f"  Test R²: {test_r2:.3f}")
            print(f"  Overfitting Ratio: {overfitting_ratio:.2f}x")
            
            # Track best model
            if test_rmse_mean < self.best_score:
                self.best_score = test_rmse_mean
                self.best_model = name
            
            return results
            
        except Exception as e:
            print(f"  ERROR training {name}: {str(e)}")
            return None
    
    def run_comparison(self):
        """Run the full model comparison"""
        print("=" * 60)
        print("GAZE TRACKING MODEL COMPARISON")
        print("=" * 60)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Define models
        models = self.define_models()
        
        # Train and evaluate each model
        print("\n=== Training and Evaluating Models ===")
        for name, config in models.items():
            result = self.train_and_evaluate_model(name, config)
            if result is not None:
                self.results[name] = result
        
        # Generate summary report
        self.generate_report()
        
        # Save best model
        self.save_best_model()
        
    def generate_report(self):
        """Generate a comprehensive comparison report"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)
        
        if not self.results:
            print("No successful model results to report!")
            return
            
        # Create results DataFrame
        results_data = []
        for name, result in self.results.items():
            results_data.append({
                'Model': name,
                'Test_RMSE': result['test_rmse_mean'],
                'Train_RMSE': result['train_rmse_mean'],
                'Test_R2': result['test_r2'],
                'CV_Score': result['best_cv_score'],
                'Overfitting_Ratio': result['overfitting_ratio']
            })
        
        df_results = pd.DataFrame(results_data)
        df_results = df_results.sort_values('Test_RMSE')
        
        # Print ranking
        print("\nModel Ranking (by Test RMSE):")
        print("-" * 50)
        for i, (_, row) in enumerate(df_results.iterrows(), 1):
            print(f"{i:2d}. {row['Model']:<15} - {row['Test_RMSE']:6.2f} pixels "
                  f"(R²: {row['Test_R2']:5.3f}, Overfit: {row['Overfitting_Ratio']:4.2f}x)")
        
        # Detailed results table
        print(f"\nDetailed Results:")
        print("-" * 80)
        print(f"{'Model':<15} {'Test RMSE':<10} {'Train RMSE':<11} {'Test R²':<8} {'CV Score':<9} {'Overfit':<7}")
        print("-" * 80)
        for _, row in df_results.iterrows():
            print(f"{row['Model']:<15} {row['Test_RMSE']:<10.2f} {row['Train_RMSE']:<11.2f} "
                  f"{row['Test_R2']:<8.3f} {row['CV_Score']:<9.2f} {row['Overfitting_Ratio']:<7.2f}x")
        
        # Best model details
        best_model_name = df_results.iloc[0]['Model']
        best_result = self.results[best_model_name]
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print("-" * 40)
        print(f"Test RMSE: {best_result['test_rmse_mean']:.2f} pixels")
        print(f"  - X component: {best_result['test_rmse_x']:.2f} pixels")
        print(f"  - Y component: {best_result['test_rmse_y']:.2f} pixels")
        print(f"Test R²: {best_result['test_r2']:.3f}")
        print(f"Overfitting ratio: {best_result['overfitting_ratio']:.2f}x")
        print(f"\nBest parameters:")
        for param, value in best_result['best_params'].items():
            print(f"  {param}: {value}")
            
        # Create visualization
        self.create_visualizations(df_results)
        
    def create_visualizations(self, df_results):
        """Create comparison visualizations"""
        print("\nGenerating visualizations...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Comparison Results', fontsize=16, fontweight='bold')
        
        # 1. Test RMSE comparison
        models = df_results['Model']
        test_rmse = df_results['Test_RMSE']
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        
        bars1 = ax1.bar(models, test_rmse, color=colors)
        ax1.set_title('Test RMSE by Model')
        ax1.set_ylabel('RMSE (pixels)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, test_rmse):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 2. R² comparison
        test_r2 = df_results['Test_R2']
        bars2 = ax2.bar(models, test_r2, color=colors)
        ax2.set_title('Test R² by Model')
        ax2.set_ylabel('R² Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(test_r2) * 1.1)
        
        # Add value labels
        for bar, value in zip(bars2, test_r2):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Overfitting ratio
        overfit_ratio = df_results['Overfitting_Ratio']
        bars3 = ax3.bar(models, overfit_ratio, color=colors)
        ax3.set_title('Overfitting Ratio by Model')
        ax3.set_ylabel('Overfitting Ratio')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No overfitting')
        ax3.legend()
        
        # Add value labels
        for bar, value in zip(bars3, overfit_ratio):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Training vs Test RMSE scatter
        train_rmse = df_results['Train_RMSE']
        scatter = ax4.scatter(train_rmse, test_rmse, c=range(len(models)), 
                            cmap='viridis', s=100, alpha=0.7)
        ax4.set_xlabel('Training RMSE')
        ax4.set_ylabel('Test RMSE')
        ax4.set_title('Training vs Test RMSE')
        ax4.grid(True, alpha=0.3)
        
        # Add diagonal line (perfect fit)
        min_val = min(min(train_rmse), min(test_rmse))
        max_val = max(max(train_rmse), max(test_rmse))
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, 
                label='Perfect fit')
        ax4.legend()
        
        # Add model labels
        for i, model in enumerate(models):
            ax4.annotate(model, (train_rmse.iloc[i], test_rmse.iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"model_comparison_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved as: {plot_filename}")
        plt.show()
        
    def save_best_model(self):
        """Save the best performing model"""
        if not self.results:
            print("No models to save!")
            return
            
        # Find best model
        best_model_name = min(self.results.keys(), 
                            key=lambda x: self.results[x]['test_rmse_mean'])
        best_result = self.results[best_model_name]
        
        # Create model package
        model_package = {
            'model': best_result['model'],
            'scaler': self.scaler,
            'model_name': best_model_name,
            'performance': {
                'test_rmse_mean': best_result['test_rmse_mean'],
                'test_r2': best_result['test_r2'],
                'overfitting_ratio': best_result['overfitting_ratio']
            },
            'best_params': best_result['best_params'],
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"best_gaze_model_{best_model_name.lower()}_{timestamp}.pkl"
        
        joblib.dump(model_package, model_filename)
        print(f"\n💾 Best model ({best_model_name}) saved as: {model_filename}")
        print(f"   Test RMSE: {best_result['test_rmse_mean']:.2f} pixels")
        print(f"   Test R²: {best_result['test_r2']:.3f}")

def main():
    """Main function to run model comparison"""
    csv_file = "calibration_data_20250826_163148.csv"
    
    print("🔍 Starting comprehensive model comparison for gaze tracking...")
    print(f"📊 Using calibration data: {csv_file}")
    
    # Run comparison
    comparison = ModelComparison(csv_file)
    comparison.run_comparison()
    
    print("\n✅ Model comparison completed!")
    print("Check the generated visualization and saved model files.")

if __name__ == "__main__":
    main()
