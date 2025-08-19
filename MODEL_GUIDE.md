# Model Saving and Loading Guide

This guide explains how to use the new model saving and loading functionality in the Eye Tracker system.

## 🎯 Overview

The eye tracker now automatically saves trained models to files, allowing you to:
- Reuse trained models without recalibrating
- Share models between different scripts
- Build applications that use pre-trained gaze prediction

## 📁 New Functions in model_utils.py

### `save_model(model, filename=None)`
Saves a trained model to disk using joblib serialization.

**Parameters:**
- `model`: The trained XGBoost model to save
- `filename` (optional): Custom filename. If not provided, uses timestamp-based naming

**Returns:** Filename where model was saved, or None if failed

**Example:**
```python
import model_utils

# After training a model
model = model_utils.train_model(calibration_data)
saved_file = model_utils.save_model(model, "my_gaze_model")
```

### `load_model(filename)`
Loads a previously saved model from disk.

**Parameters:**
- `filename`: Path to the saved model file (.joblib)

**Returns:** Loaded model object, or None if failed

**Example:**
```python
import model_utils

# Load a previously saved model
model = model_utils.load_model("gaze_model_20250819_143022.joblib")
if model:
    # Use the model for predictions
    prediction = model.predict(features)
```

### `list_saved_models(directory=".")`
Lists all saved gaze models in a directory.

**Parameters:**
- `directory` (optional): Directory to search (default: current directory)

**Returns:** List of model filenames

**Example:**
```python
import model_utils

# See what models are available
available_models = model_utils.list_saved_models()
print(f"Found {len(available_models)} saved models")
```

### `test_model_prediction(model, sample_features=None)`
Tests a model with sample or custom features to verify it works correctly.

**Parameters:**
- `model`: The model to test
- `sample_features` (optional): Custom feature values [norm_x_L, norm_y_L, norm_x_R, norm_y_R, yaw, pitch, roll]

**Returns:** Tuple of (predicted_x, predicted_y) coordinates

## 🚀 Usage Examples

### Automatic Model Saving
When you run the main eye tracker (`python main.py`), it now automatically saves the trained model:

```bash
python main.py
# ... calibration and training happens ...
# ✅ Model saved as: gaze_model_20250819_143022.joblib
```

### Using a Saved Model in Your Own Script
```python
import model_utils
import numpy as np

# Load a trained model
model = model_utils.load_model("gaze_model_20250819_143022.joblib")

# Prepare feature data (7 features)
# [left_iris_x, left_iris_y, right_iris_x, right_iris_y, head_yaw, head_pitch, head_roll]
features = np.array([[0.6, 0.4, 0.6, 0.4, 5.0, -2.0, 0.0]])  # Looking slightly right and up

# Make prediction
prediction = model.predict(features)[0]
gaze_x, gaze_y = prediction[0], prediction[1]

print(f"Predicted gaze position: ({gaze_x:.1f}, {gaze_y:.1f}) pixels")
```

### Interactive Demo
Use the included demo script to test saved models:

```bash
python model_demo.py                    # Interactive model selection
python model_demo.py my_model.joblib    # Load specific model
```

The demo script provides:
- Model loading and testing
- Sample predictions for different gaze directions
- Interactive mode for custom feature input

## 📋 Model File Format

- **Format:** Joblib serialization (.joblib files)
- **Naming:** `gaze_model_YYYYMMDD_HHMMSS.joblib` (automatic)
- **Size:** Typically 50-100 KB
- **Compatibility:** Cross-platform (Windows, macOS, Linux)

## 🔧 Integration Tips

### For External Applications
```python
# Minimal integration example
import joblib
import numpy as np

# Load model
model = joblib.load("path/to/gaze_model.joblib")

# Prepare features (example values)
features = np.array([[0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]])  # Center gaze

# Predict
prediction = model.predict(features)[0]
x, y = prediction[0], prediction[1]
```

### Feature Extraction
To use saved models, you need to extract the same 7 features:
1. **Left iris X** (normalized 0.0-1.0)
2. **Left iris Y** (normalized 0.0-1.0)  
3. **Right iris X** (normalized 0.0-1.0)
4. **Right iris Y** (normalized 0.0-1.0)
5. **Head yaw** (degrees, typically -30 to +30)
6. **Head pitch** (degrees, typically -30 to +30)
7. **Head roll** (degrees, typically -30 to +30)

## 🛠️ Troubleshooting

### Common Issues

**"No module named 'joblib'"**
```bash
pip install joblib
```

**"Model file not found"**
- Check the filename and path
- Use `model_utils.list_saved_models()` to see available models

**"Prediction errors"**
- Ensure feature values are in the correct range
- Check that you're providing exactly 7 features
- Verify the model was trained successfully

### Model Compatibility
- Models are tied to the specific feature extraction method
- Models trained with different MediaPipe versions may not be compatible
- Always test loaded models before using in production

## 📊 Model Performance

Saved models retain their original performance characteristics:
- **Training accuracy:** RMSE typically 100-200 pixels
- **Inference speed:** Sub-millisecond predictions
- **Memory usage:** ~50 KB model file, minimal RAM

Use `model_utils.test_model_prediction()` to verify performance after loading.
