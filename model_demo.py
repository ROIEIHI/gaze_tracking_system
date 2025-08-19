#!/usr/bin/env python3
"""
Standalone Eye Tracker Model Demo
=================================

This script demonstrates how to load and use a previously trained gaze tracking model
for predictions without running the full calibration pipeline.

Usage:
    python model_demo.py [model_filename]

If no filename is provided, it will list available models and let you choose.
"""

import sys
import os
import numpy as np

# Add the current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import model_utils

def interactive_model_selection():
    """Allow user to interactively select a model from available ones."""
    print("🔍 Searching for saved models...")
    available_models = model_utils.list_saved_models()
    
    if not available_models:
        print("❌ No saved models found. Please train a model first using main.py")
        return None
    
    print("\nSelect a model to load:")
    for i, model_name in enumerate(available_models, 1):
        print(f"  {i}. {model_name}")
    
    try:
        choice = input(f"\nEnter choice (1-{len(available_models)}): ")
        index = int(choice) - 1
        if 0 <= index < len(available_models):
            return available_models[index]
        else:
            print("❌ Invalid choice")
            return None
    except ValueError:
        print("❌ Please enter a valid number")
        return None

def demonstrate_model_usage(model):
    """Demonstrate various ways to use the loaded model."""
    print("\n" + "="*50)
    print("🧪 DEMONSTRATING MODEL USAGE")
    print("="*50)
    
    # Test 1: Center gaze
    print("\n1️⃣ Test: Center gaze")
    center_features = [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]  # Looking straight
    model_utils.test_model_prediction(model, center_features)
    
    # Test 2: Looking right
    print("\n2️⃣ Test: Looking right")
    right_features = [0.7, 0.5, 0.7, 0.5, 15.0, 0.0, 0.0]  # Head turned right, eyes right
    model_utils.test_model_prediction(model, right_features)
    
    # Test 3: Looking left
    print("\n3️⃣ Test: Looking left")
    left_features = [0.3, 0.5, 0.3, 0.5, -15.0, 0.0, 0.0]  # Head turned left, eyes left
    model_utils.test_model_prediction(model, left_features)
    
    # Test 4: Looking up
    print("\n4️⃣ Test: Looking up")
    up_features = [0.5, 0.3, 0.5, 0.3, 0.0, -10.0, 0.0]  # Head tilted up, eyes up
    model_utils.test_model_prediction(model, up_features)
    
    # Test 5: Looking down
    print("\n5️⃣ Test: Looking down")
    down_features = [0.5, 0.7, 0.5, 0.7, 0.0, 10.0, 0.0]  # Head tilted down, eyes down
    model_utils.test_model_prediction(model, down_features)

def interactive_prediction_mode(model):
    """Allow user to input custom feature values for prediction."""
    print("\n" + "="*50)
    print("🎯 INTERACTIVE PREDICTION MODE")
    print("="*50)
    print("Enter feature values (or 'quit' to exit):")
    print("Features: [norm_x_L, norm_y_L, norm_x_R, norm_y_R, yaw, pitch, roll]")
    print("Range: Iris positions (0.0-1.0), Head pose (-30 to +30 degrees)")
    print("Example: 0.5,0.5,0.5,0.5,0,0,0  (center gaze)")
    
    while True:
        try:
            user_input = input("\nFeatures> ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            # Parse the input
            features = [float(x.strip()) for x in user_input.split(',')]
            if len(features) != 7:
                print("❌ Please provide exactly 7 feature values")
                continue
            
            # Make prediction
            result = model_utils.test_model_prediction(model, features)
            if result:
                print(f"   ➡️ Predicted screen coordinates: ({result[0]:.1f}, {result[1]:.1f}) pixels")
            
        except ValueError:
            print("❌ Please enter numeric values separated by commas")
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function to demonstrate model loading and usage."""
    print("🎯 Eye Tracker Model Demo")
    print("=" * 30)
    
    # Determine which model to load
    if len(sys.argv) > 1:
        model_filename = sys.argv[1]
        if not os.path.exists(model_filename):
            print(f"❌ Model file not found: {model_filename}")
            model_filename = interactive_model_selection()
    else:
        model_filename = interactive_model_selection()
    
    if not model_filename:
        print("❌ No model selected. Exiting.")
        return
    
    # Load the model
    print(f"\n📤 Loading model: {model_filename}")
    model = model_utils.load_model(model_filename)
    
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Demonstrate model usage
    demonstrate_model_usage(model)
    
    # Interactive mode
    print("\n" + "="*50)
    print("Would you like to try interactive prediction mode? (y/n): ", end="")
    if input().lower().startswith('y'):
        interactive_prediction_mode(model)
    
    print("\n✅ Demo completed successfully!")
    print("\n💡 Integration Tips:")
    print("   - Use model_utils.load_model(filename) to load saved models")
    print("   - Call model.predict(features_array) for predictions")
    print("   - Features should be numpy array shaped (1, 7) for single prediction")
    print("   - Output is [x_coordinate, y_coordinate] in pixels")

if __name__ == "__main__":
    main()
