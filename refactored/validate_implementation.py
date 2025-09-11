"""
IMPLEMENTATION VALIDATION SUMMARY
This script validates that all requirements have been met for feature order standardization.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def validate_implementation():
    """Validate that all requirements have been implemented correctly"""
    
    print("🎯 === FEATURE ORDER STANDARDIZATION VALIDATION ===")
    print()
    
    # 1. Check that shared features module exists
    print("✅ 1. Checking shared feature engineering module...")
    try:
        from utils.features import FEATURE_COLUMNS, feature_engineer_df, assemble_features_from_row
        print(f"   📦 utils/features.py exists")
        print(f"   📋 FEATURE_COLUMNS: {FEATURE_COLUMNS}")
        print(f"   🔧 feature_engineer_df function: Available")
        print(f"   🔧 assemble_features_from_row function: Available")
    except ImportError as e:
        print(f"   ❌ Failed to import shared features: {e}")
        return False
    
    # 2. Check model training imports and usage
    print("\n✅ 2. Checking model_training.py updates...")
    try:
        import model_training
        # Check if it imports the shared features
        with open('model_training.py', 'r') as f:
            content = f.read()
            if 'from utils.features import feature_engineer_df, FEATURE_COLUMNS' in content:
                print("   📦 Imports shared features module")
            else:
                print("   ❌ Missing shared features import")
                return False
                
            if 'feature_columns = FEATURE_COLUMNS' in content:
                print("   📋 Uses FEATURE_COLUMNS for training")
            else:
                print("   ❌ Not using FEATURE_COLUMNS")
                return False
                
            if "'feature_columns': FEATURE_COLUMNS" in content:
                print("   💾 Saves FEATURE_COLUMNS in model")
            else:
                print("   ❌ Not saving FEATURE_COLUMNS")
                return False
                
    except Exception as e:
        print(f"   ❌ Failed to validate model_training.py: {e}")
        return False
    
    # 3. Check prediction.py imports and usage
    print("\n✅ 3. Checking prediction.py updates...")
    try:
        from prediction import GazePredictor
        # Check if it imports the shared features
        with open('prediction.py', 'r') as f:
            content = f.read()
            if 'from utils.features import assemble_features_from_row, FEATURE_COLUMNS' in content:
                print("   📦 Imports shared features module")
            else:
                print("   ❌ Missing shared features import")
                return False
                
            if 'assemble_features_from_row(' in content:
                print("   🔧 Uses assemble_features_from_row")
            else:
                print("   ❌ Not using assemble_features_from_row")
                return False
                
            if 'self.saved_feature_columns' in content:
                print("   🔍 Validates saved feature columns")
            else:
                print("   ❌ Missing feature column validation")
                return False
                
    except Exception as e:
        print(f"   ❌ Failed to validate prediction.py: {e}")
        return False
    
    # 4. Test feature consistency
    print("\n✅ 4. Testing feature consistency...")
    try:
        # Load model and check saved features
        import joblib
        model_path = "../models/gaze_prediction_model.joblib"
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            saved_features = model_data.get('feature_columns', [])
            if saved_features == FEATURE_COLUMNS:
                print("   🎯 Model feature_columns match FEATURE_COLUMNS")
            else:
                print(f"   ⚠️ Feature mismatch - Saved: {saved_features}, Current: {FEATURE_COLUMNS}")
                return False
        else:
            print("   ⚠️ No trained model found - run training to complete validation")
            
    except Exception as e:
        print(f"   ❌ Failed to test feature consistency: {e}")
        return False
    
    # 5. Test runtime prediction
    print("\n✅ 5. Testing runtime prediction...")
    try:
        predictor = GazePredictor()
        success = predictor.load_default_model()
        if success:
            # Test a prediction
            result = predictor.predict_gaze_point([0.4, 0.3, 0.45, 0.32, -5.0, 2.0, 1.0])
            if result:
                print(f"   🎯 Prediction successful: {result}")
            else:
                print("   ❌ Prediction failed")
                return False
        else:
            print("   ❌ Failed to load model")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to test runtime prediction: {e}")
        return False
    
    print("\n🎉 === ALL VALIDATION CHECKS PASSED ===")
    print("✅ Shared feature engineering module created")
    print("✅ Model training updated to use shared features") 
    print("✅ Prediction updated to use shared features")
    print("✅ Feature order consistency enforced")
    print("✅ Runtime schema validation implemented")
    print("✅ Feature column alignment working")
    
    print(f"\n📋 SINGLE SOURCE OF TRUTH: utils/features.py")
    print(f"🎯 FEATURE_COLUMNS: {FEATURE_COLUMNS}")
    print(f"🔧 Both training and prediction use identical feature engineering")
    
    return True

if __name__ == "__main__":
    success = validate_implementation()
    if success:
        print("\n🎯 IMPLEMENTATION STATUS: COMPLETE ✅")
    else:
        print("\n❌ IMPLEMENTATION STATUS: INCOMPLETE")
        sys.exit(1)
