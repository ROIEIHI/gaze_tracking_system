# Pitch Baseline Implementation Summary

## Overview
Successfully implemented pitch baseline measurement and adjustment to fix vertical accuracy issues caused by camera tilt differences between training and live runs.

## Key Features Implemented

### 1. Pitch Baseline Measurement
- **Method**: `measure_pitch_baseline(cap, seconds=1.5, min_samples=30)`
- **Function**: Collects pitch samples while user fixates screen center
- **Output**: Computes median pitch and stores in `self.pitch_baseline`
- **Fallback**: Proceeds without baseline if insufficient samples

### 2. Session Baseline Integration
- **Location**: Called in `real_time_prediction()` after camera setup
- **UX**: Shows full-screen prompt with centered dot and instructions
- **Timing**: Brief 1.5-2 second measurement before main prediction loop
- **Compatibility**: Works with both standard and text_analysis modes

### 3. Live Pitch Adjustment
- **Location**: In `predict_gaze_point()` before feature engineering
- **Process**: `pitch = pitch - self.pitch_baseline`
- **Target**: Affects `y_pitch_interaction = avg_norm_y × adjusted_pitch`
- **Preservation**: Original features array unchanged, only local adjustment

### 4. Debug Diagnostics
- **DEBUG_FEATURES=1**: 
  - Prints baseline value once per session
  - Shows pitch_raw and pitch_adjusted for first few frames
  - Displays feature engineering details
- **DEBUG_CLAMP=1**:
  - Logs when smoothed_y hits window bounds (0 or height-1)
  - Helps identify Y-coordinate clamping issues

### 5. Non-Functional Constraints Maintained
- ✅ No changes to training code or `FEATURE_COLUMNS`
- ✅ No baseline data written to saved model files (session-only)
- ✅ Existing Kalman and smoothing behavior preserved
- ✅ Current X flip behavior maintained, Y not flipped
- ✅ Strict column order and reindexing maintained

## Acceptance Checklist ✅

1. ✅ **measure_pitch_baseline** method exists and called once per session
2. ✅ **Pitch adjustment** applied before `assemble_features_from_row`
3. ✅ **No training changes** - model schema unchanged
4. ✅ **Debug output** working for both DEBUG_FEATURES and DEBUG_CLAMP
5. ✅ **Graceful fallback** when baseline measurement fails
6. ✅ **Text analysis compatibility** - reuses same cap for baseline

## Expected Impact

The pitch baseline correction should:
- **Re-center** live pitch distribution to match training data
- **Restore Y variance** that was collapsed due to camera tilt offset
- **Improve vertical accuracy** by fixing the `y_pitch_interaction` feature
- **Maintain horizontal accuracy** (X predictions unchanged)

## Usage

1. **Standard mode**: Baseline measured automatically before prediction starts
2. **Text analysis mode**: Baseline measured on same full-screen window before text content
3. **Debug mode**: Set `DEBUG_FEATURES=1` or `DEBUG_CLAMP=1` for detailed diagnostics
4. **Fallback**: System continues normally if baseline measurement fails

## Files Modified

- `prediction.py`: Core implementation of baseline measurement and adjustment
- `test_pitch_baseline.py`: Comprehensive validation test script

The implementation is ready for testing with live camera data to validate the vertical accuracy improvement.
