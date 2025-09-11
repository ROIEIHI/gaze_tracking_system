from typing import List, Tuple
import pandas as pd
import numpy as np

# This is the single source of truth for feature creation and column order.
# IMPORTANT: feature_columns ORDER must match exactly how the scaler/model were fit.
FEATURE_COLUMNS: List[str] = [
    'norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R',
    'yaw', 'pitch', 'roll',
    'x_yaw_interaction', 'y_pitch_interaction'
]

def feature_engineer_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply identical feature engineering used for both training and prediction."""
    df = df.copy()
    # Averages for interactions, not part of final features unless explicitly included in FEATURE_COLUMNS
    df['avg_norm_x'] = (df['norm_x_L'] + df['norm_x_R']) / 2.0
    df['avg_norm_y'] = (df['norm_y_L'] + df['norm_y_R']) / 2.0
    
    # Pitch source selection for backward compatibility with legacy CSVs
    if 'pitch' in df.columns:
        pitch_src = df['pitch']
    elif 'pitch_adj' in df.columns:
        pitch_src = df['pitch_adj']  # Legacy compatibility
        df['pitch'] = df['pitch_adj']  # Create pitch column for consistency
    else:
        pitch_src = df['pitch_raw']  # Fallback to raw pitch
        df['pitch'] = df['pitch_raw']
    
    # Interactions as used in training
    df['x_yaw_interaction'] = df['avg_norm_x'] * df['yaw']
    df['y_pitch_interaction'] = df['avg_norm_y'] * pitch_src
    
    return df

def assemble_features_from_row(
    norm_x_L: float, norm_y_L: float,
    norm_x_R: float, norm_y_R: float,
    yaw: float, pitch_raw: float, roll: float,
    pitch_baseline: float = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build a one-row DataFrame and engineered features in the exact FEATURE_COLUMNS order."""
    
    # Compute adjusted pitch (now stored as 'pitch')
    pitch = pitch_raw - pitch_baseline if pitch_baseline is not None else pitch_raw
    
    base = pd.DataFrame([{
        'norm_x_L': norm_x_L,
        'norm_y_L': norm_y_L,
        'norm_x_R': norm_x_R,
        'norm_y_R': norm_y_R,
        'yaw': yaw,
        'pitch': pitch,  # Use adjusted pitch, now stored as 'pitch'
        'roll': roll
    }])
    eng = feature_engineer_df(base)
    return eng, eng[FEATURE_COLUMNS].values
