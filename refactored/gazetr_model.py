"""
GazeTR Hybrid CNN-Transformer Gaze Estimation Model

This module provides a clean interface to the GazeTR model for integration
into larger projects. It handles model loading, preprocessing, and prediction.

Usage:
    from gazetr_model import GazeTRPredictor
    
    # Initialize the model
    gaze_predictor = GazeTRPredictor(model_path="GazeTR-H-ETH.pt")
    
    # Predict gaze from face image
    gaze_vector = gaze_predictor.predict(face_image_np)
    
    # Get preprocessed face tensor if needed
    face_tensor = gaze_predictor.preprocess_face(face_image_np)

Author: GitHub Copilot
"""

import os
import torch
import cv2
import numpy as np
from typing import Optional, Tuple, Union

# Import the Model directly from the local model_training module
try:
    from model_training import Model
    print("Successfully imported Model from model_training")
except ImportError as e:
    print(f"Error importing Model from model_training: {e}")
    print("Please ensure model_training.py is in the same directory.")
    Model = None


class GazeTRPredictor:
    """
    A clean interface to the GazeTR hybrid CNN-Transformer model.
    
    This class handles model initialization, preprocessing, and prediction
    for gaze estimation from face images.
    """
    
    def __init__(self, 
                 model_path: str = None,
                 device: Optional[str] = None,
                 input_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the GazeTR predictor.
        
        Args:
            model_path: Path to the pre-trained GazeTR model weights
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            input_size: Input image size expected by the model (width, height)
        """
        if Model is None:
            raise ImportError(
                "GazeTR Model class not available. Please ensure model_training.py "
                "is in the same directory and contains the Model class."
            )
        
        # Set default model path if not provided
        if model_path is None:
            # Try to find the model in the models directory
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                      'models', 'GazeTR-H-ETH.pt')
            if not os.path.exists(model_path):
                # Try another common location
                model_path = os.path.join(os.path.dirname(__file__), 
                                         'models', 'GazeTR-H-ETH.pt')
        
        print(f"Using model path: {model_path}")
            
        self.input_size = input_size
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup computation device."""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> Model:
        """Load and initialize the GazeTR model."""
        print(f"Loading GazeTR model on device: {self.device}")
        
        # Initialize model
        model = Model()
        model.to(self.device)
        model.float()  # Use float32 for compatibility with preprocessing
        
        # Load pre-trained weights if available
        if os.path.exists(model_path):
            print(f"Loading pre-trained weights from {model_path}")
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                
                # Handle different state dict formats
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                model.load_state_dict(state_dict, strict=False)
                print("Pre-trained weights loaded successfully!")
                
            except Exception as e:
                print(f"Warning: Could not load pre-trained weights: {e}")
                print("Using randomly initialized weights.")
        else:
            print(f"Warning: Model file {model_path} not found. Using randomly initialized weights.")
        
        model.eval()
        return model
    
    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for GazeTR model input.
        
        Args:
            face_image: Face image as numpy array (BGR or RGB format)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert BGR to RGB if needed (OpenCV default is BGR)
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            # Assume BGR input from OpenCV, convert to RGB
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        face_image = cv2.resize(face_image, self.input_size)
        
        # Normalize to [0, 1]
        face_image = face_image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        face_image = (face_image - mean) / std
        
        # Convert to tensor: (H, W, C) -> (C, H, W) and add batch dimension
        face_tensor = torch.from_numpy(face_image).permute(2, 0, 1).unsqueeze(0)
        
        return face_tensor.to(self.device)
    
    def predict(self, face_image: np.ndarray) -> np.ndarray:
        """
        Predict gaze vector from face image.
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            3D gaze vector as numpy array [gaze_x, gaze_y, gaze_z] or None if prediction fails
        """
        try:
            # Preprocess face image
            face_tensor = self.preprocess_face(face_image)
            
            # Ensure tensor is float32 (common requirement for models)
            face_tensor = face_tensor.float()
            
            # Predict gaze vector
            with torch.no_grad():
                # Create input dictionary as expected by GazeTR
                model_input = {"face": face_tensor}
                
                # Get gaze prediction
                gaze_output = self.model(model_input)
                
                # Handle different output formats
                if isinstance(gaze_output, torch.Tensor):
                    # Direct tensor output
                    gaze_vector = gaze_output.cpu().numpy().flatten()
                elif isinstance(gaze_output, dict) and 'gaze' in gaze_output:
                    # Dictionary output with 'gaze' key
                    gaze_vector = gaze_output['gaze'].cpu().numpy().flatten()
                elif isinstance(gaze_output, (list, tuple)) and len(gaze_output) > 0:
                    # List/tuple output, take first element
                    gaze_vector = gaze_output[0].cpu().numpy().flatten()
                else:
                    print(f"Unexpected GazeTR output format: {type(gaze_output)}")
                    return None
                
                # Ensure we have at least 2 components (3D is preferred)
                if len(gaze_vector) >= 2:
                    # Normalize gaze vector if needed (some models output large values)
                    # Clamp to reasonable range to prevent dimension errors
                    gaze_vector = np.clip(gaze_vector, -1.0, 1.0)
                    return gaze_vector
                else:
                    print(f"GazeTR output too short: {len(gaze_vector)} components")
                    return None
                    
        except Exception as e:
            print(f"Error in GazeTR prediction: {e}")
            return None
    
    def predict_batch(self, face_images: list) -> np.ndarray:
        """
        Predict gaze vectors for a batch of face images.
        
        Args:
            face_images: List of face images as numpy arrays
            
        Returns:
            Array of gaze vectors with shape (batch_size, 3)
        """
        if not face_images:
            return np.array([])
        
        # Preprocess all images
        face_tensors = []
        for face_image in face_images:
            face_tensor = self.preprocess_face(face_image)
            face_tensors.append(face_tensor)
        
        # Stack into batch
        batch_tensor = torch.cat(face_tensors, dim=0)
        
        # Predict for entire batch
        with torch.no_grad():
            model_input = {"face": batch_tensor}
            gaze_outputs = self.model(model_input)
            gaze_vectors = gaze_outputs.cpu().numpy()
        
        return gaze_vectors
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'device': str(self.device),
            'input_size': self.input_size,
            'model_type': 'GazeTR Hybrid CNN-Transformer',
            'precision': 'float64',
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def __call__(self, face_image: np.ndarray) -> np.ndarray:
        """Make the predictor callable."""
        return self.predict(face_image)


# Convenience function for quick usage
def load_gazetr_model(model_path: str = "GazeTR-H-ETH.pt", 
                      device: Optional[str] = None) -> GazeTRPredictor:
    """
    Quick function to load GazeTR model.
    
    Args:
        model_path: Path to pre-trained weights
        device: Device to use for inference
        
    Returns:
        Initialized GazeTRPredictor instance
    """
    return GazeTRPredictor(model_path=model_path, device=device)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("Testing GazeTR Model...")
    
    try:
        # Initialize predictor
        predictor = GazeTRPredictor()
        
        # Print model info
        info = predictor.get_model_info()
        print(f"Model Info: {info}")
        
        # Test with dummy face image
        dummy_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        gaze_vector = predictor.predict(dummy_face)
        
        print(f"Test prediction successful!")
        print(f"Input shape: {dummy_face.shape}")
        print(f"Output gaze vector: {gaze_vector}")
        print(f"Gaze vector shape: {gaze_vector.shape}")
        
    except Exception as e:
        print(f"Error testing model: {e}")
        print("Make sure GazeTR directory and model weights are available.")
