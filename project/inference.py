import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Optional
import logging

from project.config import Config
from project.model.multi_task_model import MultiTaskModel

class SignalInference:
    """
    Production-grade inference pipeline for signal modulation classification.
    Supports loading trained PyTorch models and performing inference on raw I/Q data.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None, config: Optional[Config] = None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        self.model = MultiTaskModel(self.config)
        
        if model_path:
            self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_path: Union[str, Path]):
        """Load model weights from a checkpoint."""
        logging.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        logging.info("Model loaded successfully.")

    def preprocess(self, signal: np.ndarray) -> torch.Tensor:
        """
        Preprocess raw I/Q signal data.
        Expected input shape: [2, L] or [L, 2] where 2 represents I and Q channels.
        Returns normalized tensor of shape [1, 2, SEQUENCE_LENGTH].
        """
        # Ensure signal is 2-channel
        if signal.shape[0] != 2 and signal.shape[1] == 2:
            signal = signal.T
        
        # Truncate or pad to SEQUENCE_LENGTH
        if signal.shape[1] > self.config.SEQUENCE_LENGTH:
            signal = signal[:, :self.config.SEQUENCE_LENGTH]
        elif signal.shape[1] < self.config.SEQUENCE_LENGTH:
            pad_width = self.config.SEQUENCE_LENGTH - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad_width)), mode='constant')

        # Normalize signal energy
        energy = np.sqrt(np.mean(np.abs(signal)**2))
        if energy > 1e-10:
            signal = signal / energy

        # Convert to tensor
        tensor = torch.from_numpy(signal).float().unsqueeze(0)  # [1, 2, L]
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Perform inference on a single raw signal.
        """
        input_tensor = self.preprocess(signal)
        
        # Forward pass
        outputs = self.model({'data': input_tensor})
        
        # Process modulation classification
        mod_probs = torch.softmax(outputs['modulation_type'], dim=1)
        conf, pred_idx = torch.max(mod_probs, dim=1)
        mod_name = self.config.get_modulation_name(pred_idx.item() + 1)
        
        # Process symbol width
        symbol_width = outputs['symbol_width'].item()
        
        # Process symbol sequence if available
        symbol_sequence = None
        if 'symbol_sequence' in outputs:
            symbol_sequence = outputs['symbol_sequence'].squeeze().cpu().numpy()

        return {
            'modulation_type': mod_name,
            'confidence': conf.item(),
            'symbol_width': symbol_width,
            'symbol_sequence': symbol_sequence,
            'raw_outputs': {k: v.cpu().numpy() for k, v in outputs.items()}
        }

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    inference = SignalInference()
    
    # Generate dummy signal (2 channels: I and Q)
    dummy_signal = np.random.randn(2, 1024)
    result = inference.predict(dummy_signal)
    
    print(f"Prediction: {result['modulation_type']} (Conf: {result['confidence']:.4f})")
    print(f"Estimated Symbol Width: {result['symbol_width']:.2e}")
