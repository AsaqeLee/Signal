import pytest
import numpy as np
import torch
from project.inference import SignalInference
from project.config import Config

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def inference(config):
    return SignalInference(config=config)

def test_preprocess_shape(inference, config):
    # Test with [2, 2048]
    signal = np.random.randn(2, 2048)
    tensor = inference.preprocess(signal)
    assert tensor.shape == (1, 2, config.SEQUENCE_LENGTH)
    
    # Test with [512, 2]
    signal = np.random.randn(512, 2)
    tensor = inference.preprocess(signal)
    assert tensor.shape == (1, 2, config.SEQUENCE_LENGTH)

def test_preprocess_normalization(inference):
    # Test energy normalization
    signal = np.random.randn(2, 1024) * 10
    tensor = inference.preprocess(signal)
    energy = torch.sqrt(torch.mean(torch.abs(tensor)**2))
    assert torch.allclose(energy, torch.tensor(1.0), atol=1e-5)

def test_predict_structure(inference):
    signal = np.random.randn(2, 1024)
    result = inference.predict(signal)
    
    assert 'modulation_type' in result
    assert 'confidence' in result
    assert 'symbol_width' in result
    assert isinstance(result['confidence'], float)
    assert 0 <= result['confidence'] <= 1

def test_config_modulation_mapping(config):
    name = config.get_modulation_name(1)
    assert name == 'BPSK'
    idx = config.get_modulation_type('BPSK')
    assert idx == 1
