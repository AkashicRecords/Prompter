"""
Unit tests for the RL model module.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Import test utilities
from test_utils import (
    temp_dir, 
    temp_file,
    device,
    validate_tensor,
    create_validated_mock
)

# Import will be available after implementation
with pytest.raises(ImportError):
    from prompter.pattern import PromptPattern
    from prompter.rl_model import PromptRL

# Mock classes for testing
class MockPromptPattern:
    """Mock PromptPattern class for testing."""
    
    def __init__(
        self,
        phrase=None,
        context=None,
        flags=None,
        frequency=0,
        success_rate=0.0,
        id=None,
        template=None,
        variables=None,
        score=0.0
    ):
        """Initialize a mock pattern."""
        # Support both old and new style initialization
        self.phrase = phrase
        self.context = context
        self.flags = flags or []
        self.frequency = frequency
        self.success_rate = success_rate
        
        # New style properties
        self.id = id or "mock-pattern-1"
        self.template = template or "Test template"
        self.variables = variables or []
        self.score = score
    
    def to_dict(self):
        """Convert pattern to a dictionary."""
        return {
            "id": self.id,
            "template": self.template,
            "variables": self.variables,
            "score": self.score,
            # Include old style properties for backward compatibility
            "phrase": self.phrase,
            "context": self.context,
            "flags": self.flags,
            "frequency": self.frequency,
            "success_rate": self.success_rate
        }
    
    def __eq__(self, other):
        """Check if two patterns are equal."""
        if not isinstance(other, MockPromptPattern):
            return False
        return self.id == other.id


# Create a validated model class
class MockPromptRL:
    """Mock PromptRL class for testing."""
    
    def __init__(self, input_size=768, hidden_size=256, output_size=64, learning_rate=0.001, device=None):
        """Initialize a mock RL model."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.device = device or torch.device("cpu")
        
        # Initialize memory
        self.memory = []
        
        # Initialize PyTorch model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def add_to_memory(self, pattern, reward):
        """Add a pattern to memory with its reward."""
        if not isinstance(pattern, MockPromptPattern):
            raise TypeError(f"Pattern must be a PromptPattern, got {type(pattern)}")
        
        if not isinstance(reward, (int, float)):
            raise TypeError(f"Reward must be a number, got {type(reward)}")
        
        self.memory.append((pattern, reward))
        
        # Limit memory size
        if len(self.memory) > 1000:
            self.memory.pop(0)
    
    def _encode_pattern(self, pattern):
        """Encode a pattern for the model."""
        if not isinstance(pattern, MockPromptPattern):
            raise TypeError(f"Pattern must be a PromptPattern, got {type(pattern)}")
        
        # Mock encoding - in reality, this would do something more complex
        mock_tensor = torch.randn(self.input_size, device=self.device)
        return mock_tensor
    
    def train(self, epochs=10, batch_size=32):
        """Train the model on memory."""
        if len(self.memory) < batch_size:
            raise ValueError(f"Not enough patterns in memory (need {batch_size}, have {len(self.memory)})")
        
        losses = []
        
        for epoch in range(epochs):
            # Sample batch from memory
            batch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
            batch = [self.memory[i] for i in batch_indices]
            
            # Extract patterns and rewards
            patterns = [item[0] for item in batch]
            rewards = torch.tensor([item[1] for item in batch], device=self.device)
            
            # Encode patterns
            encoded_patterns = torch.stack([self._encode_pattern(p) for p in patterns])
            
            # Forward pass
            outputs = self.model(encoded_patterns)
            
            # Compute loss (MSE between output and reward)
            loss = torch.nn.functional.mse_loss(outputs.squeeze(), rewards)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
        
        return losses
    
    def predict(self, pattern):
        """Predict reward for a pattern."""
        if not isinstance(pattern, MockPromptPattern):
            raise TypeError(f"Pattern must be a PromptPattern, got {type(pattern)}")
        
        # Encode pattern
        encoded = self._encode_pattern(pattern)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            output = self.model(encoded)
        self.model.train()
        
        return output.item()
    
    def save(self, path):
        """Save the model to a file."""
        if not isinstance(path, (str, Path)):
            raise TypeError(f"Path must be a string or Path, got {type(path)}")
        
        path = Path(path) if isinstance(path, str) else path
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate
        }, path)
    
    def load(self, path):
        """Load the model from a file."""
        if not isinstance(path, (str, Path)):
            raise TypeError(f"Path must be a string or Path, got {type(path)}")
        
        path = Path(path) if isinstance(path, str) else path
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model
        checkpoint = torch.load(path, map_location=self.device)
        
        # Update model parameters
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.output_size = checkpoint['output_size']
        self.learning_rate = checkpoint['learning_rate']
        
        # Recreate model with loaded parameters
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.output_size)
        ).to(self.device)
        
        # Load state dictionaries
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Recreate optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


@pytest.fixture
def sample_pattern():
    """Create a sample pattern for testing."""
    return MockPromptPattern(
        phrase="test phrase",
        context="test context",
        flags=["test"],
        frequency=10,
        success_rate=0.8,
        template="This is a {test} template",
        variables=["test"],
        score=0.75
    )


@pytest.fixture
def mock_rl_model(device):
    """Create a mock RL model for testing."""
    with patch("prompter.rl_model.PromptRL", MockPromptRL):
        from prompter.rl_model import PromptRL
        model = PromptRL(device=device)
        yield model


class TestPromptRL:
    """Test the PromptRL class."""
    
    def test_initialization(self, mock_rl_model, device):
        """Test initializing the PromptRL class."""
        # Check model is a PyTorch module
        assert isinstance(mock_rl_model.model, torch.nn.Module), "Model should be a PyTorch module"
        
        # Check device
        assert mock_rl_model.device == device, "Device should match the provided device"
        assert mock_rl_model.model[0].weight.device == device, "Model parameters should be on the correct device"
        
        # Check memory initialization
        assert isinstance(mock_rl_model.memory, list), "Memory should be a list"
        assert len(mock_rl_model.memory) == 0, "Memory should be empty initially"
        
        # Check optimizer initialization
        assert isinstance(mock_rl_model.optimizer, torch.optim.Optimizer), "Optimizer should be initialized"
    
    def test_add_to_memory(self, mock_rl_model, sample_pattern):
        """Test adding a pattern to memory."""
        # Add a pattern to memory
        mock_rl_model.add_to_memory(sample_pattern, 0.8)
        
        # Check memory contents
        assert len(mock_rl_model.memory) == 1, "Memory should contain one item"
        assert mock_rl_model.memory[0][0] == sample_pattern, "Pattern should be added to memory"
        assert mock_rl_model.memory[0][1] == 0.8, "Reward should be added to memory"
        assert isinstance(mock_rl_model.memory[0][1], float), "Reward should be a float"
        
        # Test adding with invalid pattern type
        with pytest.raises(TypeError):
            mock_rl_model.add_to_memory("not a pattern", 0.5)
        
        # Test adding with invalid reward type
        with pytest.raises(TypeError):
            mock_rl_model.add_to_memory(sample_pattern, "not a number")
    
    def test_memory_limit(self, mock_rl_model, sample_pattern):
        """Test that memory is limited to 1000 items."""
        # Add 1001 items to memory
        for i in range(1001):
            mock_rl_model.add_to_memory(sample_pattern, 0.5)
        
        # Check memory limit
        assert len(mock_rl_model.memory) == 1000, "Memory should be limited to 1000 items"
    
    def test_encode_pattern(self, mock_rl_model, sample_pattern, device):
        """Test encoding a pattern for the model."""
        # Encode a pattern
        encoded = mock_rl_model._encode_pattern(sample_pattern)
        
        # Check encoding
        assert isinstance(encoded, torch.Tensor), "Encoded pattern should be a PyTorch tensor"
        assert encoded.device == device, "Encoded pattern should be on the correct device"
        assert encoded.dtype == torch.float32, "Encoded pattern should have float32 data type"
        assert encoded.shape == (mock_rl_model.input_size,), "Encoded pattern should have correct shape"
        
        # Test with invalid pattern type
        with pytest.raises(TypeError):
            mock_rl_model._encode_pattern("not a pattern")
    
    def test_train(self, mock_rl_model, sample_pattern):
        """Test training the model."""
        # Add patterns to memory
        for i in range(50):
            mock_rl_model.add_to_memory(sample_pattern, 0.5 + i * 0.01)
        
        # Train the model
        losses = mock_rl_model.train(epochs=5, batch_size=32)
        
        # Check training results
        assert isinstance(losses, list), "Training should return a list of losses"
        assert len(losses) == 5, "Should have one loss per epoch"
        assert all(isinstance(loss, float) for loss in losses), "All losses should be floats"
        
        # Test with insufficient memory
        mock_rl_model.memory = []
        with pytest.raises(ValueError):
            mock_rl_model.train(batch_size=32)
    
    def test_predict(self, mock_rl_model, sample_pattern):
        """Test predicting reward for a pattern."""
        # Predict reward
        reward = mock_rl_model.predict(sample_pattern)
        
        # Check prediction
        assert isinstance(reward, float), "Prediction should be a float"
        
        # Test with invalid pattern type
        with pytest.raises(TypeError):
            mock_rl_model.predict("not a pattern")
    
    def test_save_load(self, mock_rl_model, sample_pattern, temp_file):
        """Test saving and loading the model."""
        # Add some patterns to memory
        for i in range(10):
            mock_rl_model.add_to_memory(sample_pattern, 0.5 + i * 0.05)
        
        # Train the model to get some weights
        mock_rl_model.train(epochs=2, batch_size=8)
        
        # Get prediction before saving
        before_save = mock_rl_model.predict(sample_pattern)
        
        # Save the model
        mock_rl_model.save(temp_file)
        
        # Verify file exists
        assert temp_file.exists(), "Save file should exist"
        
        # Create a new model instance
        new_model = MockPromptRL(device=mock_rl_model.device)
        
        # Load the model
        new_model.load(temp_file)
        
        # Get prediction after loading
        after_load = new_model.predict(sample_pattern)
        
        # Check that predictions match
        assert abs(before_save - after_load) < 1e-5, "Predictions should match after loading"
        
        # Test saving with invalid path type
        with pytest.raises(TypeError):
            mock_rl_model.save(123)
        
        # Test loading with invalid path type
        with pytest.raises(TypeError):
            new_model.load(123)
        
        # Test loading non-existent file
        with pytest.raises(FileNotFoundError):
            new_model.load(Path("non_existent_file.pt"))


@pytest.mark.integration
class TestRLModelIntegration:
    """Integration tests for the RL model."""
    
    def test_integration_with_pattern_module(self):
        """Test integration with the pattern module."""
        # This test will be implemented after both modules are created
        pass
    
    def test_training_with_real_patterns(self):
        """Test training the model with real patterns from a corpus."""
        # This test will be implemented after the module is created
        pass 