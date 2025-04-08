import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any
import numpy as np
from .cli import PromptPattern

class PromptNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 64):
        super(PromptNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class PromptRL:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = 256  # Size of the input embedding
        self.model = PromptNetwork(self.input_size).to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.optimizer = optim.Adam(self.model.parameters())
        self.memory = []
    
    def _encode_pattern(self, pattern: PromptPattern) -> torch.Tensor:
        # Simple encoding for now - can be improved with better embeddings
        features = []
        features.extend([1.0 if flag in pattern.flags else 0.0 for flag in ["ports", "IDs", "errors"]])
        features.append(pattern.frequency / 100.0)  # Normalize frequency
        features.append(pattern.success_rate)
        # Pad or truncate to input_size
        features = features[:self.input_size] + [0.0] * (self.input_size - len(features))
        return torch.tensor(features, dtype=torch.float32).to(self.device)
    
    def add_to_memory(self, pattern: PromptPattern, reward: float):
        self.memory.append((pattern, reward))
        if len(self.memory) > 1000:  # Limit memory size
            self.memory.pop(0)
    
    def train_step(self):
        if len(self.memory) < 2:
            return
        
        # Sample a batch from memory
        batch_size = min(32, len(self.memory))
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        # Prepare batch data
        patterns = [p[0] for p in batch]
        rewards = torch.tensor([p[1] for p in batch], dtype=torch.float32).to(self.device)
        
        # Forward pass
        inputs = torch.stack([self._encode_pattern(p) for p in patterns])
        outputs = self.model(inputs)
        
        # Compute loss (simple MSE for now)
        loss = nn.MSELoss()(outputs.squeeze(), rewards)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def suggest_improvements(self, pattern: PromptPattern) -> List[str]:
        with torch.no_grad():
            input_tensor = self._encode_pattern(pattern)
            output = self.model(input_tensor)
            
            # Convert model output to suggestions
            # This is a simple implementation - can be improved
            suggestions = []
            if output[0] > 0.5:
                suggestions.append("Consider adding more context")
            if output[1] > 0.5:
                suggestions.append("Try adding specific examples")
            if output[2] > 0.5:
                suggestions.append("Include error handling")
            
            return suggestions
    
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path) 