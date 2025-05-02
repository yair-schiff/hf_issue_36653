import torch

from .modules import ResBlock


class MyBackbone(torch.nn.Module):

    def __init__(
        self,
        num_layers: int = 2,
        input_dim: int = 2,
        hidden_dim: int = 128,
        output_dim: int = 2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define the layers of the backbone
        layers = [torch.nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_layers):
            layers.append(ResBlock(hidden_dim))
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        # Forward pass through the backbone
        return self.model(x)
 