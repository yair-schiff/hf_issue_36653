import torch


class ResBlock(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.fc = torch.nn.Linear(dim, dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # Apply the first linear layer
        z = self.fc(x)
        # Apply ReLU activation
        z = self.relu(z)
        return x + z
