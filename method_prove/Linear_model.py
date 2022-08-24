import torch.nn as nn

class Linear(nn.Module):
    def __init__(self):
        # N, 50
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 200),
            nn.ELU(),
            nn.Linear(200, 500),
            nn.ELU(),
            nn.Linear(500, 500),
            nn.ELU(),
            nn.Linear(500, 200),
            nn.ELU(),
            nn.Linear(200, 1),
            nn.ELU(),
        )        
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded