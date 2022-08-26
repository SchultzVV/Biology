import torch.nn as nn

class Linear(nn.Module):
    def __init__(self):
        # N, 50
        super().__init__()
        self.f1 = nn.Sequential(
            nn.Linear(1, 200),
            nn.ELU(),
            nn.Linear(200, 1),
        ) 
        self.f2 = nn.Sequential(
            nn.Linear(1, 200),
            nn.ELU(),
            nn.Linear(200, 1),
        )        
    def forward(self, x, mode):
        if mode ==1:
            return self.f1(x)
        if mode ==2:
            return self.f2(x)