import torch.nn as nn

class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.Sequential(
            nn.Linear(1, 500),
            nn.ELU(),
            nn.Linear(500, 1),
        ) 
        self.f2 = nn.Sequential(
            nn.Linear(1, 500),
            nn.ELU(),
            nn.Linear(500, 1),
        )        
    def forward(self, x, mode):
        if mode ==1:
            return self.f1(x)
        if mode ==2:
            return self.f2(x)