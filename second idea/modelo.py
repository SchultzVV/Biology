import torch.nn as nn

# ------------------DEFINE O MODELO----------------------------------------------
# -------------------------------------------------------------------------------
class Linear(nn.Module):
    def __init__(self):#,n_examples):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30, 720),
            nn.ReLU(),
            nn.Linear(720, 1500),
            nn.ReLU(),
            nn.Linear(1500, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

