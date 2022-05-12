import torch
import torch.nn as nn
import torch.optim as optim
#-------------------------------------------------------------------------------
#------------------DEFINE O MODELO----------------------------------------------
#-------------------------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_examples,3000),
            nn.ELU(),
            nn.Linear(3000,2000),
            nn.ELU(),
            nn.Linear(2000,1000),
            nn.ELU(),
            nn.Linear(1000,500),
            nn.ELU(),
            nn.Linear(500,50),
            nn.ELU(),
            nn.Linear(50,1),
            nn.Softmax()
        )