import hyperPara

import torch
import torch.nn as nn
import torch.autograd as autograd


Dim_G_Input = hyperPara.Dim_G_Input
Dim_G_Output = hyperPara.Dim_G_Output
Dim_Hidden = hyperPara.Dim_Hidden  # Model dimensionality
Dim_D_Input = hyperPara.Dim_D_Input
Dim_D_Output = hyperPara.Dim_D_Output

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(Dim_G_Input, Dim_Hidden),
            nn.ReLU(True),
            nn.Linear(Dim_Hidden, Dim_Hidden),
            nn.ReLU(True),
            nn.Linear(Dim_Hidden, Dim_Hidden),
            nn.ReLU(True),
            nn.Linear(Dim_Hidden, Dim_G_Output),
            nn.Tanh(),
        )
        self.main = main

    def forward(self, noise):
        output = self.main(noise)
        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(Dim_D_Input, Dim_Hidden),
            nn.ReLU(True),
            nn.Linear(Dim_Hidden, Dim_Hidden),
            nn.ReLU(True),
            nn.Linear(Dim_Hidden, Dim_Hidden),
            nn.ReLU(True),
            nn.Linear(Dim_Hidden, Dim_D_Output),
            nn.Sigmoid(),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)