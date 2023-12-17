import torch
import torch.nn as nn
import torch.autograd as autograd

import hyperPara

Dim_G_Input = hyperPara.Dim_G_Input
Dim_G_Output = hyperPara.Dim_G_Output
Dim_Hidden = hyperPara.Dim_Hidden  # Model dimensionality
Dim_D_Input = hyperPara.Dim_D_Input
Dim_D_Output = hyperPara.Dim_D_Output

FIXED_GENERATOR = hyperPara.FIXED_GENERATOR  # whether to hold the generator fixed at real data plus
BATCH_SIZE = hyperPara.BATCH_SIZE  # Batch size
use_cuda = hyperPara.use_cuda
LAMBDA = hyperPara.LAMBDA

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(Dim_G_Input, Dim_Hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(Dim_Hidden, Dim_Hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(Dim_Hidden, Dim_Hidden),
            nn.LeakyReLU(0.2, inplace=True),
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
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(Dim_Hidden, Dim_Hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(Dim_Hidden, Dim_Hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(Dim_Hidden, Dim_D_Output),
            # nn.Sigmoid(),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        # print(output.shape)
        # print(output.view(-1).shape)
        # exit(0)
        return output


def calc_gradient_penalty(netD, real_data, fake_data):

    # alpha = torch.rand(BATCH_SIZE, 1)
    alpha = torch.rand(real_data.size()[0], 1)
    # print(alpha)
    alpha = alpha.expand(real_data.size())
    # print(alpha)
    # exit(0)

    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty