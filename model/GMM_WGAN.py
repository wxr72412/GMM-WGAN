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

    def __init__(self, Y, num_components):
        super(Generator, self).__init__()
        self.num_samples, self.num_varibales = Y.shape
        self.num_components = num_components
        # print("self.num_samples:", self.num_samples, sep="\n")  # torch.Size([64, |X|])
        # print("self.num_varibales:", self.num_varibales, sep="\n")  # torch.Size([64, |X|])
        # print("self.num_components:", self.num_components, sep="\n")  # torch.Size([64, |X|])

        # self.num_components = 3
        # self.num_varibales = 2

        self.mu = nn.Parameter(torch.rand(self.num_components, self.num_varibales))
        self.cov = nn.Parameter(torch.eye(self.num_varibales).repeat(self.num_components, 1, 1))
        self.weight = nn.Parameter(torch.ones(self.num_components))
        # print(Y)  # torch.Size([K, N])
        # print(Y.shape)  # torch.Size([K, N])
        # print(self.mu) # torch.Size([K, N])
        # print(self.mu.shape)
        # print(self.cov)
        # print(self.cov.shape) # torch.Size([K, N, N])
        # print(self.weight)
        # print(self.weight.shape) # torch.Size([K])
        # exit(0)

        self.Dim_x_mu_cov_alpha = self.num_varibales + \
                                  self.num_components * self.num_varibales + \
                                  self.num_components * (self.num_varibales**2) + \
                                  self.num_components
        # print("Dim_x_mu_cov_alpha:", self.Dim_x_mu_cov_alpha, sep="\n")  # torch.Size([64, |X|])

        main = nn.Sequential(
            nn.Linear(self.Dim_x_mu_cov_alpha, Dim_Hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(Dim_Hidden, Dim_Hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(Dim_Hidden, Dim_Hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(Dim_Hidden, Dim_G_Output),
            nn.Tanh(),
        )
        self.main = main

    def forward(self, x):
        expand_mu = self.mu.reshape(1, -1).expand(x.shape[0], self.num_components * self.num_varibales)
        expand_cov = self.cov.reshape(1, -1).expand(x.shape[0], self.num_components * (self.num_varibales**2))
        expand_weight = self.weight.reshape(1, -1).expand(x.shape[0], self.num_components)
        x_mu_cov_alpha = torch.cat((x, expand_mu, expand_cov, expand_weight), dim=1)
        # print("x.shape:", x.shape, sep="\n")  # torch.Size([64, |X|])
        # print("expand_mu.shape:", expand_mu.shape, sep="\n")  # torch.Size([64, |X|])
        # print("expand_cov.shape:", expand_cov.shape, sep="\n")  # torch.Size([64, |X|])
        # print("expand_weight.shape:", expand_weight.shape, sep="\n")  # torch.Size([64, |X|])
        # print("shape:", x_mu_cov_alpha.shape, sep="\n")  # torch.Size([64, |X|])
        # exit(0)
        output = self.main(x_mu_cov_alpha)
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
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
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