import hyperPara

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import GMM.gmm as GMM

Dim_Hidden = hyperPara.Dim_Hidden  # Model dimensionality

Dim_G_Output = hyperPara.Dim_G_Output

Dim_D_Input = hyperPara.Dim_D_Input
Dim_D_Output = hyperPara.Dim_D_Output
use_cuda = hyperPara.use_cuda

class Generator(nn.Module):
    def __init__(self, Y, num_components):
        super(Generator, self).__init__()
        self.num_samples, self.num_varibales = Y.shape
        self.num_components = num_components


        # self.num_samples = 5
        # self.num_varibales = 2
        # self.num_components = 3

        self.mu = nn.Parameter(torch.rand(self.num_components, self.num_varibales))
        self.cov = nn.Parameter(torch.eye(self.num_varibales).repeat(self.num_components, 1, 1))
        # self.alpha = nn.Parameter(torch.ones(self.num_components))
        # self.Dim_x_mu_cov_alpha = 2*self.num_varibales + self.num_varibales**2
        self.Dim_x_mu_cov_alpha = self.num_varibales

        # print(self.mu) # torch.Size([K, N])
        # print(self.mu.shape)
        # print(self.cov)
        # print(self.cov.shape) # torch.Size([K, N, N])
        # print(self.alpha)
        # print(self.alpha.shape) # torch.Size([K])
        # exit(0)


        self.list_Linear_p = []
        for i in range(self.num_components):
            Linear_p = nn.Sequential(
                nn.Linear(self.Dim_x_mu_cov_alpha, Dim_Hidden),
                nn.ReLU(True),
                nn.Linear(Dim_Hidden, Dim_Hidden),
                nn.ReLU(True),
                nn.Linear(Dim_Hidden, Dim_Hidden),
                nn.ReLU(True),
                nn.Linear(Dim_Hidden, Dim_G_Output),
                nn.Tanh(),
            )
            self.list_Linear_p.append(Linear_p)


        # Linear_weight = nn.Sequential(
        #     nn.Linear(self.num_components, Dim_Hidden),
        #     nn.ReLU(True),
        #     nn.Linear(Dim_Hidden, self.num_components),
        #     nn.Sigmoid(),
        # )
        # self.Linear_weight = Linear_weight

    def forward(self, x):
        self.list_mu = []
        self.list_cov = []
        self.list_alpha = []


        # weight
        # alpha = self.Linear_weight(self.alpha) #[0~1]
        # sum = torch.sum(alpha)
        # self.weight = alpha / sum
        # print("alpha:", alpha, sep="\n")
        # print("sum:", sum, sep="\n")
        # print("self.weight:", self.weight, sep="\n")
        # exit(0)

        probabilities = []
        for i in range(self.num_components):
            # print("x:", x, sep="\n")  # torch.Size([64, |X|])
            # print("x.shape:", x.shape, sep="\n")  # torch.Size([64, |X|])
            # print("self.mu[i]:", self.mu[i], sep="\n")  # torch.Size([64, |X|])
            # print("self.mu[i].shape:", self.mu[i].shape, sep="\n")  # torch.Size([64, |X|])
            # print("self.cov[i]:", self.cov[i], sep="\n")  # torch.Size([64, |X|])
            # print("self.cov[i].shape:", self.cov[i].shape, sep="\n")  # torch.Size([64, |X|])
            # print("self.alpha[i]:", self.alpha[i], sep="\n")  # torch.Size([64, |X|])
            # print("self.alpha[i].shape:", self.alpha[i].shape, sep="\n")  # torch.Size([64, |X|])
            expand_mu = self.mu[i].reshape(1, -1).expand(x.shape[0], self.num_varibales)
            expand_cov = self.cov[i].reshape(1, -1).expand(x.shape[0], self.num_varibales * self.num_varibales)
            # expand_weight = self.weight[i].reshape(1, -1).expand(x.shape[0], 1)
            # print("expand_mu:", expand_mu, sep="\n")  # torch.Size([64, |X|])
            # print("expand_cov:", expand_cov, sep="\n")  # torch.Size([64, |X|])
            # print("expand_alpha:", expand_alpha, sep="\n")  # torch.Size([64, |X|])
            x_mu_cov_alpha = torch.cat((x, expand_mu, expand_cov), dim=1)
            # print("x_mu_cov_alpha:", x_mu_cov_alpha, sep="\n")  # torch.Size([64, |X|])
            # print("x_mu_cov_alpha.shape:", x_mu_cov_alpha.shape, sep="\n")  # torch.Size([64, |X|])
            # exit(0)

            prob = self.list_Linear_p[i](x)
            # print("prob:", prob, sep="\n")
            # print("prob.shape:", prob.shape, sep="\n")  # torch.Size([64, 1])
            probabilities.append(prob)
            # exit(0)

        # Calculate the probability for each sample by combining the probabilities of all components
        probabilities = torch.stack(probabilities, dim=1)
        # print(probabilities)
        # print("probabilities.shape:", probabilities.shape, sep="\n")  # torch.Size([64, K])
        # exit(0)

        # 计算每个样本的概率值
        likelihood = torch.sum(probabilities, dim=1)
        # print(likelihood)
        # print("likelihood.shape:", likelihood.shape, sep="\n")  # torch.Size([64, 1])
        # exit(0)
        return likelihood.reshape(x.shape[0], -1)