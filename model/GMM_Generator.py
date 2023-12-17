import hyperPara

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import GMM.gmm as GMM

Dim_Hidden = hyperPara.Dim_Hidden  # Model dimensionality
Dim_D_Input = hyperPara.Dim_D_Input
Dim_D_Output = hyperPara.Dim_D_Output
use_cuda = hyperPara.use_cuda

class Generator(nn.Module):
    def __init__(self, Y, num_components):
        super(Generator, self).__init__()
        self.num_samples, self.num_varibales = Y.shape
        self.num_components = num_components

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


    def forward(self, x):
        # weight
        sum = torch.sum(self.weight)
        self.list_weight = self.weight / sum
        print("alpha:", self.weight, sep="\n")
        print("sum:", sum, sep="\n")
        print("self.list_weight:", self.list_weight, sep="\n")
        exit(0)

        self.list_mu = []
        self.list_cov = []
        self.list_alpha = []
        probabilities = []


        if self.num_components >= 1:
            mu1 = self.Linear_mu1(self.list_mu_Tensor[0])
            self.list_mu.append(mu1)
            cov1 = self.Linear_cov1(self.list_cov_Tensor[0].reshape(self.num_varibales, self.num_varibales))
            self.list_cov.append(cov1)

            diff = x - mu1
            # print("x:", x, sep="\n")  # torch.Size([64, |X|])
            # print("mu1:", mu1, sep="\n")  # torch.Size([|X|])
            # print("diff:", diff, sep="\n")  # torch.Size([64, |X|])
            # print("diff.shape:", diff.shape, sep="\n")  # torch.Size([64, |X|])
            # exit(0)

            # print("cov1:", cov1, sep="\n")
            # print("cov1.shape:", cov1.shape, sep="\n")  # torch.Size([num_varibales, num_varibales])
            inverse_cov = torch.inverse(cov1)
            # print("inverse_cov:", inverse_cov, sep="\n")
            # print("inverse_cov.shape:", inverse_cov.shape, sep="\n")  # torch.Size([num_varibales, num_varibales])


            exponent = -0.5 * torch.sum(torch.mul(diff, (torch.matmul(inverse_cov, diff.t())).t()), dim=1)
            # print("inverse_cov:", inverse_cov, sep="\n")  # torch.Size([|X|, |X|])
            # print("torch.matmul(inverse_cov, diff.t()):", torch.matmul(inverse_cov, diff.t()), sep="\n")  # [|X|, |X|] * [|X|, 64] = [|X|, 64]
            # print("torch.matmul(inverse_cov, diff.t()).shape:", torch.matmul(inverse_cov, diff.t()).shape, sep="\n")  # [|X|, 64]
            # print("(torch.matmul(inverse_cov, diff.t())).t().shape:", (torch.matmul(inverse_cov, diff.t())).t().shape, sep="\n")  # [64, |X|]
            # print("torch.mul(diff, (torch.matmul(inverse_cov, diff.t()))):", torch.mul(diff, (torch.matmul(inverse_cov, diff.t())).t()), sep="\n")
            # print("torch.mul(diff, (torch.matmul(inverse_cov, diff.t()))).shape:", torch.mul(diff, (torch.matmul(inverse_cov, diff.t())).t()).shape, sep="\n")  # torch.Size([64, |X|])
            # print("torch.sum(torch.mul(diff, (torch.matmul(inverse_cov, diff.t())).t()), dim=1).shape:",
            #       torch.sum(torch.mul(diff, (torch.matmul(inverse_cov, diff.t())).t()), dim=1).shape, sep="\n")  # torch.Size([64])
            # print("exponent:", exponent, sep="\n")
            # print("exponent.shape:", exponent.shape, sep="\n")  # torch.Size([64])
            # exit(0)


            det_cov = torch.det(cov1)
            norl_term = torch.sqrt(((2 * 3.141592653) ** self.num_varibales) * det_cov)

            prob = (self.list_weight[0] * torch.exp(exponent)) / torch.sqrt(norl_term)
            probabilities.append(prob)


        if self.num_components >= 2:
            mu2 = self.Linear_mu2(self.list_mu_Tensor[1])
            self.list_mu.append(mu2)
            cov2 = self.Linear_cov2(self.list_cov_Tensor[1].reshape(-1))
            self.list_cov.append(cov2)

        if self.num_components >= 3:
            mu3 = self.Linear_mu3(self.list_mu_Tensor[2])
            self.list_mu.append(mu3)
            cov3 = self.Linear_cov3(self.list_cov_Tensor[2].reshape(-1))
            self.list_cov.append(cov3)

        # # Calculate the probability density for each component
        # for i in range(self.num_components):
        #     # print("self.list_mu[i].shape:", self.list_mu[i].shape, sep="\n")  # torch.Size([64, |X|])
        #     # print("x.shape:", x.shape, sep="\n")  # torch.Size([64, |X|])
        #     # diff = x - self.list_mu[i]
        #     # print("diff.shape:", diff.shape, sep="\n")  # torch.Size([64, |X|])
        #
        #     # print("self.list_cov[i].shape:", self.list_cov[i].shape, sep="\n")  # torch.Size([64, |X|])
        #     if self.num_varibales == 1:
        #         inverse_cov = self.list_cov[i]
        #     else:
        #         inverse_cov = torch.inverse(self.list_cov[i])
        #     # print("inverse_cov.shape:", inverse_cov.shape, sep="\n")  # torch.Size([64, |X|])
        #
        #     # print(torch.matmul(torch.inverse(self.list_cov[i]), diff.t())) # cov-1 * (x-mu)
        #     # print((torch.matmul(torch.inverse(self.cov[i]), diff.t())).t())
        #     # print(torch.mul(  diff, (torch.matmul(torch.inverse(self.cov[i]), diff.t())).t()  ))
        #     # print(torch.sum(torch.mul(diff, (torch.matmul(torch.inverse(self.cov[i]), diff.t())).t()), dim=1)) # (x-mu)T * cov-1 * (x-mu)
        #     exponent = -0.5 * torch.sum(torch.mul(diff, (torch.matmul(inverse_cov, diff.t())).t()), dim=1)
        #     # exponent = -0.5 * torch.sum(torch.matmul(diff, torch.matmul(torch.inverse(self.cov[i]), diff.t())), dim=1) #错误
        #     # print("exponent.shape:", exponent.shape, sep="\n")
        #     # print("exponent.shape:", exponent.shape, sep="\n")  # torch.Size([64])
        #     # exit(0)
        #
        #     # Calculate the probability density for this component
        #     if self.num_varibales == 1:
        #         det_cov = torch.tensor(self.list_cov[i].item())
        #     else:
        #         det_cov = torch.det(self.list_cov[i])
        #     # print("det_cov:", det_cov, sep="\n")  # torch.Size([64, |X|])
        #     norl_term = torch.sqrt( ((2 * 3.14159265358979323846) ** self.num_varibales) * det_cov )
        #     # print("(2 * 3.14159265358979323846) ** self.num_varibales:", (2 * 3.14159265358979323846) ** self.num_varibales, sep="\n")
        #     # print("self.list_cov[i]:", self.list_cov[i], sep="\n")
        #     # print("torch.det(self.list_cov[i]):", torch.det(self.list_cov[i]), sep="\n")
        #     # print("norl_term:", norl_term, sep="\n")
        #
        #     prob = (self.list_weight[i] * torch.exp(exponent)) / torch.sqrt(norl_term)
        #     # print(prob)
        #     # print("prob.shape:", prob.shape, sep="\n")  # torch.Size([64])
        #     # exit(0)
        #     probabilities.append(prob)
        # # print(probabilities)
        # # exit(0)


        # Calculate the probability for each sample by combining the probabilities of all components
        probabilities = torch.stack(probabilities, dim=1)
        # print(probabilities)
        # print("probabilities.shape:", probabilities.shape, sep="\n")  # torch.Size([64, K])
        # exit(0)
        # 计算每个样本的概率值
        # likelihood = torch.sum(probabilities, dim=1)
        likelihood = torch.sum(probabilities, dim=1)
        # print(likelihood)
        # print("likelihood.shape:", likelihood.shape, sep="\n")  # torch.Size([64])
        # exit(0)
        return likelihood.reshape(x.shape[0], -1)