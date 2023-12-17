import hyperPara

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import GMM.gmm as GMM
import math

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
        self.mu = nn.Parameter(torch.randn(self.num_components, self.num_varibales))
        self.cov = nn.Parameter(torch.ones(self.num_components, self.num_varibales))
        self.weight = nn.Parameter(torch.ones(self.num_components))

        # self.mu = nn.Parameter(torch.FloatTensor([0, 4, -3]).reshape(-1, 1))  # 每个组件的均值
        # self.cov = nn.Parameter(torch.FloatTensor([1, 0.5, 1.5]).reshape(-1, 1))  # 每个组件的标准差
        # self.weight = nn.Parameter(torch.FloatTensor([0.4, 0.3, 0.3]).reshape(-1))  # 每个组件的混合权重
        # print(Y)  # torch.Size([K, N])
        # print(Y.shape)  # torch.Size([K, N])

        # print(self.mu) # torch.Size([K, N])
        # print(self.mu.shape)
        # print(self.cov)
        # print(self.cov.shape) # torch.Size([K, N, N])
        # print(self.weight)
        # print(self.weight.shape) # torch.Size([K])
        # exit(0)



        self.sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU(True)


    def forward(self, x):
        # weight
        # self.ReLU_cov = self.ReLU(self.cov)
        # print("self.ReLU_cov:", self.ReLU_cov, sep="\n")
        # print("self.ReLU_cov.shape:", self.ReLU_cov.shape, sep="\n")
        # exit(0)

        sigmoid_weight = self.sigmoid(self.weight)
        self.lamda = sigmoid_weight / torch.sum(sigmoid_weight)
        # print("weight:", self.weight, sep="\n")
        # print("self.lamda:", self.lamda, sep="\n")
        # exit(0)

        list_probs = []

        # Calculate the probability density for each component
        for i in range(self.num_components):

            # x = x.reshape(-1, 1).expand(x.shape[0], 2)

            diff = x - self.mu[i]
            # print("x:", x, sep="\n")  # torch.Size([64, |X|])
            # print("self.mu[i]:", self.mu[i], sep="\n")  # torch.Size([|X|])
            # print("diff:", diff, sep="\n")  # torch.Size([64, |X|])
            # print("diff.shape:", diff.shape, sep="\n")  # torch.Size([64, |X|])
            # exit(0)

            square_diff = torch.square(diff)
            # print("square_diff:", square_diff, sep="\n")  # torch.Size([64, |X|])
            # print("square_diff.shape:", square_diff.shape, sep="\n")  # torch.Size([64, |X|])
            # exit(0)

            inverse_cov = 1.0 / self.cov[i]
            # print("self.cov[i]:", self.cov[i], sep="\n")  # torch.Size([64, |X|])
            # print("inverse_cov:", inverse_cov, sep="\n")  # torch.Size([64, |X|])
            # print("inverse_cov.shape:", inverse_cov.shape, sep="\n")  # torch.Size([64, |X|])
            # exit(0)

            up_term = torch.matmul(square_diff, inverse_cov)
            # print("up_term:", up_term, sep="\n")  # torch.Size([64, |X|])
            # print("up_term.shape:", up_term.shape, sep="\n")  # torch.Size([64, |X|])
            # exit(0)

            exp = torch.exp(up_term * -0.5)
            # print("exp:", exp, sep="\n")  # torch.Size([64, |X|])
            # print("exp.shape:", exp.shape, sep="\n")  # torch.Size([64, |X|])
            # exit(0)

            det_cov = torch.prod(self.cov[i])
            # print("det_cov:", det_cov, sep="\n")  # torch.Size([64, |X|])
            # print("det_cov.shape:", det_cov.shape, sep="\n")  # torch.Size([64, |X|])
            # exit(0)

            norl_term = 1 / torch.sqrt(det_cov * ((2 * math.pi) ** self.num_varibales))
            # print("norl_term:", norl_term, sep="\n")  # torch.Size([64, |X|])
            # print("norl_term.shape:", norl_term.shape, sep="\n")  # torch.Size([64, |X|])
            # exit(0)

            prob = norl_term * exp
            # print("prob:", prob, sep="\n")  # torch.Size([64, |X|])
            # print("prob.shape:", prob.shape, sep="\n")  # torch.Size([64, |X|])
            # exit(0)

            list_probs.append(prob)



        # Calculate the probability for each sample by combining the probabilities of all components
        probs = torch.stack(list_probs, dim=1)
        # print("probabilities:", probs, sep="\n")  # torch.Size([64, K])
        # print("probabilities.shape:", probs.shape, sep="\n")  # torch.Size([64, K])
        # exit(0)

        # 计算每个样本的概率值
        # likelihood = torch.sum(probabilities, dim=1)
        # likelihood = torch.sum(probs, dim=1)
        likelihood = torch.matmul(probs, self.lamda)
        # print(likelihood)
        # print("likelihood.shape:", likelihood.shape, sep="\n")  # torch.Size([64])
        # exit(0)

        return likelihood.reshape(x.shape[0], -1)