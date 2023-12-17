# -*- coding: utf-8 -*-
# ----------------------------------------------------
# Copyright (c) 2017, Wray Zheng. All Rights Reserved.
# Distributed under the BSD License.
# ----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# DEBUG = True
DEBUG = False

######################################################
# 调试输出函数
# 由全局变量 DEBUG 控制输出
######################################################
def debug(*args, **kwargs):
    global DEBUG
    if DEBUG:
        print(*args, **kwargs)


######################################################
# 第 k 个模型的高斯分布密度函数
# 每 i 行表示第 i 个样本在各模型中的出现概率
# 返回一维列表
######################################################
def phi(Y, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(Y)


######################################################
# E 步：计算每个高斯分布对样本的响应度
# Y 为样本矩阵，每个样本一行，只有一个特征时为列向量
# mu 为均值多维数组，每行表示一个样本各个特征的均值
# cov 为协方差矩阵的数组，alpha 为模型响应度数组
######################################################
def getExpectation(Y, mu, cov, alpha, E_step=1):
    # 样本数
    N = Y.shape[0]
    # 模型数
    K = alpha.shape[0]

    # 为避免使用单个高斯分布或样本，导致返回结果的类型不一致
    # 因此要求样本数和高斯分布个数必须大于1
    # assert N > 1, "There must be more than one sample!"
    # assert K > 1, "There must be more than one gaussian model!"

    # 响应度矩阵，行对应样本，列对应响应度
    gamma = np.mat(np.zeros((N, K)))
    # print("Y:", Y, sep="\n")
    # print("gamma:", gamma, sep="\n")

    # 计算各高斯分布中所有样本出现的概率，行对应样本，列对应高斯分布
    prob = np.zeros((N, K))
    # print("prob:", prob, sep="\n")
    # exit(0)

    for k in range(K):
        # print("mu[k]:", mu[k], sep="\n")
        # print("cov[k]:", cov[k], sep="\n")
        prob[:, k] = phi(Y, mu[k], cov[k])
    prob = np.mat(prob)
    # print("prob:", prob, sep="\n")


    # 计算每个高斯分布对每个样本的概率值
    for k in range(K):
        gamma[:, k] = alpha[k] * prob[:, k]
    # print(gamma)

    # 计算每个样本的概率值
    if E_step == 0:
        return np.sum(gamma, axis=1)

    # print(gamma)
    # 计算每个高斯分布对样本的响应度
    for i in range(N):
        gamma[i, :] /= (np.sum(gamma[i, :]))
    # print(gamma)
    # exit(0)
    return gamma


######################################################
# M 步：迭代模型参数
# Y 为样本矩阵，gamma 为响应度矩阵
######################################################
def maximize(Y, gamma):
    # 样本数和特征数
    N, D = Y.shape
    # 模型数
    K = gamma.shape[1]

    # 初始化参数值
    mu = np.zeros((K, D))
    cov = []
    alpha = np.zeros(K)

    # 更新每个模型的参数
    for k in range(K):
        # 第 k 个模型对所有样本的响应度之和
        Nk = np.sum(gamma[:, k])
        # 更新 mu
        # 对每个特征求均值
        mu[k, :] = np.sum(np.multiply(Y, gamma[:, k]), axis=0) / Nk
        # print("Y:", Y, sep="\n")
        # print("gamma:", gamma, sep="\n")
        # print("gamma[:, k]:", gamma[:, k], sep="\n")
        # print("np.multiply(Y, gamma[:, k]), axis=0:", np.multiply(Y, gamma[:, k]), sep="\n")
        # print("np.sum(np.multiply(Y, gamma[:, k]), axis=0):", np.sum(np.multiply(Y, gamma[:, k]), axis=0), sep="\n")
        # print(np.sum(np.multiply(Y, gamma[:, k]), axis=0) / Nk, sep="\n")
        # print(mu[k, :], sep="\n")
        # print(mu, sep="\n")
        # exit(0)
        # 更新 cov
        cov_k = (Y - mu[k]).T * np.multiply((Y - mu[k]), gamma[:, k]) / Nk
        # print("Y:", Y, sep="\n")
        # print("mu[k]:", mu[k], sep="\n")
        # print("Y - mu[k]", Y - mu[k], sep="\n")
        # print("gamma[:, k]", gamma[:, k], sep="\n")
        # print("np.multiply((Y - mu[k]), gamma[:, k])", np.multiply((Y - mu[k]), gamma[:, k]), sep="\n")
        # print("(Y - mu[k]).T * np.multiply((Y - mu[k]), gamma[:, k])", (Y - mu[k]).T * np.multiply((Y - mu[k]), gamma[:, k]), sep="\n")
        # exit(0)
        cov.append(cov_k)
        # 更新 alpha
        alpha[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, alpha


######################################################
# 数据预处理
# 将所有数据都缩放到 0 和 1 之间
######################################################
def scale_data(Y):
    # 对每一维特征分别进行缩放
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    debug("Data scaled.")
    return Y


######################################################
# 初始化模型参数
# shape 是表示样本规模的二元组，(样本数, 特征数)
# N（论文为m）表示样本数
# D（论文为N） 表示特征数（变量数）
# K（论文为K） 表示模型个数
######################################################
def init_params(shape, K):
    N, D = shape # (272, 2)
    mu = np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    debug("Parameters initialized.")
    debug("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    debug("mu.shape:", mu.shape, "cov.shape:", cov.shape, "alpha.shape:", alpha.shape, sep="\n")
    debug()
    return mu, cov, alpha


######################################################
# 高斯混合模型 EM 算法
# 给定样本矩阵 Y，计算模型参数
# K 为模型个数
# times 为迭代次数
######################################################
def GMM_EM(Y, K, times):
    Y = scale_data(Y)
    # print(Y)
    # print(Y.shape) # (272, 2)
    # exit(0)
    mu, cov, alpha = init_params(Y.shape, K)
    # print("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    # print("mu.shape:", mu.shape, "cov.shape:", cov.shape, "alpha.shape:", alpha.shape, sep="\n")
    # exit(0)

    for i in range(times):
        gamma = getExpectation(Y, mu, cov, alpha)
        mu, cov, alpha = maximize(Y, gamma)
        print("time:", i +1, "mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    debug("{sep} Result {sep}".format(sep="-" * 20))
    debug("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    return mu, cov, alpha
