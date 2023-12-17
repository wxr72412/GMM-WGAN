import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import os, sys
sys.path.append(os.getcwd())

import init.init_F as init_F
import model.GAN as GAN
import model.WGAN as WGAN

import model.GMM_WGAN as GMM_WGAN
import model.GMM_Generator_cov2 as GMM_in_WGAN
import metrics.metrics as metrics

import hyperPara

# import sklearn.datasets
import tflib as lib
import tflib.plot
import tflib.debug as debug

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# MODE = hyperPara.MODE
# DATASET = hyperPara.DATASET

use_cuda = hyperPara.use_cuda
ITERS = hyperPara.ITERS
CRITIC_ITERS = hyperPara.CRITIC_ITERS
FIXED_GENERATOR = hyperPara.FIXED_GENERATOR  # whether to hold the generator fixed at real data plus
BATCH_SIZE = hyperPara.BATCH_SIZE  # Batch size
BATCH_SIZE_plot = hyperPara.BATCH_SIZE_plot
num_gaussians = hyperPara.num_gaussians
K = hyperPara.num_components
lr = hyperPara.lr
image = hyperPara.image


def run(MODE, DATASET, data, data_plot, task_type="learning", ground_truth=None, column_indices=None, search_values=None, dict_V_q_not_state=None, dict_V_e_in=None, normal_type=True):
    # print("data:", data, sep="\n")
    # print("data_plot:", data_plot, sep="\n")
    sample_data = data[:, 0:data.shape[-1] - 1]  # [采样样本]
    sample_data_plot = data_plot[:, 0:data_plot.shape[-1] - 1]  # [采样样本]
    # print("sample_data:", sample_data, sep="\n")
    # print("sample_data_plot:", sample_data_plot, sep="\n")
    ############################################################
    ############################################################
    normalize_coefficien = np.max(sample_data, axis=0)
    if normal_type == False:
        normalize_coefficien = 1.0
    # print("normalize_coefficien:", normalize_coefficien, sep="\n")

    # 对每一列进行归一化
    sample_data_norm = sample_data / normalize_coefficien
    sample_data_plot_norm = sample_data_plot / normalize_coefficien
    # print("sample_data_norm:", sample_data_norm, sep="\n")
    # print("sample_data_norm.shape:", sample_data_norm.shape, sep="\n")
    # print("sample_data_plot_norm:", sample_data_plot_norm, sep="\n")
    # print("sample_data_plot_norm.shape:", sample_data_plot_norm.shape, sep="\n")
    # exit(0)
    ############################################################
    ############################################################

    p = data[:, data.shape[-1] - 1].reshape(-1, 1)
    p_plot = data_plot[:, data_plot.shape[-1] - 1].reshape(-1, 1)
    # print("p:", p, sep="\n")
    # print("p.shape:", p.shape, sep="\n")
    data_norm = np.array(np.concatenate((sample_data_norm, p), axis=1))  # [采样样本, 生成分布对应的概率值]
    data_plot_norm = np.array(np.concatenate((sample_data_plot_norm, p_plot), axis=1))  # [采样样本, 生成分布对应的概率值]
    # print("data_norm:", data_norm, sep="\n")
    # print("data_plot_norm:", data_plot_norm, sep="\n")
    # exit(0)

    netG = None
    netD = None
    if MODE == 'GAN':
        netG = GAN.Generator()
        netD = GAN.Discriminator()
    elif MODE == 'WGAN':
        netG = WGAN.Generator()
        netD = WGAN.Discriminator()
    elif MODE == 'GMM_WGAN':
        netG = GMM_WGAN.Generator(sample_data, K)
        netD = GMM_WGAN.Discriminator()
    elif MODE == 'GMM_in_GAN':
        netG = GMM_in_WGAN.Generator(sample_data, K)
        netD = GAN.Discriminator()
    elif MODE == 'GMM_in_WGAN':
        netG = GMM_in_WGAN.Generator(sample_data, K)
        netD = WGAN.Discriminator()
    # elif MODE == 'GMM_WGAN':
    #     netG = GMM_WGAN.Generator()
    #     netD = GMM_WGAN.Discriminator()

    netG.apply(init_F.weights_init)
    netD.apply(init_F.weights_init)
    # for param_tensor in netG.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
    #     print(param_tensor, '\t', netG.state_dict()[param_tensor].size(), netG.state_dict()[param_tensor])
    # exit(0)
    # print(netG)
    # print(netD)
    # exit(0)

    # BCE_loss = torch.nn.BCELoss()
    if use_cuda:
        netG = netG.cuda()
        netD = netD.cuda()
    #     BCE_loss = BCE_loss.cuda()

    # print(torch.cuda.is_available()) # True
    # print("default device: {}".format(torch.Tensor([4,5,6]).device)) # default device: cpu
    # print(next(netG.parameters()).device) # cuda:0
    # exit(0)

    optimizerG = optim.Adam(netG.parameters(), lr, betas=(0.5, 0.9))
    optimizerD = optim.Adam(netD.parameters(), lr, betas=(0.5, 0.9))
    # optimizerG = optim.SGD(netG.parameters(), lr) # 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行
    # optimizerD = optim.SGD(netD.parameters(), lr) # 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行
    # optimizerG = optim.RMSprop(netG.parameters(), lr) # 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行
    # optimizerD = optim.RMSprop(netD.parameters(), lr) # 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行
    # print(optimizerG) # Adam (Parameter Group 0, amsgrad: False, betas: (0.5, 0.9), eps: 1e-08, lr: 0.0001, weight_decay: 0)
    # exit(0)

    # one = torch.FloatTensor([1]) # tensor([1.])
    # mone = one * -1 # tensor([-1.]) torch.Size([1])
    # if use_cuda:
    #     one = one.cuda()
    #     mone = mone.cuda()

    # [采样样本+概率值]
    real_data = torch.Tensor(data_norm)
    # print("real_data:", real_data, sep="\n")
    # print("real_data.shape:", real_data.shape, sep="\n")
    # exit(0)

    if use_cuda:
        real_data = real_data.cuda()
    real_data_v = autograd.Variable(real_data)

    # [采样样本]
    # noise = torch.Tensor(np.random.normal(0, 4, BATCH_SIZE)).reshape((BATCH_SIZE, 1)) # 从高斯分布随机采样样本，并将维度由(BATCH_SIZE, )变为(BATCH_SIZE, 1)，以吻合D的输入
    # noise = real_data[:, 0].reshape(BATCH_SIZE, 1)
    noise = real_data[:, 0:data.shape[-1] - 1]
    # print(noise)
    # print(noise.shape) # torch.Size([BATCH_SIZE, 1])
    # exit(0)
    if use_cuda:
        noise = noise.cuda()
    noise_v = autograd.Variable(noise)  # totally freeze netG
    # print("noise_v:", noise_v, sep="\n")
    # print("noise_v.shape:", noise_v.shape, sep="\n")
    # exit(0)

    # [预测的采样样本]
    real_data_plot = torch.Tensor(data_plot_norm)
    noise_plot = real_data_plot[:, 0:data.shape[-1] - 1]  # [采样样本]
    if use_cuda:
        noise_plot = noise_plot.cuda()
    noise_v_plot = autograd.Variable(noise_plot)  # totally freeze netG
    # print(noise_v_plot)
    # print(noise_v_plot.shape) # torch.Size([BATCH_SIZE, 1])
    # exit(0)

    min_MAE = 1000
    best_prob_predict = None
    best_ITERS = None
    list_WD = []

    print('iteration...')
    for iteration in range(ITERS+1):
        # print(iteration)
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        # exit(0)

        for iter_d in range(CRITIC_ITERS):
            netD.zero_grad()

            # train with real [采样样本+概率值]
            # print("real_data_v:", real_data_v, sep="\n")
            # print("real_data_v.shape:", real_data_v.shape, sep="\n")
            D_real = netD(real_data_v)
            # print(D_real) # tensor([ 0.0001, -0.0008,  0.0007, -0.0008, -0.0008, -0.0008, -0.0008, -0.0006, 0.0008, -0.0008], device='cuda:0', grad_fn=<ViewBackward>)
            # exit(0)

            # train with fake [采样样本+预测的概率值]
            p_fake = autograd.Variable(netG(noise_v).data) # 采样样本对应生成分布的概率值
            # print(p_fake)
            # print(p_fake.shape)
            # exit(0)
            fake = torch.cat((noise_v, p_fake), dim=1) # [采样样本, 生成分布对应的概率值]
                # print(fake)  # torch.Size([10, 2])
                # print(fake.shape)  # torch.Size([10, 2])
            D_fake = netD(fake)
            # exit(0)

            D_cost = 0
            Wasserstein_D = 0
            if MODE == 'GAN' or MODE == 'GMM_GAN' or MODE == 'GMM_in_GAN':
                D_cost = - (torch.log(D_real).mean() + torch.log(1-D_fake).mean())
            elif MODE == 'WGAN' or MODE == 'GMM_WGAN' or MODE == 'GMM_in_WGAN':
                # print(real_data_v)
                # print(fake)
                # exit(0)
                gradient_penalty = WGAN.calc_gradient_penalty(netD, real_data_v.data, fake.data) # train with gradient penalty
                D_fake = D_fake.mean()
                D_real = D_real.mean()
                # D_cost = D_fake - D_real
                D_cost = D_fake - D_real + gradient_penalty
                # D_cost = - (torch.log(D_real).mean() + torch.log(1 - D_fake).mean())
                Wasserstein_D = (D_real - D_fake).item()

            ############################################################
            # D_real = D_real.reshape(-1)
            # D_fake = D_fake.reshape(-1)
            # D_real.backward(mone)
            # D_fake.backward(one)
            # gradient_penalty.backward()
            ############################################################
            D_cost.backward()
            ############################################################
            # for p in netD.parameters():  # reset requires_grad
            #     print(p.grad)
            # exit(0)
            ############################################################
            optimizerD.step()
            if iteration % 100 == 0 and iter_d % CRITIC_ITERS == 0:
                print("Totel_Iters: " + str(iteration) + ", D_Iters: " + str(iter_d) + ", D_cost: " + str(D_cost.item()) + ", Wasserstein_D: " + str(Wasserstein_D))

        if not FIXED_GENERATOR:
            ############################
            # (2) Update G network
            ###########################
            for para in netD.parameters():
                para.requires_grad = False  # to avoid computation
            netG.zero_grad()

            # train with fake [采样样本+预测的概率值]
            p_fake = netG(noise_v)
            fake = torch.cat((noise_v, p_fake), dim=1) # [采样样本, 生成分布对应的概率值]
            G_fake = netD(fake)

            G_cost = 0
            if MODE == 'GAN' or MODE == 'GMM_GAN' or MODE == 'GMM_in_GAN':
                # D_cost = torch.log(D_real).mean() + torch.log(1 - D_fake).mean()
                G_cost = torch.log(1 - G_fake).mean()
            elif MODE == 'WGAN' or MODE == 'GMM_WGAN' or MODE == 'GMM_in_WGAN':
                # G_cost = torch.log(1 - G_fake).mean()
                G_cost = - G_fake.mean()
            #####################################################
            # G_fake = G_fake.reshape(-1)
            # G_fake.backward(mone)
            #####################################################
            G_cost.backward()
            #####################################################
            # for p in netG.parameters():  # reset requires_grad
            #     print(p.grad)
            # exit(0)
            #####################################################
            optimizerG.step()

            # if iteration % 100 == 0:
            #     print("Totel_Iters: " + str(iteration) + ", G_cost: "+ str(G_cost.item()))

        # Calculate dev loss and generate samples every 100 iters
        if DATASET == 'gaussian':
            lib.plot.plot('tmp/' + DATASET + '/' + str(num_gaussians) + '/' + MODE + '/' + 'D cost', D_cost.cpu().data.numpy())
            lib.plot.plot('tmp/' + DATASET + '/' + str(num_gaussians) + '/' + MODE + '/' + 'G cost', G_cost.cpu().data.numpy())
            if MODE == 'WGAN' or MODE == 'GMM_WGAN' or MODE == 'GMM_in_WGAN':
                lib.plot.plot('tmp/' + DATASET + '/' + str(num_gaussians) + '/' + MODE + '/' + 'wasserstein distance', Wasserstein_D)
        else:
            lib.plot.plot('tmp/' + DATASET + '/' + MODE + '/' + 'D cost', D_cost.cpu().data.numpy())
            lib.plot.plot('tmp/' + DATASET + '/' + MODE + '/' + 'G cost', G_cost.cpu().data.numpy())
            if MODE == 'WGAN' or MODE == 'GMM_WGAN' or MODE == 'GMM_in_WGAN':
                lib.plot.plot('tmp/' + DATASET + '/'+ MODE + '/' + 'wasserstein distance', Wasserstein_D)


        p_fake_plot = netG(noise_v_plot)

        # fake_plot = torch.cat((noise_v_plot * normalize_coefficien, p_fake_plot), dim=1)  # [采样样本, 生成分布对应的概率值]
        # fake_plot = fake_plot.cpu().detach().numpy()
        fake_plot = np.array(np.concatenate((sample_data_plot, p_fake_plot.cpu().detach().numpy()), axis=1))  # [采样样本, 生成分布对应的概率值]

        # print("data:", data, sep="\n")
        # print("data.shape:", data.shape, sep="\n")  # (BATCH_SIZE, 2)
        # print("fake_plot:", fake_plot, sep="\n")  # (BATCH_SIZE, 2)
        # print("fake_plot.shape:", fake_plot.shape, sep="\n")  # (BATCH_SIZE, 2)
        # exit(0)

        # if iteration % 100 == 0 and (MODE == 'GMM_GAN' or MODE == 'GMM_WGAN' or MODE == 'GMM_in_GAN' or MODE == 'GMM_in_WGAN'):
        #     print("mu:", netG.mu, "cov:", netG.cov, "lamda:", netG.lamda, sep="\n")
            # print("noise_v_plot:", noise_v_plot, sep="\n")  # (BATCH_SIZE, 2)
            # print("fake_plot:", fake_plot, sep="\n")

        if iteration % 100 == 0 and image == True and data.shape[1] == 2:
            lib.plot.flush()
            init_F.generate_image(data_plot, fake_plot, DATASET, MODE)
            lib.plot.tick()
            # exit(0)
            # print("Image Done!")

        if iteration % 100 == 0 and task_type == "learning":
            ave_MAE, WD = metrics.MAS_MSE_KL_JS_WD(data_plot, fake_plot, is_show=True)
            list_WD.append(WD)

        if iteration % 100 == 0 and task_type == 'inference':
            prob_predict = init_F.generate_p(ground_truth, fake_plot, column_indices, search_values, dict_V_q_not_state, dict_V_e_in)
            MAE = metrics.MAS_MSE_KL_JS_WD(np.array(ground_truth).reshape(-1, 1), np.array(prob_predict).reshape(-1, 1), is_show=False)
            if iteration >= 1000 and MAE < min_MAE:
                min_MAE = MAE
                best_prob_predict = prob_predict
                best_ITERS = iteration


    if task_type=="learning":
        print("list_WD: ", list_WD)
        return fake_plot
    elif task_type == 'inference':
        print("min_MAE: ", min_MAE)  # (BATCH_SIZE, 2)
        print("best_ITERS :", best_ITERS)  # (BATCH_SIZE, 2)
        return best_prob_predict
    else:
        print("No task_type!")
        exit(0)