import hyperPara

import random
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import sklearn.datasets

import torch
import torch.autograd as autograd



FIXED_GENERATOR = hyperPara.FIXED_GENERATOR  # whether to hold the generator fixed at real data plus
BATCH_SIZE = hyperPara.BATCH_SIZE  # Batch size
use_cuda = hyperPara.use_cuda
DATASET = hyperPara.DATASET
num_gaussians = hyperPara.num_gaussians


# Dataset iterator
def train_data_gen(DATASET, num_samples):
    if DATASET == '25gaussians':
        dataset = []
        for i in range(100000 / 25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828  # stdev
        while True:
            for i in range(len(dataset) / BATCH_SIZE):
                return dataset[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

    elif DATASET == 'swissroll':

        while True:
            data = sklearn.datasets.make_swiss_roll(
                n_samples=BATCH_SIZE,
                noise=0.25
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5  # stdev plus a little
            return data

    elif DATASET == '8gaussians':
        scale = 2.
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        # [(2.0, 0.0), (-2.0, 0.0),
        # (0.0, 2.0), (0.0, -2.0),
        # (1.414213562373095, 1.414213562373095), (1.414213562373095, -1.414213562373095),
        # (-1.414213562373095, 1.414213562373095), (-1.414213562373095, -1.414213562373095)]
        while True:
            dataset = []
            for i in range(BATCH_SIZE):
                point = np.random.randn(2) * .02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414  # stdev--Standard Deviation
            return dataset

    elif DATASET == 'gaussian':

        if num_gaussians == 1:
            # 均值和标准差
            mean = 5  # 高斯分布的均值
            std_dev = 1  # 高斯分布的标准差
            # 生成随机数据
            if num_samples == BATCH_SIZE:
                samples = np.linspace(0, 7, num_samples)
            else:
                samples = np.linspace(0, 10, num_samples)
            # samples = np.sort((np.random.normal(mean, 4, num_samples))) # 从高斯分布随机采样样本，排序是为了画折线图
            # x = np.linspace(-4 * std_dev, 4 * std_dev, 1000)
            pdf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((samples - mean) / std_dev) ** 2) # 采样样本对应分布的概率值
            # print(pdf)
            # exit(0)
        elif num_gaussians == 2:
            means = [-3, 4]  # 每个组件的均值
            std_devs = [1, 0.5]  # 每个组件的标准差
            weights = [0.7, 0.3]  # 每个组件的混合权重
            # 生成随机数据
            if num_samples == BATCH_SIZE:
                samples = np.linspace(-5, 5, num_samples)
            else:
                samples = np.linspace(-10, 10, num_samples)
            # samples = np.sort((np.random.normal(sum(means), 5, num_samples))) # 从高斯分布随机采样样本，排序是为了画折线图
            pdf = np.zeros_like(samples)
            for i in range(num_gaussians):
                pdf += weights[i] * (1 / (std_devs[i] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((samples - means[i]) / std_devs[i]) ** 2) # 采样样本对应分布的概率值
        elif num_gaussians == 3:
            means = [0, 4, -3]  # 每个组件的均值
            std_devs = [1, 0.5, 1.5]  # 每个组件的标准差
            weights = [0.4, 0.3, 0.3]  # 每个组件的混合权重
            # 生成随机数据
            if num_samples == BATCH_SIZE:
                samples = np.linspace(-10, 5, num_samples)
            else:
                samples = np.linspace(-10, 10, num_samples)
            # samples = np.sort((np.random.normal(sum(means), 5, num_samples))) # 从高斯分布随机采样样本，排序是为了画折线图
            pdf = np.zeros_like(samples)
            for i in range(num_gaussians):
                pdf += weights[i] * (1 / (std_devs[i] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((samples - means[i]) / std_devs[i]) ** 2) # 采样样本对应分布的概率值
            # print(samples)
            # print(pdf)
            # exit(0)

        while True:
            dataset = []
            for i in range(num_samples):
                point = [0, 0]
                # point = [0, 0, 0]
                point[0] = samples[i]
                point[1] = pdf[i]
                # point[2] = pdf[i]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            return dataset

    else:
        print("train_data_gen error!")
        exit(0)


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def generate_p(ground_truth, fake, column_indices, search_values, dict_V_q_not_state, dict_V_e_in):
    # print("fake:", fake, sep="\n")
    # print("column_indices:", column_indices, sep="\n")
    # print("search_values:", search_values, sep="\n")
    # print("dict_V_q_not_state:", dict_V_q_not_state, sep="\n")
    # print("dict_V_e_in:", dict_V_e_in, sep="\n")
    # exit(0)

    weights = fake[:, fake.shape[-1] - 1]  # [输入概率+预测概率]
    # print("weights:", weights, sep="\n")
    prob_sum = weights.sum()
    # print("prob_sum:", prob_sum, sep="\n")
    prob_norm = weights/prob_sum
    # print("prob_norm:", prob_norm, sep="\n")

    fake[:, fake.shape[-1] - 1] = prob_norm
    # print("fake:", fake, sep="\n")
    # exit(0)

    # 使用布尔索引查找符合条件的行： 缺值对应的预测概率值
    fake_matching_rows = fake[np.all(fake[:, column_indices] == search_values, axis=1)][0]
    # print("fake_matching_rows:", fake_matching_rows, sep="\n") # [4.00000000e+01 4.00000000e+00 1.25000000e+02 1.50066804e-02]

    # 原始BN推理出的真实概率值
    prob_predict = fake_matching_rows[-1]

    return prob_predict


# frame_index = [0]
# def generate_image(true_dist, netG, netD, DATASET):
#     """
#     Generates and saves a plot of the true distribution, the generator, and the
#     critic.
#     """
#     N_POINTS = 10
#     RANGE = 3
#
#     points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
#
#
#
#     points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
#     points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
#     points = points.reshape((-1, 2))
#     # print(points)
#     # print(points.shape)
#     # exit(0)
#
#     points_v = autograd.Variable(torch.Tensor(points))
#     if use_cuda:
#         points_v = points_v.cuda()
#     disc_map = netD(points_v).cpu().data.numpy()
#     # print(disc_map)
#     # print(disc_map.shape)
#     # exit(0)
#
#     noise = torch.randn(BATCH_SIZE, 2)
#     if use_cuda:
#         noise = noise.cuda()
#     noisev = autograd.Variable(noise)
#     # true_dist_v = autograd.Variable(torch.Tensor(true_dist).cuda() if use_cuda else torch.Tensor(true_dist))
#     samples = netG(noisev).cpu().data.numpy()
#     # print(samples)
#     # print(samples.shape)
#     # exit(0)
#
#     plt.clf()
#
#     # x = y = np.linspace(-RANGE, RANGE, N_POINTS)
#     # plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
#
#     plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
#     if not FIXED_GENERATOR:
#         plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')
#
#     plt.savefig('tmp/' + DATASET + '/' + 'frame' + str(frame_index[0]) + '.jpg')
#
#     frame_index[0] += 1


config = {
    "font.family": 'Times New Roman',
    "font.size":11,
    "mathtext.fontset": 'stix',
    # "font.serif": ['SimSun'],
}
rcParams.update(config)
rcParams['xtick.direction'] = 'in'  # 将x轴的刻度线方向设置向内
rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 12,
        'color': 'k',}

frame_index = [0]
def generate_image(data, fake, DATASET, MODE):
    # 绘制直方图
    fig, axs = plt.subplots(figsize=(5, 3.5))
    # plt.hist(samples, bins=100, density=True, alpha=0.5, color='b', label='样本直方图')
    # plt.hist(samples, range=(-2,2), bins=1000, density=True, alpha=0.5, color='b', label='样本直方图')

    # plt.plot(data[:, 0], data[:, 1], 'r', lw=1, label='概率密度函数')
    axs.plot(data[:, 0], data[:, 1], 'r', lw=1, label='Probability density function')

    axs.tick_params(top='on', right='on', which='both', labelsize=10, labelcolor='k')

    # 绘制采样数据的散点图
    # plt.scatter(fake[:, 0], fake[:, 1], marker='+', alpha=0.5, label='生成数据', color='b')
    axs.scatter(fake[:, 0], fake[:, 1], marker='x', alpha=0.5, s=25, label='Generated samples', color='b')

    # plt.title('生成数据和高斯分布概率密度函数')

    # plt.xlabel('随机变量值')
    axs.set_xlabel('$X$', font, labelpad=1)

    # plt.ylabel('概率密度')
    axs.set_ylabel('Probability', font, labelpad=5)

    axs.minorticks_off() # 移除小刻度

    axs.xaxis.tick_bottom()
    axs.yaxis.tick_left()

    # 移除网格线
    axs.grid(axis='y', color = 'w', linestyle = '--', linewidth = 0.5) # 网格线设置为白色，等于删除

    axs.legend(fontsize=12, ncol=1, labelspacing=0.5, handlelength=1, columnspacing=0.01, handletextpad=0.5, loc="upper left")
    plt.grid(True)
    if DATASET == 'gaussian':
        axs.set_xlim(-11, 11)
        axs.set_xticks(ticks=(-10, -5, 0, 5, 10))
        # axs.set_xticklabels(['$10^1$', '$10^3$', '$10^5$'])
        axs.set_ylim(-0.01, 0.28)
        # axs.set_ylim(-0.05, 0.25)
        # axs.set_yticks(ticks=(0, 0.1, 0.2))
    else:
        plt.xlim(-100, 100)

    fig.tight_layout(pad=0.2)

    # plt.show()
    if DATASET == 'gaussian':
        plt.savefig('tmp/' + DATASET + '/' + str(num_gaussians) + '/' + MODE + '/' + 'frame' + str(frame_index[0]) + '.pdf')
    else:
        plt.savefig('tmp/' + DATASET + '/' + MODE + '/' + 'frame' + str(frame_index[0]) + '.jpg')
    frame_index[0] += 1