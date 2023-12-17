import matplotlib.pyplot as plt
import numpy as np
import random
import metrics.metrics as metrics

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import os, sys
sys.path.append(os.getcwd())

import init.init_F as init_F
import hyperPara
import GMM.gmm as GMM
import tflib as lib
import tflib.plot


# DATASET = hyperPara.DATASET
num_components = hyperPara.num_components
GMM_ITERS = hyperPara.GMM_ITERS
BATCH_SIZE = hyperPara.BATCH_SIZE
image = hyperPara.image

def run(MODE, DATASET, data, data_plot, task_type="learning", ground_truth=None, column_indices=None, search_values=None, dict_V_q_not_state=None, dict_V_e_in=None):
    # print("data:", data, sep="\n")
    # print("data.shape:", data.shape, sep="\n")

    weights = data[:, data.shape[-1]-1] # [实际分布对应的概率值]
    # print("weights:", weights, sep="\n")
    # print("weights.shape:", weights.shape, sep="\n")

    sample_data = data[:, 0:data.shape[-1]-1] # [采样样本]
    sample_data_plot = data_plot[:, 0:data.shape[-1] - 1]  # [采样样本]
    # print("sample_data:", sample_data, sep="\n")
    # print("sample_data.shape:", sample_data.shape, sep="\n") # (BATCH_SIZE, 1)
    # exit(0)

    # 计算每列的最小值
    # min_values = sample_data.min(axis=0)
    # print("min_values:", min_values, sep="\n")
    # 归一化：除以每列的最大值
    # sample_data_norm = sample_data - min_values
    # sample_data_plot_norm = sample_data_plot - min_values

    normalize_coefficien = 1.0
    if DATASET == 'heart':
        normalize_coefficien = 20.0
    elif DATASET == 'bone':
        normalize_coefficien = 10.0
    elif DATASET == 'hepar2':
        normalize_coefficien = 10.0
    elif DATASET == 'munin1':
        normalize_coefficien = 50.0

    sample_data_norm = sample_data / normalize_coefficien
    sample_data_plot_norm = sample_data_plot / normalize_coefficien
    # print("sample_data_norm:", sample_data_norm, sep="\n")
    # print("sample_data_plot_norm:", sample_data_plot_norm, sep="\n")

    Y = random.choices(sample_data_norm, weights=weights, cum_weights=None, k=BATCH_SIZE) # 根据CPT计算的离散概率值，采样出对应的样本
    Y = np.matrix(Y, copy=True)
    # print("Y:", Y, sep="\n")
    # print("Y.shape:", Y.shape, sep="\n") # (256, 1)
    # exit(0)




    K = num_components
    # scale_Y = GMM.scale_data(Y)
    # print("Y:", Y, sep="\n")
    # print("scale_Y .shape:", scale_Y .shape, sep="\n")
    # exit(0)
    mu, cov, alpha = GMM.init_params(Y.shape, K)
    # print("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    # print("mu.shape:", mu.shape, "cov.shape:", cov.shape, "alpha.shape:", alpha.shape, sep="\n") # (K, D) (K, D, D) (K,)
    # exit(0)

    min_MAE = 1000
    best_prob_predict = None
    best_ITERS = None
    # 计算 GMM 模型参数
    for i in range(GMM_ITERS+1):
        gamma = GMM.getExpectation(Y, mu, cov, alpha, E_step=1)
        mu, cov, alpha = GMM.maximize(Y, gamma)
        # print("time:", i + 1, "mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
        # exit(0)

        # sample_data_plot_norm = np.matrix(sample_data_plot_norm, copy=True)  # 归一化后的[采样样本]
        # print("sample_data_plot_norm:", sample_data_plot_norm, sep="\n")
        p = GMM.getExpectation(sample_data_plot_norm, mu, cov, alpha, E_step=0)  # 行：第i个样本； 列：第k个高斯分布； 计算每个分布对每个样本的概率值
        # print("p:", p, sep="\n")
        # print("p.shape:", p.shape, sep="\n")
        # exit(0)
        # fake_plot_not_norm = sample_data_plot_norm + min_values
        fake_plot_not_norm = sample_data_plot_norm * normalize_coefficien
        fake_plot_not_norm = np.array(np.concatenate((fake_plot_not_norm, p), axis=1))  # [采样样本, 生成分布对应的概率值]
        # print("fake:", fake, sep="\n")
        # print("fake.shape:", fake.shape, sep="\n")  # (BATCH_SIZE, 2)
        # exit(0)
        # print(type(fake)) # <class 'numpy.ndarray'>
        # print("data:", data, sep="\n")  # (BATCH_SIZE, 2)
        # print("data.shape:", data.shape, sep="\n")  # (BATCH_SIZE, 2)
        # print(type(data)) # <class 'numpy.ndarray'>
        # exit(0)
        # fake = fake.cpu().detach().numpy()


        # Calculate dev loss and generate samples every 100 iters
        if i % 100 == 0 and image == True and data.shape[1] == 2:
            lib.plot.flush()
            init_F.generate_image(data_plot, fake_plot_not_norm, DATASET, MODE)
            lib.plot.tick()
            print("Image Done!")

        if i % 100 == 0 and task_type == "learning":
            metrics.MAS_MSE_KL_JS_WD(data_plot, fake_plot_not_norm)
        if i % 100 == 0 and task_type == 'inference':
            prob_predict = init_F.generate_p(ground_truth, fake_plot_not_norm, column_indices, search_values,
                                             dict_V_q_not_state, dict_V_e_in)
            MAE = metrics.MAS_MSE_KL_JS_WD(np.array(ground_truth).reshape(-1, 1), np.array(prob_predict).reshape(-1, 1), is_show=False)
            if MAE < min_MAE:
                min_MAE = MAE
                best_prob_predict = prob_predict
                best_ITERS = i

    # print("{sep} Result {sep}".format(sep="-" * normalize_coefficien))
    # print("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")

    if task_type=="learning":
        return fake_plot_not_norm
    elif task_type == 'inference':
        print("min_MAE: ", min_MAE)  # (BATCH_SIZE, 2)
        print("best_ITERS :", best_ITERS)  # (BATCH_SIZE, 2)
        return best_prob_predict
    else:
        print("No task_type!")
        exit(0)