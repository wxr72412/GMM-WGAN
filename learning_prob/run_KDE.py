import matplotlib.pyplot as plt
import numpy as np
import random
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import os, sys
sys.path.append(os.getcwd())

import init.init_F as init_F
import hyperPara
import GMM.gmm as GMM
import tflib as lib
import tflib.plot
from sklearn.neighbors import KernelDensity
import metrics.metrics as metrics

# DATASET = hyperPara.DATASET
num_components = hyperPara.num_components
GMM_ITERS = hyperPara.GMM_ITERS
BATCH_SIZE = hyperPara.BATCH_SIZE
image = hyperPara.image


def run(MODE, DATASET, data, data_plot, task_type="learning", ground_truth=None, column_indices=None, search_values=None, dict_V_q_not_state=None, dict_V_e_in=None):

    weights = data[:, data.shape[-1]-1] # [实际分布对应的概率值]
    # print("weights:", weights, sep="\n")
    # print("weights.shape:", weights.shape, sep="\n")
    sample_data = data[:, 0:data.shape[-1]-1] # [采样样本]
    # print("sample_data:", sample_data, sep="\n")
    # print("sample_data.shape:", sample_data.shape, sep="\n") # (BATCH_SIZE, 1)
    # exit(0)
    Y = random.choices(sample_data, weights=weights, cum_weights=None, k=BATCH_SIZE) # 根据CPT计算的离散概率值，采样出对应的样本
    # Y = np.matrix(Y, copy=True)
    # print("Y:", Y, sep="\n")
    # print("Y.shape:", Y.shape, sep="\n")
    # exit(0)


    # 创建KDE模型并拟合数据
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
    kde.fit(Y)

    # 生成一系列值用于估计概率密度函数
    sample_data_plot = data_plot[:, 0:data_plot.shape[-1] - 1]  # [采样样本]
    # print("sample_data_plot:", sample_data_plot, sep="\n")

    # 使用KDE模型估计概率密度函数
    p = np.exp(kde.score_samples(sample_data_plot))
    p = np.expand_dims(np.array(p), axis=1)
    # print("p:", p, sep="\n")
    # print("p.shape:", p.shape, sep="\n")

    fake = np.array(np.concatenate((sample_data_plot, p), axis=1))  # [采样样本, 生成分布对应的概率值]
    # print("fake:", fake, sep="\n")
    # print("fake.shape:", fake.shape, sep="\n")
    # exit(0)


    if image == True and data.shape[1] == 2:
        lib.plot.flush()
        init_F.generate_image(data_plot, fake, DATASET, MODE)
        lib.plot.tick()
        print("Image Done!")

    if task_type == "learning":
        metrics.MAS_MSE_KL_JS_WD(data_plot, fake)
        return fake
    elif task_type == 'inference':
        prob_predict = init_F.generate_p(ground_truth, fake, column_indices, search_values,
                                         dict_V_q_not_state, dict_V_e_in)
        MAE = metrics.MAS_MSE_KL_JS_WD(np.array(ground_truth).reshape(-1, 1), np.array(prob_predict).reshape(-1, 1))
        return prob_predict
    else:
        print("No task_type!")
        exit(0)
