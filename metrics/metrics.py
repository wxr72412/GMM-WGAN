import numpy as np
import scipy.stats
import cv

#
# def KL_divergence_2(p, q):
#     return scipy.stats.entropy(p, q)
#
# def JS_divergence_2(p, q):
#     M = (p + q) / 2
#     return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


def KL_divergence(p, q):
    """
    有时也称为相对熵，KL距离。对于两个概率分布P、Q，二者越相似，KL散度越小。
    KL散度满足非负性
    KL散度是不对称的，交换P、Q的位置将得到不同结果。
    """
    return p * np.log(p / q)

def JS_divergence(p, q):
    """
    JS散度基于KL散度，同样是二者越相似，JS散度越小。
        JS散度的取值范围在0-1之间，完全相同时为0
        JS散度是对称的
    """
    M = (p + q) / 2
    return 0.5 * KL_divergence(p, M) + 0.5 * KL_divergence(q, M)

def Wasserstein(p, q):
    return scipy.stats.wasserstein_distance(p, q)



def MAS_MSE_KL_JS_WD(data_plot, fake_plot, is_show=True):
    # print("data_plot:", data_plot, sep="\n")
    # print("data_plot.shape:", data_plot.shape, sep="\n") # (4, 2)
    # print("fake_plot:", fake_plot, sep="\n")
    # print("fake_plot.shape:", fake_plot.shape, sep="\n") # (4, 2)
    # exit(0)

    p_groudtruth = data_plot[:, data_plot.shape[-1] - 1].reshape(-1, 1)
    p_predict = fake_plot[:, fake_plot.shape[-1] - 1].reshape(-1, 1)
    # print("p_groudtruth:", p_groudtruth, sep="\n")
    # print("p_fake:", p_predict, sep="\n") # (4, 2)
    # exit(0)

    all_MAE = abs(p_groudtruth - p_predict)
    all_MSE = (p_groudtruth - p_predict) ** 2
    # print("all_MAE:", all_MAE, sep="\n")
    # print("all_MSE:", all_MSE, sep="\n") # (4, 2)
    # exit(0)

    ave_MAE = np.average(all_MAE)
    ave_MSE = np.average(all_MSE)


    max_MAE = np.max(all_MAE)
    max_MSE = np.max(all_MSE)


    all_KL = KL_divergence(p_groudtruth, p_predict)
    KL = np.sum(all_KL)
    ave_KL = np.average(all_KL)
    max_KL = np.max(all_KL)


    all_JS = JS_divergence(p_groudtruth, p_predict)
    JS = np.sum(all_JS)
    ave_JS = np.average(all_JS)
    max_JS = np.max(all_MAE)


    WD = Wasserstein(p_groudtruth.reshape(-1), p_predict.reshape(-1))

    if is_show:
        print("ave_MAE:", ave_MAE)
        # print("ave_MSE:", ave_MSE, sep="\n")  # (4, 2)
        print("max_MAE:", max_MAE)
        # print("max_MSE:", max_MSE, sep="\n")  # (4, 2)
        # print("all_KL:", all_KL, sep="\n")
        # print("KL:", KL, sep="\n")
        print("ave_KL:", ave_KL)
        # print("max_KL:", max_KL, sep="\n")  # (4, 2)
        # print("all_JS:", all_JS, sep="\n")  # (4, 2)
        # print("JS:", JS, sep="\n")
        print("ave_JS:", ave_JS)
        # print("max_JS:", max_JS, sep="\n")  # (4, 2)
        # exit(0)
        print("WD:", WD)  # (4, 2)

    return ave_MAE, WD

