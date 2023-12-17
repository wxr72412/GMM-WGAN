import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_kernels
import tflib as lib
import tflib.plot
import init.init_F as init_F
import hyperPara
import metrics.metrics as metrics

# DATASET = hyperPara.DATASET
number_clusters = hyperPara.number_clusters
image = hyperPara.image

def run(MODE, DATASET, data, data_plot, task_type="learning", ground_truth=None, column_indices=None, search_values=None, dict_V_q_not_state=None, dict_V_e_in=None):
    # 生成一些示例数据，包括缺失值
    # X = np.array([[1], [2], [3], [4], [5]])
    # Y = np.array([[0.1], [0.2], [0.3], [0.2], [0.1]])
    # data = (np.concatenate((X, Y), axis=1))  # [采样样本, 生成分布对应的概率值]
    # print("data:", data, sep="\n")
    # print("data.shape:", data.shape, sep="\n") # (3, 2)
    # print("data_plot:", data_plot, sep="\n")
    # print("data_plot.shape:", data_plot.shape, sep="\n") # (4, 2)
    # exit(0)


    # fake_X = data_plot[:, 0:data.shape[-1]-1] # [采样样本]
    # fake_Y = data_plot[:, data.shape[-1]-1] # [实际分布对应的概率值]
    # print(fake_X)
    # print(fake_X.shape)
    # print(fake_Y)
    # print(fake_Y.shape)

    # fake = (np.concatenate((fake_X, fake_Y), axis=1))  # [采样样本, 生成分布对应的概率值]
    fake = data_plot # [采样样本, 生成分布对应的概率值]
    # print("原始数组：")
    # print(fake)
    # print(fake.shape)
    # exit(0)

    # 计算每列的最大值
    # max_values = data.max(axis=0)
    max_values = 1.0
    # max_values = fake.max(axis=0)
    # print("每列的最大值：")
    # print(fake_max_values)

    # 归一化：除以每列的最大值
    data_norm = data / max_values
    fake_norm = fake / max_values
    # print("除以每列的最大值：")
    # print(data_norm)
    # print(fake_norm)
    # exit(0)


    # 使用K-means聚类将数据分为两个簇
    kmeans = KMeans(n_clusters=number_clusters)
    kmeans.fit(data_norm)
    # print(kmeans.labels_)
    # print(data_norm[kmeans.labels_ == 0])
    # print(data_norm[kmeans.labels_ == 1])
    # exit(0)

    # 计算簇质心
    cluster_centers = kmeans.cluster_centers_
    # print(cluster_centers)
    # exit(0)


    imputation_data = []
    # 插补缺失值，使用核函数度量相似度和灰色关联分析
    for f in fake_norm:
        # print(f)
        cluster_label = kmeans.predict(f.reshape(1, -1))
        cluster_center = cluster_centers[cluster_label]
        # print("cluster_label:", cluster_label, sep="\n")
        # print("cluster_center:", cluster_center, sep="\n")
        # exit(0)

        # 使用核函数度量数据点与簇质心之间的相似度
        cluster_center_similarity_scores = pairwise_kernels(f.reshape(1, -1), cluster_center.reshape(1, -1), metric='rbf') # rbf_kernel函数计算两个向量之间的径向基函数（RBF）内核 1相同 0不相同
        # print("cluster_center_similarity_scores:", cluster_center_similarity_scores, sep="\n")
        # exit(0)

        # # 归一化相似度得到权重
        # weights = similarity_scores / np.sum(similarity_scores)

        # 计算灰色关联度
        other_similarity_scores = []
        # print(np.where(kmeans.labels_ == cluster_label)[0])
        # exit(0)
        for other_index in np.where(kmeans.labels_ == cluster_label)[0]: # 与待插值节点 同类节点的序号
            # 使用核函数度量数据点与其他数据点之间的相似度
            other_similarity_scores.append(pairwise_kernels(f.reshape(1, -1), data_norm[other_index].reshape(1, -1), metric='rbf'))
        # print("other_similarity_scores:", other_similarity_scores, sep="\n")
        # exit(0)

        # 计算权重
        weight_sum = cluster_center_similarity_scores + sum(other_similarity_scores)
        # print("weight_sum:", weight_sum, sep="\n")
        # exit(0)

        cluster_center_weight = cluster_center_similarity_scores / (weight_sum)
        # print("cluster_center_weight:", cluster_center_weight, sep="\n")
        # 使用权重进行插补
        imputed_value = cluster_center_weight * cluster_center
        # print("imputed_value:", imputed_value, sep="\n")

        for other_index, other_similarity_score in zip(np.where(kmeans.labels_ == cluster_label)[0], other_similarity_scores):  # 与待插值节点 同类节点的序号
            other_weight = other_similarity_score / (weight_sum)
            imputed_value = imputed_value + other_weight * data_norm[other_index]
        #     print("other_weight:", other_weight, sep="\n")
        #     print("data_norm[other_index]:", data_norm[other_index], sep="\n")
        #     print("imputed_value:", imputed_value, sep="\n")
        # print("imputed_value:", imputed_value, sep="\n")
        # print()
        # exit(0)
        imputation_data.append(imputed_value.reshape(-1))

    imputation_data = np.array(imputation_data)
    # 打印结果
    # print("插补后的数据:")
    # print(imputation_data)
    # print(imputation_data.shape)
    # exit(0)

    p = imputation_data[:, imputation_data.shape[-1]-1]  # [采样样本]
    p = np.expand_dims(np.array(p), axis=1)
    # print(p)
    # print(p.shape)
    # exit(0)

    sample_data_plot = fake_norm[:, 0:data_plot.shape[-1] - 1]  # [采样样本]
    fake_plot_norm = np.array(np.concatenate((sample_data_plot, p), axis=1))  # [采样样本, 生成分布对应的概率值]
    # print(fake_plot_norm)
    # exit(0)

    # 去归一化：乘以每列的最大值
    fake_plot_not_norm = fake_plot_norm * max_values
    # print("乘以每列的最大值：")
    # print(max_values)
    # print(fake_plot_not_norm)
    # exit(0)

    # print("WM!")
    if image == True and data.shape[1] == 2:
        lib.plot.flush()
        init_F.generate_image(data_plot, fake_plot_not_norm, DATASET, MODE)
        lib.plot.tick()
        print("Image Done!")


    if task_type=="learning":
        metrics.MAS_MSE_KL_JS_WD(data_plot, fake_plot_not_norm)
        return fake_plot_not_norm
    elif task_type == 'inference':
        prob_predict = init_F.generate_p(ground_truth, fake_plot_not_norm, column_indices, search_values, dict_V_q_not_state, dict_V_e_in)
        MAE = metrics.MAS_MSE_KL_JS_WD(np.array(ground_truth).reshape(-1, 1), np.array(prob_predict).reshape(-1, 1))
        return prob_predict
    else:
        print("No task_type!")
        exit(0)





