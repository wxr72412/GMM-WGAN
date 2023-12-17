import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import hyperPara
import init.init_F as init_F
import tflib as lib
import tflib.plot
import metrics.metrics as metrics

# DATASET = hyperPara.DATASET
image = hyperPara.image

def run(MODE, DATASET, data, data_plot, task_type="learning", ground_truth=None, column_indices=None, search_values=None, dict_V_q_not_state=None, dict_V_e_in=None):

    # # 创建一个示例多维输入数据集
    # X_list = [[0.4], [0.3], [0.5], [0.1], [0.3], [0.6], [0.8], [0.7], [0.9], [0.75]]
    # y_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # # X = np.random.rand(10, 1)  # 假设有100个样本，每个样本有10个特征
    # X = np.array(X_list)
    # # y = np.random.randint(0, 2, 10)  # 二元分类任务，随机生成标签
    # y = np.array(y_list)
    # print("X:", X, sep="\n")
    # print("X.shape:", X.shape, sep="\n")
    # print("y:", y, sep="\n")
    # print("y.shape:", y.shape, sep="\n")
    # exit(0)


    # print("data:", data, sep="\n")
    # print("data.shape:", data.shape, sep="\n")

    weights = data[:, data.shape[-1] - 1]  # [实际分布对应的概率值]
    # print("weights:", weights, sep="\n")
    # print("weights.shape:", weights.shape, sep="\n")
    sample_data = data[:, 0:data.shape[-1] - 1]  # [采样样本]
    # print("sample_data:", sample_data, sep="\n")
    # print("sample_data.shape:", sample_data.shape, sep="\n") # (BATCH_SIZE, 1)
    # exit(0)

    X = sample_data[:500]
    # print("X:", X, sep="\n")  # [ 0  1 ... 63]
    # print("X.shape:", X.shape, sep="\n")
    # [[-10.]
    #  [-9.68254]
    #  ...
    #  [10.]]
    # exit(0)

    # 初始化一个空字典来存储键和对应的序号
    sample_to_index = {}
    prob_to_index = {}
    # 初始化一个序号计数器
    index = 0
    # 遍历列表中的元素
    for item, prob in zip(X, weights):
        # print(item)
        # print(prob)
        # 将子列表转换为元组，以便作为键
        key = tuple(item)
        # 如果元组作为键还没有在字典中，将其添加到字典并分配一个新的序号
        if key not in sample_to_index:
            sample_to_index[key] = index
            prob_to_index[index] = prob
            index += 1
    # 打印键和对应的序号
    # for key, value in sample_to_index.items():
    #     print(key, value)
    # (-10.0,) 0
    # (-9.68254,) 1
    # ...
    # (10.0,) 63
    # for key, value in prob_to_index.items():
    #     print(key, value)
    # 0 1.4892091e-06
    # 1 3.9098686e-06
    ...
    # 63 3.9063662e-18
    # exit(0)

    y_list = []
    # 遍历列表中的元素
    for item in X:
        # 将子列表转换为元组，以便作为键
        key = tuple(item)
        y_list.append(sample_to_index[key])
    y = np.array(y_list)[:500]
    # print("y:", y, sep="\n") # [ 0  1 ... 63]
    # print("y.shape:", y.shape, sep="\n")
    # exit(0)


    # 划分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(X_train)
    # print(y_train)
    # print(X_train.shape)
    # print(X_test)
    # print(y_test)
    # print(X_test.shape)
    # exit(0)

    # 标准化数据
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # print(X_train)
    # exit(0)

    # 创建SVM分类器
    clf = svm.SVC(probability=True)  # probability=True启用概率估计

    # 训练（拟合）模型
    clf.fit(X, y)

    # 使用CalibratedClassifierCV进行概率校准
    calibrated_clf = CalibratedClassifierCV(estimator=clf, method='sigmoid', cv='prefit')
    calibrated_clf.fit(X, y)



    # 获取概率估计
    sample_data_plot = data_plot[:, 0:data_plot.shape[-1] - 1]  # [采样样本]
    # print(sample_data_plot)
    # exit(0)

    y_prob_plot = calibrated_clf.predict_proba(sample_data_plot)  # 选择属于各类的概率值
    # print(y_prob_plot)
    # exit(0)
    # 找到每行中最大概率值的序号
    max_prob_indices = np.argmax(y_prob_plot, axis=1)
    # print(max_prob_indices)
    # print(max_prob_indices.shape)
    # exit(0)

    # print(prob_to_index)
    p = []
    for index in max_prob_indices:
        p.append(prob_to_index[index])
    p = np.expand_dims(np.array(p), axis=1)
    # print(sample_data_plot.shape)
    # print(p.shape)
    # exit(0)

    fake = np.array(np.concatenate((sample_data_plot, p), axis=1))  # [采样样本, 生成分布对应的概率值]
    # print(fake)

    # print(data.shape[1])
    # exit(0)

    # print("SVM!")
    if image == True and data.shape[1] == 2:
        # print("fake:", fake, sep="\n")
        # print("fake.shape:", fake.shape, sep="\n")  # (BATCH_SIZE, 2)
        # print(type(fake)) # <class 'numpy.ndarray'>
        # print("data:", data, sep="\n")  # (BATCH_SIZE, 2)
        # print("data.shape:", data.shape, sep="\n")  # (BATCH_SIZE, 2)
        # print(type(data)) # <class 'numpy.ndarray'>
        # exit(0)
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

