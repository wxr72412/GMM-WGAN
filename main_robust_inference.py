from pgmpy.inference import VariableElimination
from pgmpy.readwrite.BIF import BIFReader, BIFWriter
import numpy as np
import os
from itertools import product
import hyperPara
import init.init_F as init_F
import time
import metrics.metrics as metrics


def my_infer(MODE, DATASET, dict_V_e_in, dict_V_q_not_state, flag_missing_values=True):
    # MODE = hyperPara.MODE
    # DATASET = hyperPara.DATASET
    # dict_V_e_in = hyperPara.V_e_in
    # print("dict_V_e_in:", dict_V_e_in, sep="\n") # {}
    # dict_V_q_not_state = hyperPara.V_q_not
    # print("dict_V_q_not_state:", dict_V_q_not_state, sep="\n") # {'age': '40', 'bilirubin': '4', 'platelet': '125'}

    list_V_q_not = list(dict_V_q_not_state.keys())
    # print("list_V_q_not:", list_V_q_not, sep="\n") # ['age', 'bilirubin', 'platelet']
    # exit(0)

    data_path = "data\\" + DATASET + '\\'
    path_origin_BN = data_path + DATASET + '.bif'
    origin_BN = BIFReader(path_origin_BN, n_jobs=1).get_model()

    path_missing_BN = data_path + DATASET + '_missing.bif'
    missing_BN = BIFReader(path_missing_BN, n_jobs=1).get_model()
    # print("origin_BN.nodes:", origin_BN.nodes, sep="\n")
    # print("len(origin_BN.nodes):", len(origin_BN.nodes))
    # print("origin_BN.edges:", origin_BN.edges, sep="\n")
    # print("len(origin_BN.edges):", len(origin_BN.edges))
    # num_prob = 0
    # for pdf in origin_BN.get_cpds():
    #     # print(pdf)
    #     # print(len(pdf.values.reshape(-1, 1)))
    #     num_prob += len(pdf.values.reshape(-1, 1))
    # print("num_prob:", num_prob)
    # exit(0)

    # 创建 Variable Elimination 推理引擎
    VE_origin_BN = VariableElimination(origin_BN)
    VE_missing_BN = VariableElimination(missing_BN)

    # 推理计算：查询变量的概率分布
    print("VE...")
    VE_start_time = time.time()
    origin_BN_result = VE_origin_BN.query(variables=list_V_q_not, evidence=dict_V_e_in, show_progress=False)
    missing_BN_result = VE_missing_BN.query(variables=list_V_q_not, evidence=dict_V_e_in, show_progress=False)
    VE_stop_time = time.time()
    VE_time = round(VE_stop_time - VE_start_time, 4)
    print("VE_time: " + str(VE_time))
    # origin_distributions = origin_result.
    # print(origin_BN_result)
    # +---------+---------------+---------------+-------------------------------+
    # | age     | bilirubin     | platelet      |   phi(age,bilirubin,platelet) |
    # +=========+===============+===============+===============================+
    # | age(82) | bilirubin(54) | platelet(450) |                        0.0001 |
    # +---------+---------------+---------------+-------------------------------+
    # | age(82) | bilirubin(54) | platelet(225) |                        0.0012 |
    # +---------+---------------+---------------+-------------------------------+
    # | age(82) | bilirubin(54) | platelet(125) |                        0.0003 |
    # +---------+---------------+---------------+-------------------------------+
    # | age(82) | bilirubin(54) | platelet(50)  |                        0.0002 |
    # +---------+---------------+---------------+-------------------------------+
    # ...
    # print(missing_BN_result)
    # +---------+---------------+---------------+-------------------------------+
    # | age     | bilirubin     | platelet      |   phi(age,bilirubin,platelet) |
    # +=========+===============+===============+===============================+
    # | age(82) | bilirubin(54) | platelet(450) |                        0.0004 |
    # +---------+---------------+---------------+-------------------------------+
    # | age(82) | bilirubin(54) | platelet(225) |                        0.0037 |
    # +---------+---------------+---------------+-------------------------------+
    # | age(82) | bilirubin(54) | platelet(50)  |                        0.0006 |
    # +---------+---------------+---------------+-------------------------------+
    # ...
    # print(origin_result.values)
    # print(origin_result.variables) # ['age', 'bilirubin', 'platelet']
    # exit(0)

    if list_V_q_not != origin_BN_result.variables or list_V_q_not != missing_BN_result.variables:
        print("list_V_q_not != origin_result.variables or list_V_q_not != missing_BN_result.variables!!!")
        exit(0)
    # exit(0)

    dict_V_q_all_state = dict_V_q_not_state.copy()
    dict_V_q_in_state = dict_V_q_not_state.copy()
    # print("dict_V_q_in_state:", dict_V_q_in_state, sep="\n")

    # print("list_V_q_not:", list_V_q_not, sep="\n")
    for V in list_V_q_not:
        # print(missing_BN.get_cpds(V).state_names[V])
        dict_V_q_all_state[V] = [int(item) for item in origin_BN.get_cpds(V).state_names[V]]
        dict_V_q_in_state[V] = [int(item) for item in missing_BN.get_cpds(V).state_names[V]]
    # print("dict_V_q_all_state:", dict_V_q_all_state, sep="\n")  # {'age': [82, 58, 40, 15], 'bilirubin': [54, 13, 4, 0], 'platelet': [450, 225, 125, 50]}
    # print("dict_V_q_in_state:", dict_V_q_in_state, sep="\n") # {'age': [82, 58, 15], 'bilirubin': [54, 13, 0], 'platelet': [450, 225, 50]}
    # print("dict_V_q_not_state:", dict_V_q_not_state, sep="\n") # {'age': '40', 'bilirubin': '4', 'platelet': '125'}
    # exit(0)

    # 提取字典的值作为列表
    origin_BN_value_lists = list(dict_V_q_all_state.values())
    missing_BN_value_lists = list(dict_V_q_in_state.values())
    # print("origin_BN_value_lists:", origin_BN_value_lists, sep="\n") # [[82, 58, 40, 15], [54, 13, 4, 0], [450, 225, 125, 50]]
    # print("missing_BN_value_lists:", missing_BN_value_lists, sep="\n") # [[82, 58, 15], [54, 13, 0], [450, 225, 50]]
    # exit(0)

    # 计算值的笛卡尔积
    origin_BN_data_cartesian_product = list(product(*origin_BN_value_lists))
    missing_BN_data_cartesian_product = list(product(*missing_BN_value_lists))
    # print("origin_BN_data_cartesian_product:", origin_BN_data_cartesian_product, sep="\n")
    # [(82, 54, 450), (82, 54, 225), (82, 54, 125), (82, 54, 50),
    # (82, 13, 450), (82, 13, 225), (82, 13, 125), (82, 13, 50),
    # (82, 4, 450), (82, 4, 225), (82, 4, 125), (82, 4, 50),
    # (82, 0, 450), (82, 0, 225), (82, 0, 125), (82, 0, 50),
    # (58, 54, 450), (58, 54, 225), (58, 54, 125), (58, 54, 50), (58, 13, 450), (58, 13, 225), (58, 13, 125), (58, 13, 50), (58, 4, 450), (58, 4, 225), (58, 4, 125), (58, 4, 50), (58, 0, 450), (58, 0, 225), (58, 0, 125), (58, 0, 50), (40, 54, 450), (40, 54, 225), (40, 54, 125), (40, 54, 50), (40, 13, 450), (40, 13, 225), (40, 13, 125), (40, 13, 50), (40, 4, 450), (40, 4, 225), (40, 4, 125), (40, 4, 50), (40, 0, 450), (40, 0, 225), (40, 0, 125), (40, 0, 50), (15, 54, 450), (15, 54, 225), (15, 54, 125), (15, 54, 50), (15, 13, 450), (15, 13, 225), (15, 13, 125), (15, 13, 50), (15, 4, 450), (15, 4, 225), (15, 4, 125), (15, 4, 50), (15, 0, 450), (15, 0, 225), (15, 0, 125), (15, 0, 50)]
    # print("missing_BN_data_cartesian_product:", missing_BN_data_cartesian_product, sep="\n")
    #[(82, 54, 450), (82, 54, 225), (82, 54, 50),
    # (82, 13, 450), (82, 13, 225), (82, 13, 50),
    # (82, 0, 450), (82, 0, 225), (82, 0, 50),
    # (58, 54, 450), (58, 54, 225), (58, 54, 50), (58, 13, 450), (58, 13, 225), (58, 13, 50), (58, 0, 450), (58, 0, 225), (58, 0, 50), (15, 54, 450), (15, 54, 225), (15, 54, 50), (15, 13, 450), (15, 13, 225), (15, 13, 50), (15, 0, 450), (15, 0, 225), (15, 0, 50)]
    # exit(0)

    # 遍历每个 C 取值并显示概率值
    origin_BN_data_probabilities = origin_BN_result.values.reshape(-1)
    missing_BN_data_probabilities = missing_BN_result.values.reshape(-1)
    if len(origin_BN_data_cartesian_product) != len(origin_BN_data_probabilities) or len(missing_BN_data_cartesian_product) != len(missing_BN_data_probabilities):
        print("len(origin_BN_data_cartesian_product) != len(origin_BN_result.values) or len(missing_BN_data_cartesian_product) != len(missing_BN_result.values)!!!")
        exit(0)
    # for state, probability in zip(origin_BN_data_cartesian_product, origin_BN_result.values.reshape(-1)):
    #     print(state)
    #     print(probability)
    # exit(0)

    origin_BN_data_samples = np.array(origin_BN_data_cartesian_product)
    missing_BN_data_samples = np.array(missing_BN_data_cartesian_product)
    origin_BN_data_probabilities = np.array(origin_BN_data_probabilities).reshape(-1, 1)
    missing_BN_data_probabilities = np.array(missing_BN_data_probabilities).reshape(-1, 1)
    # print("origin_BN_data_samples:", origin_BN_data_samples, sep="\n") # [[ 82  54 450] [ 82  54 225] ... [ 15   0  50]]
    # print("origin_BN_data_samples.shape:", origin_BN_data_samples.shape, sep="\n") # (64, 3)
    # print("missing_BN_data_samples:", missing_BN_data_samples, sep="\n") # [[ 82  54 450] [ 82  54 225] ... [ 15   0  50]]
    # print("missing_BN_data_samples.shape:", missing_BN_data_samples.shape, sep="\n") # (27, 3)
    # print("origin_BN_data_probabilities:", origin_BN_data_probabilities, sep="\n")
    # print("origin_BN_data_probabilities.shape:", origin_BN_data_probabilities.shape, sep="\n") # (64,)
    # print("missing_BN_data_probabilities:", missing_BN_data_probabilities, sep="\n")
    # print("missing_BN_data_probabilities.shape:", missing_BN_data_probabilities.shape, sep="\n") # (27,)

    origin_BN_data = np.concatenate((origin_BN_data_samples, origin_BN_data_probabilities), axis=1)
    missing_BN_data = np.concatenate((missing_BN_data_samples, missing_BN_data_probabilities), axis=1)
    # print("origin_BN_data:", origin_BN_data, sep="\n")
    # [[8.20000000e+01 5.40000000e+01 4.50000000e+02 1.20470728e-04]
    #  [8.20000000e+01 5.40000000e+01 2.25000000e+02 1.20765064e-03]
    #  [8.20000000e+01 5.40000000e+01 1.25000000e+02 2.86588672e-04]
    #  [8.20000000e+01 5.40000000e+01 5.00000000e+01 1.86080493e-04]
    # ...
    print("origin_BN_data.shape:", origin_BN_data.shape, sep="\n") # (64, 4)
    # print("origin_BN_data.type:", type(origin_BN_data), sep="\n") # <class 'numpy.ndarray'>
    # print("missing_BN_data:", missing_BN_data, sep="\n")
    # [[8.20000000e+01 5.40000000e+01 4.50000000e+02 3.64543521e-04]
    #  [8.20000000e+01 5.40000000e+01 2.25000000e+02 3.68357350e-03]
    #  [8.20000000e+01 5.40000000e+01 5.00000000e+01 5.81563648e-04]
    # ...
    # [1.50000000e+01 0.00000000e+00 5.00000000e+01 4.13126681e-03]]
    print("missing_BN_data.shape:", missing_BN_data.shape, sep="\n") # (27, 4)
    # print("missing_BN_data.type:", type(missing_BN_data), sep="\n") # <class 'numpy.ndarray'>
    # exit(0)

    # 指定要搜索的列索引和搜索值
    # search_values = [82, 54, 125]
    search_values = [int(item) for item in dict_V_q_not_state.values()]  # 缺值
    column_indices = [int(item) for item in range(len(search_values))]  # 搜索缺值对应的列索引:
    # print(search_values) # [40, 4, 125]
    # print(column_indices) # [0, 1, 2]


    # 使用布尔索引查找符合条件的行： 缺值在原始BN推理出的真实概率值
    matching_rows = origin_BN_data[np.all(origin_BN_data[:, column_indices] == search_values, axis=1)][0]
    # print("matching_rows:", matching_rows, sep="\n") # [4.00000000e+01 4.00000000e+00 1.25000000e+02 1.50066804e-02]

    # 原始BN推理出的真实概率值
    ground_truth = matching_rows[-1]
    # print("ground_truth:", ground_truth, sep="\n") # 0.015006680397000767
    print("ground_truth: P(" + str(dict_V_q_not_state) + "|" + str(dict_V_e_in) + ")" + " = " + str(ground_truth))
    # exit(0)

    if flag_missing_values == False:
        return ground_truth, VE_time

    # exit(0)


    # 学习概率密度，离散化+归一化，得到预测的概率值
    data = missing_BN_data
    data_plot = origin_BN_data
    # print("data:", data, sep="\n")
    # print("data.shape:", data.shape, sep="\n") # (3, 2)
    # print("data_plot:", data_plot, sep="\n")
    # print("data_plot.shape:", data_plot.shape, sep="\n") # (4, 2)
    # exit(0)

    # 当变量数量较多时，增加采样的样本数
    if MODE == 'GMM' or MODE == 'KDE':
        if data.shape[0] > hyperPara.BATCH_SIZE:
            hyperPara.BATCH_SIZE = data.shape[0]
            print("hyperPara.BATCH_SIZE:", hyperPara.BATCH_SIZE, sep="\n")

    # print(MODE)
    #

    Learning_start_time = time.time()
    from learning_prob import run_GAN, run_KDE, run_SVM, run_GMM, run_WM
    task_type = "inference"
    print('learning...')
    if MODE == 'GAN' or MODE == 'WGAN' or MODE == 'GMM_in_GAN' or MODE == 'GMM_in_WGAN':
        prob_predict = run_GAN.run(MODE, DATASET, data, data_plot, task_type, ground_truth, column_indices, search_values, dict_V_q_not_state, dict_V_e_in)
    elif MODE == 'GMM':
        prob_predict = run_GMM.run(MODE, DATASET, data, data_plot, task_type, ground_truth, column_indices, search_values, dict_V_q_not_state, dict_V_e_in)
    elif MODE == 'KDE':
        prob_predict = run_KDE.run(MODE, DATASET, data, data_plot, task_type, ground_truth, column_indices, search_values, dict_V_q_not_state, dict_V_e_in)
    elif MODE == 'SVM':
        prob_predict = run_SVM.run(MODE, DATASET, data, data_plot, task_type, ground_truth, column_indices, search_values, dict_V_q_not_state, dict_V_e_in)
    elif MODE == 'WM':
        prob_predict = run_WM.run(MODE, DATASET, data, data_plot, task_type, ground_truth, column_indices, search_values, dict_V_q_not_state, dict_V_e_in)
    else:
        print("No Mode!")
        exit(0)

    print("prob_predict: P(" + str(dict_V_q_not_state) + "|" + str(dict_V_e_in) + ")" + " = " + str(prob_predict))
    Learning_stop_time = time.time()
    Learning_time = round(Learning_stop_time - Learning_start_time, 4)
    # print("Learning_time: " + str(Learning_time))
    # print("fake:", fake, sep="\n")
    # print("fake.shape:", fake.shape, sep="\n") # (4, 2)
    # exit(0)
    if ground_truth <= 0:
        ground_truth = 0.000001
        print("prob_predict < 0 !!!")
    if prob_predict <= 0:
        prob_predict = 0.000001
        print("prob_predict < 0 !!!")

    return ground_truth, prob_predict, VE_time, Learning_time




##########################################################
##########################################################
##########################################################
init_F.set_seed(hyperPara.seed)
hyperPara.image = True
# list_MODE = ['WM', 'SVM', 'KDE']
# list_MODE = ['WM']
# list_MODE = ['SVM']
# list_MODE = ['KDE']
# list_MODE = ['GMM']
# list_MODE = ['GAN']
# list_MODE = ['WGAN']
# list_MODE = ['GMM_WGAN']
# list_MODE = ['GMM_in_GAN']
list_MODE = ['GMM_in_WGAN']


# list_DATASET = ['hepar2']
# list_DATASET = ['heart']
# list_DATASET = ['bone']
list_DATASET = ['munin1']

# 默认4类证据/查询变量为空 {}
list_V_e_in = {}
list_V_e_not = {}
list_V_q_in = {}
list_V_q_not = {}



# 设置4类证据/查询变量
for DATASET in list_DATASET:
    if DATASET == 'hepar2':
        # case3
        list_V_q_in = {'Hyperbilirubinemia': '1'}
        # 'WM', 'SVM', 'KDE', 'GMM'
        list_V_e_not = [
                        # {'age': '82'},
                        # {'age': '82', 'bilirubin': '4'},
                        {'age': '82', 'bilirubin': '4', 'platelet': '125'},
                        {'age': '82', 'bilirubin': '4', 'platelet': '125', 'alt': '67'}]
        # GAN
        # list_V_e_not = [{'age': '82'}]
        # list_V_e_not = [{'age': '82', 'bilirubin': '4'}]
        # list_V_e_not = [{'age': '82', 'bilirubin': '4', 'platelet': '125'}]
        # list_V_e_not = [{'age': '82', 'bilirubin': '4', 'platelet': '125', 'alt': '67'}]

        # case7
        # 'WM', 'SVM', 'KDE', 'GMM'
        # list_V_q_not = [{'age': '82'}, {'age': '82', 'bilirubin': '4'},
        #                 {'age': '82', 'bilirubin': '4', 'platelet': '125'},
        #                 {'age': '82', 'bilirubin': '4', 'platelet': '125', 'alt': '67'}]
        # GAN
        # list_V_q_not = [{'age': '82'}]
        # list_V_q_not = [{'age': '82', 'bilirubin': '4'}]
        # list_V_q_not = [{'age': '82', 'bilirubin': '4', 'platelet': '125'}]
        # list_V_q_not = [{'age': '82', 'bilirubin': '4', 'platelet': '125', 'alt': '67'}]

    elif DATASET == 'munin1':
        # case3
        list_V_q_in = {'R_APB_QUAN_MUPPOLY': '12'}
        # 'WM', 'SVM', 'KDE', 'GMM'
        # list_V_e_not = [
        #                 {'R_MED_DCV_EW': '52'},
        #                 {'R_MED_DCV_EW': '52', 'R_APB_ALLAMP_WA': '10'},
                        # {'R_MED_DCV_EW': '52', 'R_APB_ALLAMP_WA': '10', 'R_MEDD2_ALLCV_WD': '44'},
                        # {'R_MED_DCV_EW': '52', 'R_APB_ALLAMP_WA': '10', 'R_MEDD2_ALLCV_WD': '44', 'R_MEDD2_ALLAMP_WD': '10'}]
        # GAN
        # list_V_e_not = [{'R_MED_DCV_EW': '52'}]
        list_V_e_not = [{'R_MED_DCV_EW': '52', 'R_APB_ALLAMP_WA': '10'}]
        # list_V_e_not = [{'R_MED_DCV_EW': '52', 'R_APB_ALLAMP_WA': '10', 'R_MEDD2_ALLCV_WD': '44'}]
        # list_V_e_not = [{'R_MED_DCV_EW': '52', 'R_APB_ALLAMP_WA': '10', 'R_MEDD2_ALLCV_WD': '44', 'R_MEDD2_ALLAMP_WD': '10'}]

        # case7
        # 'WM', 'SVM', 'KDE', 'GMM'
        # list_V_q_not = [{'R_MED_DCV_EW': '52'}, {'R_MED_DCV_EW': '52', 'R_APB_ALLAMP_WA': '0.1'},
        #                 {'R_MED_DCV_EW': '52', 'R_APB_ALLAMP_WA': '10', 'R_MEDD2_ALLCV_WD': '44'},
        #                 {'R_MED_DCV_EW': '52', 'R_APB_ALLAMP_WA': '10', 'R_MEDD2_ALLCV_WD': '44', 'R_MEDD2_ALLAMP_WD': '10'}]
        # GAN
        # list_V_q_not = [{'R_MED_DCV_EW': '52'}]
        # list_V_q_not = [{'R_MED_DCV_EW': '52', 'R_APB_ALLAMP_WA': '10'}]
        # list_V_q_not = [{'R_MED_DCV_EW': '52', 'R_APB_ALLAMP_WA': '10', 'R_MEDD2_ALLCV_WD': '44'}]
        # list_V_q_not = [{'R_MED_DCV_EW': '52', 'R_APB_ALLAMP_WA': '10', 'R_MEDD2_ALLCV_WD': '44', 'R_MEDD2_ALLAMP_WD': '10'}]

    elif DATASET == 'heart':
        # case3
        list_V_q_in = {'num': '1'}
        # 'WM', 'SVM', 'KDE', 'GMM'
        list_V_e_not = [
                        # {'age': '60'},
                        # {'age': '60', 'trestbps': '160'},
                        {'age': '60', 'trestbps': '160', 'thalach': '160'},
                        {'age': '60', 'trestbps': '160', 'thalach': '160', 'oldpeak': '1'}]

        # GAN
        # list_V_e_not = [{'age': '60'}]
        # list_V_e_not = [{'age': '60', 'trestbps': '160'}]
        # list_V_e_not = [{'age': '60', 'trestbps': '160', 'thalach': '160'}]
        # list_V_e_not = [{'age': '60', 'trestbps': '160', 'thalach': '160', 'oldpeak': '1'}]
        #
        # case7
        # 'WM', 'SVM', 'KDE', 'GMM'
        # list_V_q_not = [{'age': '60'}, {'age': '60', 'trestbps': '160'},
        #                 {'age': '60', 'trestbps': '160', 'thalach': '160'},
        #                 {'age': '60', 'trestbps': '160', 'thalach': '160', 'oldpeak': '1'}]
        # GAN
        # list_V_q_not = [{'age': '60'}]
        # list_V_q_not = [{'age': '60', 'trestbps': '160'}]
        # list_V_q_not = [{'age': '60', 'trestbps': '160', 'thalach': '160'}]
        # list_V_q_not = [{'age': '60', 'trestbps': '160', 'thalach': '160', 'oldpeak': '1'}]

    elif DATASET == 'bone':
        # case3
        list_V_q_in = {'survival_status': '1'}
        # list_V_q_in = {'survival_status': '0'}
        # 'WM', 'SVM', 'KDE', 'GMM'
        # list_V_e_not = [
                        # {'Recipient_age': '15'},quannegnben
                        # {'Recipient_age': '15', 'CD34_kgx10d6': '20'},
                        # {'Recipient_age': '15', 'CD34_kgx10d6': '20', 'CD3d_kgx10d8': '15'},
                        # {'Recipient_age': '15', 'CD34_kgx10d6': '20', 'CD3d_kgx10d8': '15', 'Rbody_mass': '40'}]
        # GAN
        # list_V_e_not = [{'Recipient_age': '15'}]
        # list_V_e_not = [{'Recipient_age': '15', 'CD34_kgx10d6': '20'}]
        list_V_e_not = [{'Recipient_age': '15', 'CD34_kgx10d6': '20', 'Rbody_mass': '40'}]
        # list_V_e_not = [{'Recipient_age': '15', 'CD34_kgx10d6': '20', 'CD3d_kgx10d8': '15', 'S': '40'}]

        # # case7
        # # 'WM', 'SVM', 'KDE', 'GMM'
        # list_V_q_not = [{'Recipient_age': '15'}, {'Recipient_age': '15', 'CD34_kgx10d6': '20'},
        #                 {'Recipient_age': '15', 'CD34_kgx10d6': '20', 'CD3d_kgx10d8': '15'},
        #                 {'Recipient_age': '15', 'CD34_kgx10d6': '20', 'CD3d_kgx10d8': '15', 'Rbody_mass': '40'}]
        # # GAN
        # # list_V_q_not = [{'Recipient_age': '15'}]
        # # list_V_q_not = [{'Recipient_age': '15', 'CD34_kgx10d6': '20'}]
        # # list_V_q_not = [{'Recipient_age': '15', 'CD34_kgx10d6': '20', 'CD3d_kgx10d8': '15'}]
        # # list_V_q_not = [{'Recipient_age': '15', 'CD34_kgx10d6': '20', 'CD3d_kgx10d8': '15', 'Rbody_mass': '40'}]


    # print("list_V_e_in: " + str(list_V_e_in))
    # print("list_V_e_not: " + str(list_V_e_not))
    # print("list_V_q_in: " + str(list_V_q_in))
    # print("list_V_q_not: " + str(list_V_q_not))

    # case3:  all values in 𝐯𝐸 and no values in 𝐯𝑄 that are missing in 𝜃
    if list_V_e_in == {} and list_V_e_not != {} and list_V_q_in != {} and list_V_q_not == {}:
        # pass
        for V_e_not in list_V_e_not:
            for MODE in list_MODE:
                print("case3!")
                start_time = time.time()
                ground_truth_p12, VE_time_p12 = my_infer(MODE, DATASET, dict_V_e_in={}, dict_V_q_not_state=list_V_q_in, flag_missing_values=False)

                if MODE == 'GAN' or MODE == 'WGAN' or MODE == 'GMM_in_GAN' or MODE == 'GMM_in_WGAN':
                    hyperPara.Dim_G_Input = len(V_e_not)
                    hyperPara.Dim_D_Input = hyperPara.Dim_G_Input + 1
                    # print("hyperPara.Dim_G_Input", hyperPara.Dim_G_Input, sep="\n")
                    # print("hyperPara.Dim_D_Input", hyperPara.Dim_D_Input, sep="\n")

                # if MODE == 'GAN' or MODE == 'WGAN':
                #     hyperPara.lr = 1e-4
                # elif MODE == 'GMM_in_WGAN':
                #     hyperPara.lr = 0.0025

                ground_truth_p11, prob_predict_p11, VE_time_p11, Learning_time_p11 = my_infer(MODE, DATASET, dict_V_e_in=list_V_q_in, dict_V_q_not_state=V_e_not)
                print("VE_time_p11: " + str(VE_time_p11) + ", Learning_time_p11: " + str(Learning_time_p11))
                metrics.MAS_MSE_KL_JS_WD(np.array(ground_truth_p11).reshape(-1, 1), np.array(prob_predict_p11).reshape(-1, 1))
                # exit(0)

                # if MODE == 'GAN' or MODE == 'WGAN' or MODE == 'GMM_in_GAN' or MODE == 'GMM_in_WGAN':
                #     hyperPara.Dim_G_Input = len(V_e_not)
                #     hyperPara.Dim_D_Input = hyperPara.Dim_G_Input + 1
                #     # print("hyperPara.Dim_G_Input", hyperPara.Dim_G_Input, sep="\n")
                #     # print("hyperPara.Dim_D_Input", hyperPara.Dim_D_Input, sep="\n")
                # ground_truth_p2, prob_predict_p2, VE_time_p2, Learning_time_p2 = my_infer(MODE, DATASET, dict_V_e_in={},dict_V_q_not_state=V_e_not)
                # metrics.MAS_MSE_KL_JS_WD(np.array(ground_truth_p2).reshape(-1, 1), np.array(prob_predict_p2).reshape(-1, 1))
                # stop_time = time.time()
                #
                #
                # if ground_truth_p11 == 0 or ground_truth_p12 == 0:
                #     ground_truth = 0.0
                # else:
                #     ground_truth = round((ground_truth_p11*ground_truth_p12)/ground_truth_p2, 8)
                # prob_predict = round((prob_predict_p11 * ground_truth_p12) / prob_predict_p2, 8)
                # print("case3 " + "P(" + str(list_V_q_in) + "|" + str(V_e_not) + "): " + DATASET + ", " + MODE)
                # print("ground_truth: " + str(ground_truth) + ", prob_predict: " + str(prob_predict))
                # metrics.MAS_MSE_KL_JS_WD(np.array(ground_truth).reshape(-1, 1), np.array(prob_predict).reshape(-1, 1))
                # print("total time: " + str(round(stop_time - start_time, 4)) +
                #       ", VE_time: " + str(VE_time_p11+VE_time_p2) +
                #       ", Learning_time: " + str(Learning_time_p11+Learning_time_p2))
                # print()



    # case7
    elif list_V_e_not == {} and list_V_q_in == {} and list_V_q_not != {}:
        print("case7!")
        for V_q_not in list_V_q_not:
            for MODE in list_MODE:
                if MODE == 'GAN' or MODE == 'WGAN' or MODE == 'GMM_in_GAN' or MODE == 'GMM_in_WGAN':
                    hyperPara.Dim_G_Input = len(V_q_not)
                    hyperPara.Dim_D_Input = hyperPara.Dim_G_Input + 1
                    # print("hyperPara.Dim_G_Input", hyperPara.Dim_G_Input, sep="\n")
                    # print("hyperPara.Dim_D_Input", hyperPara.Dim_D_Input, sep="\n")
                start_time = time.time()
                ground_truth, prob_predict = my_infer(MODE, DATASET, list_V_e_in, V_q_not)
                stop_time = time.time()
                metrics(ground_truth, prob_predict)
                print(DATASET + ", : " + MODE + ", : " + str(V_q_not) + ", total time: " + str(round(stop_time-start_time, 4)))
    else:
        print("other cases!")