import numpy as np

import init.init_F as init_F
import hyperPara
from learning_prob import run_GAN, run_KDE, run_SVM, run_GMM, run_WM
import time

MODE = hyperPara.MODE
BATCH_SIZE = hyperPara.BATCH_SIZE  # 训练样本，作为图上的概率密度函数
BATCH_SIZE_plot = hyperPara.BATCH_SIZE_plot  # 待预测概率值的样本，作为图上的实际概率值
DATASET = hyperPara.DATASET



init_F.set_seed(hyperPara.seed)
data = init_F.train_data_gen(DATASET, BATCH_SIZE) # (BATCH_SIZE, 2) [采样样本, 实际分布对应的概率值]
data_plot = init_F.train_data_gen(DATASET, BATCH_SIZE_plot)
# print("data:", data, sep="\n")
# print("data.shape:", data.shape, sep="\n") # (3, 2)
# print("data_plot:", data_plot, sep="\n")
# print("data_plot.shape:", data_plot.shape, sep="\n") # (4, 2)
# exit(0)

task_type = "learning"
Learning_start_time = time.time()
if MODE == 'GAN' or MODE == 'WGAN' or MODE == 'GMM_GAN' or MODE == 'GMM_WGAN' or MODE == 'GMM_in_GAN' or MODE == 'GMM_in_WGAN':
    run_GAN.run(MODE, DATASET, data, data_plot, task_type, normal_type= False)
elif MODE == 'GMM':
    run_GMM.run(MODE, DATASET, data, data_plot, task_type)
elif MODE == 'KDE':
    run_KDE.run(MODE, DATASET, data, data_plot, task_type)
elif MODE == 'SVM':
    run_SVM.run(MODE, DATASET, data, data_plot, task_type)
elif MODE == 'WM':
    run_WM.run(MODE, DATASET, data, data_plot, task_type)
else:
    print("No Mode!")
    exit(0)
Learning_stop_time = time.time()
Learning_time = round(Learning_stop_time - Learning_start_time, 4)
print("Learning_time:", Learning_time, sep="\n") # (4, 2)





