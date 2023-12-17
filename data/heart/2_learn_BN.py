# 读取包含训练样本的文本文件
with open('heart.txt', 'r') as file:
    data = file.readlines()

# 分割每行数据并将其转换为列表
data = [line.strip().split(',') for line in data]
# print(data)
# exit(0)

# 现在，data是一个包含训练样本数据的列表，每个样本都是一个包含14个属性值的子列表。
# 创建一个数据框，其中每列代表一个属性
import pandas as pd
data_df = pd.DataFrame(data, columns=["age", "sex", "cp", "trestbps", "chol",
                                      "fbs", "restecg", "thalach", "exang", "oldpeak",
                                      "slope", "ca", "thal", "num"])
# print(data_df)
# exit(0)





from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator, BicScore
from pgmpy.models import BayesianNetwork
# 创建一个空的 BayesianNetwork 实例
# model = BayesianNetwork()
# 使用 HillClimbSearch 和 K2Score 来学习 BN 结构
hc = HillClimbSearch(data_df)
best_model = hc.estimate(scoring_method='bicscore')

# 输出学习到的 BN 结构
print(best_model.edges())
# exit(0)

# 创建一个贝叶斯网络模型，这里的边需要根据您的实际网络结构定义
# ["trestbps", "chol", "fbs"])


edges = [('sex', 'thal'), ('exang', 'cp'), ('exang', 'slope'), ('slope', 'oldpeak'), ('thal', 'num'), ('num', 'exang'), ('num', 'ca')]
edges.append(('age', 'thal'))
edges.append(('restecg', 'thalach'))
edges.append(('num', 'thalach'))
edges.append(('chol', 'fbs'))
edges.append(('trestbps', 'chol'))
edges.append(('cp', 'trestbps'))
model = BayesianNetwork(edges)
# model = BayesianNetwork(best_model.edges())
# 使用数据来学习参数
model.fit(data_df, estimator=MaximumLikelihoodEstimator)


from pgmpy.readwrite.BIF import BIFWriter
# 指定输出的文件名
output_file = 'heart.bif'
# 创建 BIFWriter 实例并将 BN 结构写入 BIF 文件
bif_writer = BIFWriter(model)
bif_writer.write_bif(output_file)

