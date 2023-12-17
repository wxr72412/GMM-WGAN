# 读取包含训练样本的文本文件
with open('processed_bone_marrow.txt', 'r') as file:
    data = file.readlines()

# 分割每行数据并将其转换为列表
data = [line.strip().split(',') for line in data]
# print(data)
# exit(0)

# 现在，data是一个包含训练样本数据的列表，每个样本都是一个包含14个属性值的子列表。
# 创建一个数据框，其中每列代表一个属性
import pandas as pd
data_df = pd.DataFrame(data, columns=["Recipient_gender", "Stemcell_source", "IIIV",
                                      "Gender_match", "Donor_ABO", "Recipient_ABO", "Recipient_Rh", "ABO_match",
                                      "CMV_status", "Donor_CMV", "Recipient_CMV", "Disease", "Risk_group",
                                      "Txpost_relapse", "Disease_group", "HLA_match", "HLA_mismatch", "Antigen",
                                      "Allel", "HLAgrI", "Recipient_age", "Recipient_age_10", "Recipient_age_int",
                                      "Relapse", "aGvHD_III_IV", "extcGvHD", "CD34_kgx10d6", "CD3d_CD34",
                                      "CD3d_kgx10d8", "Rbody_mass", "survival_status"])
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
print(len(best_model.edges()))
# exit(0)

# 创建一个贝叶斯网络模型，这里的边需要根据您的实际网络结构定义
# ["trestbps", "chol", "fbs"])
edges = \
[('Stemcell_source', 'CD3d_CD34'), ('IIIV', 'extcGvHD'), ('Gender_match', 'Recipient_gender'), ('Donor_ABO', 'Risk_group'), ('Recipient_ABO', 'Risk_group'),
 ('CMV_status', 'Recipient_CMV'), ('Donor_CMV', 'CMV_status'), ('Recipient_CMV', 'Gender_match'), ('Disease', 'Disease_group'), ('ABO_match', 'Disease'),
 ('ABO_match', 'Relapse'), ('Txpost_relapse', 'ABO_match'), ('HLA_match', 'HLAgrI'), ('HLA_match', 'Allel'), ('HLA_match', 'HLA_mismatch'),
 ('HLA_mismatch', 'Donor_CMV'), ('Allel', 'Antigen'), ('Recipient_age_10', 'aGvHD_III_IV'), ('Recipient_age_int', 'Recipient_age_10'), ('Recipient_age_int', 'Recipient_age'),
 ('Recipient_age_int', 'CD34_kgx10d6'), ('Relapse', 'survival_status'), ('aGvHD_III_IV', 'IIIV'), ('CD34_kgx10d6', 'Stemcell_source'), ('Rbody_mass', 'Recipient_age_int')]

# "Recipient_Rh",
# "CD3d_kgx10d8",
edges.append(('Rbody_mass', 'CD3d_kgx10d8'))
edges.append(('CD3d_kgx10d8', 'CD3d_CD34'))

edges.append(('survival_status', 'CD3d_CD34'))
edges.append(('survival_status', 'Recipient_age_int'))

edges.append(('survival_status', 'HLA_match'))

edges.append(('Donor_ABO', 'ABO_match'))
edges.append(('Recipient_ABO', 'ABO_match'))
edges.append(('ABO_match', 'Recipient_Rh'))

model = BayesianNetwork(edges)
# model = BayesianNetwork(best_model.edges())
# 使用数据来学习参数
model.fit(data_df, estimator=MaximumLikelihoodEstimator)


from pgmpy.readwrite.BIF import BIFWriter
# 指定输出的文件名
output_file = 'bone.bif'
# 创建 BIFWriter 实例并将 BN 结构写入 BIF 文件
bif_writer = BIFWriter(model)
bif_writer.write_bif(output_file)

