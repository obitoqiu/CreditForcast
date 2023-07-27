import pandas as pd
from preprocessing import *
from model import lgbm_feature_importance


# 读取文件
train_xy = pd.read_csv("./competition/data/train_xy.csv", header=0, sep=",")
train_x = pd.read_csv("./competition/data/train_x.csv", header=0, sep=",")
test_all = pd.read_csv("./competition/data/test_all.csv", header=0, sep=",")

print(train_xy.shape)
print(train_x.shape)
print(test_all.shape)

# 合并train 和 test 作为待处理数据
train = train_xy.copy()
test = test_all.copy()
test['y'] = -1
data = pd.concat([train, test], axis=0)  # train_xy，test_all索引上连接
print(train.shape)
print(test.shape)
print(data.shape)

# 获取特征列,去除(id, group, y)3项特征
features = [feat for feat in train.columns.values if feat not in ['cust_id', 'cust_group', 'y']]
print("所有特征的维度：", len(features))

# 得到模型的输入(X, y)
train_id = train['cust_id']
y = train["y"].values
X = train[features].values
print("X shape:", X.shape)
print("y shape:", y.shape)

test_id = test['cust_id']
test = test[features].values
print("test shape", test.shape)

#preview_fill_default(data, 'x_88', 'rank')
# 输出lgbm模型训练中重要的参数
#lgbm_feature_importance(X, y, features)
