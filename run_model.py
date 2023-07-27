from preprocessing import *
from model import *


# 读取文件
train_xy = pd.read_csv("./competition/data/train_xy.csv", header=0, sep=",")
test_all = pd.read_csv("./competition/data/test_all.csv", header=0, sep=",")
print(train_xy.shape)
print(test_all.shape)

# 合并train 和 test 作为待处理数据
train = train_xy.copy()
test = test_all.copy()
test['y'] = -1
data = pd.concat([train, test], axis=0)  # train_xy，test_all索引上连接
print(train.shape)
print(test.shape)
print(data.shape)

# 特征变量x1-x95是数值型变量，x96-x157是类别型变量
numerical_features = []
categorical_features = []
for i in range(0, 157):
    if i < 95:
        numerical_features.append(train.columns[i+3])
    else:
        categorical_features.append(train.columns[i+3])
print("numerical features: ", len(numerical_features))
print("category features: ", len(categorical_features))

# 统计每条样本缺省的特征数量作为新的特征 (信用可能和某些特征的缺省、缺省特征的数量水平有关)
data = get_nan_count(data, data.columns.values, 7)

# top30重要的特征
imp_feat = ['x_80', 'x_2', 'x_81', 'x_95', 'x_52',
            'x_1', 'x_48', 'x_93', 'x_40', 'x_54',
            'x_63', 'x_45', 'x_55', 'x_78', 'x_50',
            'x_59', 'x_43', 'x_62', 'x_47', 'x_69',
            'x_19', 'x_61', 'x_157', 'x_42', 'x_53',
            'x_44', 'x_97', 'x_51', 'x_29', 'x_30']

print("important features:", len(imp_feat))

# 对部分重要的特征进行填充(Top K = 5, 10, 15, 20, 25...)
for feat in imp_feat[:10]:
    if feat in numerical_features:
        data[feat] = data[feat].replace(-99, np.nan)
        data[feat] = data[feat].fillna(data[feat].mean())  # 均值填充比众数稍好
    if feat in categorical_features:    # 类别特征不作任何处理
        print("这是类别特征：", feat)

# 把数值型的特征处理为rank特征
for feat in numerical_features:
    data[feat] = data[feat].rank() / float(data.shape[0])  # 数值特征转化成排序特征并归一化

# 划分train test set
train = data.loc[data['y'] != -1, :]  # train set
test = data.loc[data['y'] == -1, :]  # test set
print(train.shape)
print(test.shape)

# 获取特征列,去除(id, group, y)3项特征
no_features = ['cust_id', 'cust_group', 'y']
features = [feat for feat in train.columns.values if feat not in no_features]
print("所有特征的维度：", len(features))

# 得到模型的输入(X, y)
train_id = train['cust_id'].values
y = train['y'].values.astype(int)
X = train[features].values
print("X shape:", X.shape)
print("y shape:", y.shape)

test_id = test['cust_id'].values
test_data = test[features].values
print("test shape", test_data.shape)

# 训练模型
lgbm_train(X, y)

# 提取训练完毕的模型进行预测
# model_file = 'model/lgbm_0.81814.txt'
# model_predict(model_file, test_data, test_id)
