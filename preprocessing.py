import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 绘图基本参数
config = {"font.family": "serif",
          "mathtext.fontset": "stix",
          "font.serif": ["SimSun"],
          "font.size": 10,
          "axes.unicode_minus": False
         }
plt.rcParams.update(config)


# 统计每条样本缺省的特征数
def count_nan_row(data):
    df = data.copy()
    df = df.replace(-99, np.nan)
    df['nan_count'] = df.shape[1] - df.count(axis=1).values  # 总列 - 非缺省值的列
    x = range(df.shape[0])
    y = df['nan_count']

    plt.scatter(x, y, s=0.7)
    plt.plot([15000, 15000], [30, 140], c='r', ls='--')
    plt.title('每个样本缺省的特征值数量')
    plt.xlabel('样本数')
    plt.ylabel('缺省特征数')

    plt.show()


# 对标签与特征缺省值的分布关系进行可视化
def label_nan_distribution(data):
    df = data.copy()
    df = df.replace(-99, np.nan)
    df['nan_count'] = df.shape[1] - df.count(axis=1).values  # 总列 - 非缺省值的列
    x1, x2, y1, y2 = [], [], [], []
    i = 1
    for index, row in df.iterrows():
        if row['y'] == 0:
            x1.append(i)
            y1.append(row['nan_count'])
            i += 1
        else:
            x2.append(i)
            y2.append(row['nan_count'])
            i += 1

    plt.scatter(x1, y1, s=1, c='b', marker='o')
    plt.scatter(x2, y2, s=1, c='r', marker='x')
    plt.xlim(0, df.shape[0]+200)
    plt.title('不同风险等级与特征缺省数量的分布')
    plt.xlabel('样本数')
    plt.ylabel('缺省特征数')
    plt.legend(['低风险', '高风险'], loc='upper right')

    plt.show()


# 统计每项特征的缺省率
def count_nan_column(data, features=None):
    if features is None:
        df = data.copy()
    else:
        df = data[features].copy()
    df = df.replace(-99, np.nan)
    plt.rcParams['font.size'] = 6
    count_list = df.count()
    default_ratio = (df.shape[0] - count_list) / df.shape[0]
    bins_df = pd.DataFrame({'default_ratio': default_ratio, 'feature_index': np.arange(1, len(default_ratio)+1)})

    plt.bar(np.arange(1, len(default_ratio)+1), bins_df['default_ratio'].values, width=0.7)
    plt.xticks(bins_df['feature_index'].values, rotation=90, fontsize=2.5, fontweight='bold')
    plt.title('每项特征的缺省率')
    plt.xlabel('特征下标')
    plt.ylabel('缺省率')
    plt.savefig('img/每项特征的缺省率', dpi=1500)

    plt.show()


# 缺省特征填充效果预览(measure参数值为空则仅展示密度图)
def preview_fill_default(data, feature, measure='mean'):
    data[feature].plot(kind='kde')
    plt.show()
    if measure is not None:
        if measure == 'mean':
            df = data[feature].replace(-99, np.nan)
            df.fillna(df.mean()).plot(kind='kde')

        elif measure == 'rank':
            df = data[feature].rank() / data.shape[0]
            df.plot(kind='kde')

        plt.show()


# 针对类别型变量绘制交叉表
def preview_crosstab(data, feature1, feature2):
    pd.crosstab(data[feature1], data[feature2]).plot(kind='bar')
    plt.show()


# 统计每行缺省的特征数量作为一项新的特征'nan_count'
def get_nan_count(data, feats, bins=5):
    df = data[feats].copy()
    df = df.replace(-99, np.nan)
    df['nan_count'] = df.shape[1] - df.count(axis=1).values
    # 对特征‘nan_count’分层(默认等分为5层)+离散化后合并至源数据
    dummy = pd.get_dummies(pd.cut(df['nan_count'], bins), prefix='nan')
    print('dummy shape: ', dummy.shape)
    res = pd.concat([data, dummy], axis=1)
    print(res.shape)
    return res

