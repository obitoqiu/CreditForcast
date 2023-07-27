import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import lightgbm as lgbm


# 训练lgbm模型
def lgbm_train(X, y):
    # 均匀样本分K折抽样
    K = 5
    auc_cv = []
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=2018)
    for train_in, test_in in skf.split(X, y):
        X_train, X_test, y_train, y_test = X[train_in], X[test_in], y[train_in], y[test_in]
        train_data = lgbm.Dataset(X_train, y_train)
        eval_data = lgbm.Dataset(X_test, y_test, reference=train_data)
        # 参数设置
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'learning_rate': 0.02,
            'num_leaves': 16,
            'max_depth': 4,
            'min_child_weight': 6,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'metric': {'auc'},
            'verbose': -1,
        }
        # 模型训练
        start = time.time()
        print('start training...')
        model = lgbm.train(params,
                           train_data,
                           num_boost_round=2000,
                           valid_sets=eval_data,
                           early_stopping_rounds=100,
                           verbose_eval=100,)

        # 模型预测
        print('start predicting...')
        y_pred = model.predict(X_test)
        # 模型评估
        tmp_auc = roc_auc_score(y_test, y_pred)
        auc_cv.append(tmp_auc)
        print('valid auc: ', tmp_auc)

    # 交叉验证平均分数
    mean_auc = round(np.mean(auc_cv), 5)
    print('auc cv: ')
    print(auc_cv)
    print('mean auc: ', mean_auc)
    filepath = 'model/lgbm_' + str(mean_auc) + '.txt'
    end = time.time()
    print('run with time: ', (end - start) / 60)
    # 模型保存
    model.save_model(filepath)
    print('model saved in ', filepath)


# 模型预测
def model_predict(model_file, test_data, test_id):
    model = lgbm.Booster(model_file=model_file)
    pred = model.predict(test_data)
    result = pd.DataFrame()
    result['cust_id'] = test_id
    result['pred_prob'] = pred
    filepath = model_file.replace('model/', 'result/').replace('.txt', '.csv')
    result.to_csv(filepath, sep=',', index=False)
    print('predict result saved in ', filepath)


# 输出lgbm模型训练中重要的参数
def lgbm_feature_importance(train_x, train_y, features=None):
    train_data = lgbm.Dataset(train_x, train_y)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'learning_rate': 0.02,
        'num_leaves': 16,
        'max_depth': 4,
        'min_child_weight': 6,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'metric': {'auc'},
        'verbose': -1,
        # 'lambda_l1':0.25,
        # 'lambda_l2':0.5,
        # 'scale_pos_weight':1,
    }
    print('start...')
    start = time.time()
    print('start training...')
    model = lgbm.train(params,
                       train_data,
                       num_boost_round=450,
                       valid_sets=train_data,
                       early_stopping_rounds=100,
                       verbose_eval=100,)
    end = time.time()
    print('over training...')
    print('run with time: ', (end - start) / 60)
    # 显示前30维特征
    lgbm.plot_importance(model, max_num_features=30, figsize=(20, 10))
    plt.show()

    if features is None:
        features = np.arange(1, train_x.shape[1] + 1)

    # 将特征重要程度与特征名关联并按重要程度降序输出
    df = pd.DataFrame({'feature': features, 'importance': model.feature_importance()}).sort_values(by='importance',
                                                                                                   ascending=False)
    useful_feature = df.loc[df['importance'] != 0, 'feature'].tolist()
    print('useful features size:', len(useful_feature))
    print('useful features top30: ', useful_feature[:30])
