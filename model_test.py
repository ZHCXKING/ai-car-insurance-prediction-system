import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from joblib import dump
# %%
# 读取数据，划分数据集
random_state = 42
x = pd.read_excel("data/AWMencoder.xlsx",
                  usecols=lambda column: column != 'Insurer')
y = pd.read_excel("data/AWMencoder.xlsx",
                  usecols=lambda column: column == 'Insurer')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
x_train.to_excel('test/x_train.xlsx', index=False)
y_train.to_excel('test/y_train.xlsx', index=False)
x_test.to_excel('test/x_test.xlsx', index=False)
y_test.to_excel('test/y_test.xlsx', index=False)
# %%
# 去掉时间权重序列
x_train.drop(['time_series'], axis=1, inplace=True)
x_test.drop(['time_series'], axis=1, inplace=True)
# %%
# 计算标准化后的训练集和测试集（在测试集上使用相同的均值和方差）
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)
pd.DataFrame(x_train_scaler).to_excel('test/x_train_scaler.xlsx', index=False)
pd.DataFrame(x_test_scaler).to_excel('test/x_test_scaler.xlsx', index=False)
# %%
# 计算归一化后的训练集和测试集（在测试集上使用相同的均值和反差）
MMscaler = MinMaxScaler()
x_train_MMscaler = MMscaler.fit_transform(x_train)
x_test_MMscaler = MMscaler.transform(x_test)
pd.DataFrame(x_train_MMscaler).to_excel('test/x_train_MMscaler.xlsx', index=False)
pd.DataFrame(x_test_MMscaler).to_excel('test/x_test_MMscaler.xlsx', index=False)
# %%
# 计算方差膨胀因子VIF，判断数据是否线性结构


def VIFModel(x, y):
    data = pd.concat([x, y], axis=1)
    data = add_constant(data)
    vif_data = []
    for i in range(data.shape[1]):
        vif = variance_inflation_factor(data.values, i)
        vif_data.append({'Variable': data.columns[i], 'VIF': vif})
    vif_df = pd.DataFrame(vif_data)
    return vif_df


vif_df = VIFModel(x_train, y_train)
# %%
# 计算累计解释方差贡献度，确定主成分数量, 绘制Scree plot


def explainVal(data, thresholds):
    pca = PCA()
    pca.fit(data)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative_variance_ratio >= thresholds) + 1
    print('主成分的个数为', n_components)
    plt.plot(range(1, len(explained_variance_ratio) + 1),
             explained_variance_ratio, marker='o')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance')
    plt.title('Scree Plot')
    plt.grid(True)
    plt.show()
    return n_components


# n_component = explainVal(x_train_scaler, 0.95)  # 控制输入的数据是否标准化，还有阈值
# %%
# 交叉验证并使用XGBoost进行建模


def XGBmodel(x, y, x_test, y_test, random_state):
    param_grid = {'max_depth': [3, 4, 5, 6, 7],
                  'n_estimators': [50, 100, 150, 200, 250]}
    xgb = XGBClassifier(objective='multi:softmax',
                        n_jobs=-1, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=xgb, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x, y)
    print(grid_search.best_params_)
    best_xgb = grid_search.best_estimator_
    test_score = best_xgb.score(x_test, y_test)
    print('XGB Test set accuracy:', test_score)
    dump(best_xgb, 'model/XGBmodel.joblib')


XGBmodel(x_train_scaler, y_train, x_test_scaler, y_test, random_state)
# %%
# 交叉验证并使用RF建模


def RFmodel(x, y, x_test, y_test, random_state):
    param_grid = {'max_depth': [20, 21, 22, 23, 24],
                  'n_estimators': [200, 250, 300, 350, 400]}
    rf = RandomForestClassifier(n_jobs=-1, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(x, y)
    print(grid_search.best_params_)
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('RF Test set accuracy:', accuracy)
    dump(best_rf, 'model/RFmodel.joblib')


RFmodel(x_train_scaler, y_train['Insurer'],
        x_test_scaler, y_test['Insurer'], random_state)
# %%
# 交叉验证并使用GNB建模


def GNBmodel(x, y, x_test, y_test):
    gnb = GaussianNB()
    gnb.fit(x, y)
    y_pred = gnb.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('GNB Test set accuracy:', accuracy)
    dump(gnb, 'model/GNBmodel.joblib')


# GNBmodel(x_train, y_train['Insurer'], x_test, y_test['Insurer'])
# %%
# 交叉验证并使用KNN建模


def KNNmodel(x, y, x_test, y_test, random_state):
    param_grid = {'n_neighbors': range(1, 31), 
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(
        estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x, y)
    print(grid_search.best_params_)
    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('KNN Test set accuracy:', accuracy)
    dump(best_knn, 'model/KNNmodel.joblib')


KNNmodel(x_train_scaler, y_train['Insurer'],
         x_test_scaler, y_test['Insurer'], random_state)
# %%
# 交叉验证并使用log_reg建模


def LOG_REG_model(x, y, x_test, y_test, random_state):
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    log_reg = LogisticRegression(
        multi_class='multinomial', max_iter=100000, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=log_reg, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x, y)
    print(grid_search.best_params_)
    best_log_reg = grid_search.best_estimator_
    y_pred = best_log_reg.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('LOG_REG Test set accuracy:', accuracy)
    dump(best_log_reg, 'model/LOG_REGmodel.joblib')


# LOG_REG_model(x_train, y_train['Insurer'], x_test, y_test['Insurer'], random_state)
# %%
# 交叉验证并使用ridge 建模


def RIDGEmodel(x, y, x_test, y_test, random_state):
    param_grid = {'alpha': [0.1, 1.0, 10.0],
                  'fit_intercept': [True, False],
                  'solver': ['auto', 'sag', 'lbfgs'], }
    ridge = RidgeClassifier(max_iter=100000, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=ridge, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x, y)
    print(grid_search.best_params_)
    best_ridge = grid_search.best_estimator_
    y_pred = best_ridge.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('RIDGE Test set accuracy:', accuracy)
    dump(best_ridge, 'model/RIDGE_REGmodel.joblib')


# RIDGEmodel(x_train, y_train['Insurer'], x_test, y_test['Insurer'], random_state)
# %%
# 交叉验证并使用MLP建模


def MLPmodel(x, y, x_test, y_test, random_state):
    param_grid = {'hidden_layer_sizes': [(10, 10, 10, 10), (5, 5, 5, 5), (3, 3, 3, 3)],
                  'activation': ['identity', 'logistic', 'tanh', 'relu']}
    mlp = MLPClassifier(max_iter=10000, alpha=0.001, beta_1=0.7,
                        beta_2=0.7, learning_rate_init=0.001, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=mlp, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(x, y)
    print(grid_search.best_params_)
    best_mlp = grid_search.best_estimator_
    y_pred = best_mlp.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('MLP Test set accuracy:', accuracy)
    dump(best_mlp, 'model/MLPmodel.joblib')


MLPmodel(x_train_scaler, y_train['Insurer'],
         x_test_scaler, y_test['Insurer'], random_state)
# %%
# 交叉验证并使用GPC建模


def GPCmodel(x, y, x_test, y_test, random_state):
    param_grid = {'multi_class': ['one_vs_rest', 'one_vs_one']}
    gpc = GaussianProcessClassifier(
        max_iter_predict=100000, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=gpc, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(x, y)
    print(grid_search.best_params_)
    best_gpc = grid_search.best_estimator_
    y_pred = best_gpc.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('GPC Test set accuracy:', accuracy)
    dump(best_gpc, 'model/GPCmodel.joblib')


# GPCmodel(x_train, y_train['Insurer'], x_test, y_test['Insurer'], random_state)
# %%
# 交叉验证并使用SVC建模


def SVMmodel(x, y, x_test, y_test, random_state):
    param_grid = {'C': [0.1, 1, 10, 100],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    svm = SVC(random_state=random_state, max_iter=100000)
    grid_search = GridSearchCV(
        estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x, y)
    print(grid_search.best_params_)
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('SVM Test set accuracy:', accuracy)
    dump(best_svm, 'model/SVMmodel.joblib')


# SVMmodel(x_train_MMscaler, y_train['Insurer'], x_test_MMscaler, y_test['Insurer'], random_state)
