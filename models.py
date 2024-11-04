import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
# %%
# 读取数据，划分数据集
random_state = 42
file_path = 'data/3MergerAWM.xlsx'  #数据(随机时间戳)
filetest_path = 'data/3Copy of Data.xlsx'  #数据（有时间戳）
data_1 = pd.read_excel(file_path, usecols=lambda column: column not in ['Settlement date', 'Insurer'])
data_2 = pd.read_excel(filetest_path, usecols=lambda column: column not in ['Settlement date', 'Insurer'])
x = pd.concat([data_1, data_2], axis=0, ignore_index=True)
data_1 = pd.read_excel(file_path, usecols=lambda column: column in ['Insurer'])
data_2 = pd.read_excel(filetest_path, usecols=lambda column: column in ['Insurer'])
y = pd.concat([data_1, data_2], axis=0, ignore_index=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
# %%
# 计算标准化后的训练集和测试集（在测试集上使用相同的均值和方差）
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)
# %%
# 计算归一化后的训练集和测试集（在测试集上使用相同的均值和反差）
MMscaler = MinMaxScaler()
x_train_MMscaler = MMscaler.fit_transform(x_train)
x_test_MMscaler = MMscaler.transform(x_test)
# %%
# 交叉验证并使用XGBoost进行建模


def XGBmodel(x, y, x_test, y_test, random_state):
    param_grid = {'max_depth': [3, 4, 5, 6, 7],
                  'n_estimators': [50, 100, 150, 200, 250]}
    xgb = XGBClassifier(objective='multi:softmax',n_jobs=-1, random_state=random_state)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x, y)
    print(grid_search.best_params_)
    best_xgb = grid_search.best_estimator_
    test_score = best_xgb.score(x_test, y_test)
    print('XGB Test set accuracy:', test_score)


XGBmodel(x_train_MMscaler, y_train['Insurer'], x_test_MMscaler, y_test['Insurer'], random_state)
# %%
# 交叉验证并使用RF建模


def RFmodel(x, y, x_test, y_test, random_state):
    param_grid = {'max_depth': [20, 21, 22, 23, 24],
                  'n_estimators': [200, 250, 300, 350, 400]}
    rf = RandomForestClassifier(n_jobs=-1, random_state=random_state)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, )
    grid_search.fit(x, y)
    print(grid_search.best_params_)
    best_rf = grid_search.best_estimator_
    test_score = best_rf.score(x_test, y_test)
    print('RF Test set accuracy:', test_score)


RFmodel(x_train_MMscaler, y_train['Insurer'], x_test_MMscaler, y_test['Insurer'], random_state)
# %%
# 交叉验证并使用GNB建模


def GNBmodel(x, y, x_test, y_test, random_state):
    gnb = GaussianNB()
    gnb.fit(x, y)
    test_score = gnb.score(x_test, y_test)
    print('GNB Test set accuracy:', test_score)


GNBmodel(x_train_MMscaler, y_train['Insurer'], x_test_MMscaler, y_test['Insurer'], random_state)
# %%
# 交叉验证并使用KNN建模


def KNNmodel(x, y, x_test, y_test, random_state):
    param_grid = {'n_neighbors': range(1, 31), 
                  'weights': ['uniform', 'distance']}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x, y)
    print(grid_search.best_params_)
    best_knn = grid_search.best_estimator_
    test_score = best_knn.score(x_test, y_test)
    print('KNN Test set accuracy:', test_score)


KNNmodel(x_train_MMscaler, y_train['Insurer'], x_test_MMscaler, y_test['Insurer'], random_state)
# %%
# 交叉验证并使用log_reg建模


def LOG_REG_model(x, y, x_test, y_test, random_state):
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    log_reg = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=random_state, n_jobs=-1)
    grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x, y)
    print(grid_search.best_params_)
    best_log_reg = grid_search.best_estimator_
    test_score = best_log_reg.score(x_test, y_test)
    print('LOG_REG Test set accuracy:', test_score)


LOG_REG_model(x_train_MMscaler, y_train['Insurer'], x_test_MMscaler, y_test['Insurer'], random_state)
# %%
# 交叉验证并使用ridge 建模


def RIDGEmodel(x, y, x_test, y_test, random_state):
    param_grid = {'alpha': [0.1, 1.0, 10.0],
                  'fit_intercept': [True, False],
                  'solver': ['auto', 'sag', 'lbfgs'], }
    ridge = RidgeClassifier(max_iter=1000, random_state=random_state)
    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x, y)
    print(grid_search.best_params_)
    best_ridge = grid_search.best_estimator_
    test_score = best_ridge.score(x_test, y_test)
    print('RIDGE Test set accuracy:', test_score)


RIDGEmodel(x_train_MMscaler, y_train['Insurer'], x_test_MMscaler, y_test['Insurer'], random_state)
# %%
# 交叉验证并使用MLP建模


def MLPmodel(x, y, x_test, y_test, random_state):
    param_grid = {'hidden_layer_sizes': [(10, 10, 10, 10), (5, 5, 5, 5), (3, 3, 3, 3)],
                  'activation': ['identity', 'logistic', 'tanh', 'relu']}
    mlp = MLPClassifier(max_iter=1000,learning_rate_init=0.001, random_state=random_state)
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x, y)
    print(grid_search.best_params_)
    best_mlp = grid_search.best_estimator_
    test_score = best_mlp.score(x_test, y_test)
    print('MLP Test set accuracy:', test_score)


MLPmodel(x_train_MMscaler, y_train['Insurer'], x_test_MMscaler, y_test['Insurer'], random_state)
# %%
# 交叉验证并使用SVC建模


def SVMmodel(x, y, x_test, y_test, random_state):
    param_grid = {'C': [0.1, 1, 10, 100],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    svm = SVC(random_state=random_state, max_iter=1000)
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x, y)
    print(grid_search.best_params_)
    best_svm = grid_search.best_estimator_
    test_score = best_svm.score(x_test, y_test)
    print('SVM Test set accuracy:', test_score)


SVMmodel(x_train_MMscaler, y_train['Insurer'], x_test_MMscaler, y_test['Insurer'], random_state)