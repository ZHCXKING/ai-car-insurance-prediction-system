import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
# %%
#设置基本参数
random_state = 42  #随机种子
file_path = 'data/3MergerAWM.xlsx'  #训练集数据(随机时间戳)
filetest_path = 'data/3Copy of Data.xlsx'  #测试机数据（有时间戳）
# %%
x_train = pd.read_excel(file_path, usecols=lambda column: column not in ['Insurer', 'Settlement date', 'Model'])
y_train = pd.read_excel(file_path, usecols=lambda column: column == 'Insurer')
x_test = pd.read_excel(filetest_path, usecols=lambda column: column not in ['Insurer', 'Settlement date', 'Model'])
y_test = pd.read_excel(filetest_path, usecols=lambda column: column == 'Insurer')
# %%
# 交叉验证并使用XGBoost进行建模


def XGBmodel(x, y, random_state):
    param_grid = {'max_depth': [3, 4, 5, 6, 7],
                  'n_estimators': [50, 100, 150, 200, 250]}
    xgb = XGBClassifier(objective='multi:softmax', n_jobs=-1, random_state=random_state)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x, y)
    print(grid_search.best_params_)
    best_xgb = grid_search.best_estimator_
    dump(best_xgb, 'model/XGBmodel.joblib')


#XGBmodel(x_train, y_train['Insurer'], random_state)
# %%
# 交叉验证并使用RF建模


def RFmodel(x, y, random_state):
    param_grid = {'max_depth': [20, 21, 22, 23, 24],
                  'n_estimators': [150, 200, 250, 300, 350]}
    rf = RandomForestClassifier(n_jobs=-1, random_state=random_state)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x, y)
    print(grid_search.best_params_)
    best_rf = grid_search.best_estimator_
    dump(best_rf, 'model/RFmodel.joblib')


#RFmodel(x_train, y_train['Insurer'], random_state)
# %%
# 使用测试集
XGB = load('model/XGBmodel.joblib')
RF = load('model/RFmodel.joblib')
XGB = XGB.fit(x_train, y_train['Insurer'])
RF = RF.fit(x_train, y_train['Insurer'])
print('The accuracy:', XGB.score(x_test, y_test))
print('The accuracy:', RF.score(x_test, y_test))
# %%
# 使用votting投票分类器
voting = VotingClassifier(estimators=[('xgb', XGB), ('rf', RF)], voting='soft', n_jobs=-1)
voting = voting.fit(x_train, y_train['Insurer'])
y_pred = voting.predict(x_test)
print(accuracy_score(y_test, y_pred))
