import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score
# %%
# 导入XGB,RF,KNN,MLP这四个模型，导入训练集和测试集
XGB = load('model/XGBmodel.joblib')
RF = load('model/RFmodel.joblib')
KNN = load('model/KNNmodel.joblib')
MLP = load('model/MLPmodel.joblib')
x_train_scaler = pd.read_excel('test/x_train_scaler.xlsx')
y_train = pd.read_excel('test/y_train.xlsx')
x_test_scaler = pd.read_excel('test/x_test_scaler.xlsx')
y_test = pd.read_excel('test/y_test.xlsx')
weight1 = pd.read_excel('test/x_train.xlsx', usecols=['time_series'])
weight2 = pd.read_excel('test/x_test.xlsx', usecols=['time_series'])
x_train_scaler_weight = pd.concat([x_train_scaler, weight1], axis=1)
x_test_scaler_weight = pd.concat([x_test_scaler, weight2], axis=1)
x_train_scaler_weight.rename(columns={'time_series': 9}, inplace=True)
x_test_scaler_weight.rename(columns={'time_series': 9}, inplace=True)
# %%
# 重新训练这四个模型的时间权重
XGB = XGB.fit(x_train_scaler, y_train['Insurer'], sample_weight=weight1['time_series'])
RF = RF.fit(x_train_scaler, y_train['Insurer'], sample_weight=weight1['time_series'])
KNN = KNN.fit(x_train_scaler_weight, y_train['Insurer'])
MLP = MLP.fit(x_train_scaler_weight, y_train['Insurer'])
# %%
# 对这四个模型进行预测
XGB_pred = XGB.predict(x_test_scaler)
print(accuracy_score(y_test, XGB_pred))
RF_pred = RF.predict(x_test_scaler)
print(accuracy_score(y_test, RF_pred))
KNN_pred = KNN.predict(x_test_scaler_weight)
print(accuracy_score(y_test, KNN_pred))
MLP_pred = MLP.predict(x_test_scaler_weight)
print(accuracy_score(y_test, MLP_pred))