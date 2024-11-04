import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# %%
# 读取数据，合并数据集，并划分数据集
random_state = 42  #随机种子
file_path = 'data/3MergerAWM.xlsx'  #训练集数据(随机时间戳)
filetest_path = 'data/3Copy of Data.xlsx'  #测试机数据（有时间戳）
df_1 = pd.read_excel(file_path, usecols=lambda column: column in ['Age', 'Driving exp', 'NCD', 'Premium', 'Year'])
df_2 = pd.read_excel(filetest_path, usecols=lambda column: column in ['Age', 'Driving exp', 'NCD', 'Premium', 'Year'])
df = pd.concat([df_1, df_2], axis=0, ignore_index=True)
x = df.iloc[:, :-1]
y = df.iloc[:, -1:]
# %%
# 查看每一个特征的数字特征
# 均值，中位数，众数，方差，标准差，极大值，极小值，偏度，峰度
characteristic = ['mean', 'median', 'plural', 'variance', 'standard deviation', 'max', 'min', 'Skewness', 'Kurtosis']
statics = pd.DataFrame(columns=df.columns)
for feature in df.columns:
    list = []
    list.append(df[feature].mean())
    list.append(df[feature].median())
    list.append(df[feature].mode()[0])
    list.append(df[feature].var())
    list.append(df[feature].std())
    list.append(df[feature].max())
    list.append(df[feature].min())
    list.append(df[feature].skew())
    list.append(df[feature].kurt())
    statics[feature] = list
statics['index'] = characteristic
statics = statics.set_index('index')
# %%
# 计算相关系数
data_1 = pd.read_excel(file_path, usecols=lambda column: column not in ['Settlement date'])
data_2 = pd.read_excel(filetest_path, usecols=lambda column: column not in ['Settlement date'])
data = pd.concat([data_1, data_2], axis=0, ignore_index=True)
correlation_matrix = data.corr(method='pearson')
# %%
# 标准化，计算累计解释方差贡献度，确定主成分数量, 绘制Scree plot
thresholds = 0.95
data_1 = pd.read_excel(file_path, usecols=lambda column: column not in ['Settlement date', 'Insurer'])
data_2 = pd.read_excel(filetest_path, usecols=lambda column: column not in ['Settlement date', 'Insurer'])
data = pd.concat([data_1, data_2], axis=0, ignore_index=True)
scaler = StandardScaler()
data = scaler.fit_transform(data)
pca = PCA()
pca.fit(data)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
n_components = np.argmax(cumulative_variance_ratio >= thresholds) + 1
print('主成分的个数为', n_components)
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Variance')
plt.title('Scree Plot')
plt.grid(True)
plt.show()