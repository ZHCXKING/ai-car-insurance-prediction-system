import pandas as pd
import numpy as np
import json
# %%
# 查看数据情况，确认是否有需要的列
filepath = 'data/Copy of Data.xlsx'
columns = ['Settlement date', 'Age', 'Driving exp', '職業', 'Model',
           'Coverage', '保險公司', 'Premium', 'Make', 'Year', 'NCD']


def infomation():
    df = pd.read_excel(filepath, usecols=columns, sheet_name='Sheet1')
    df.rename(columns={'職業': 'Occupation', '保險公司': 'Insurer'}, inplace=True)
    print(df.info())
    print(df.duplicated())
    print("缺失值统计：", df.isnull().sum())
    print("重复值统计：", df.duplicated().sum())
    return df


df = infomation()
# %%
# 数据获取合并，并直接删除缺失值和重复值


def load_drop(df):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


df = load_drop(df)
# %%
# 读取数据集，预处理数据集


def cleanned(df):
    #处理年龄
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').replace({np.nan: 70})
    #处理驾龄
    df['Driving exp'] = df['Driving exp'].replace({'多於 10 年': 11, '暫准駕駛執照 ( P牌 ）': 0, '>':11})
    df['Driving exp'] = df['Driving exp'].astype(str)
    df['Driving exp'] = df['Driving exp'].str.extract(r'(\d+)').astype(int)
    #处理职业
    df['Occupation'] = df['Occupation'].str.split('|').str[0].str.strip()
    #处理车辆年限
    df['Year'] = df['Year'].astype(str).replace({'1980 或之前': 1980}).astype(int)
    #处理保单日期
    df['Settlement date'] = df['Settlement date'].dt.date
    #处理车辆型号
    df['Model'] = df['Model'].astype(str).str.lower().str.strip()
    #处理汽车使用年限
    df['Year'] = 2024 - df['Year']
    return df


df = cleanned(df)
# %%
# 保存数据
cols = ['Settlement date', 'Age', 'Driving exp', 'Occupation', 'Coverage', 'Model',
        'NCD', 'Premium', 'Make', 'Year', 'Insurer']
df = df[cols]
df.to_excel('data/2Copy of Data.xlsx', sheet_name='Sheet1', index=False)
# %%
# 使用保存好的模型进行编码
cols = ['Occupation', 'Coverage', 'Insurer', 'Make', 'Model']
for col in cols:
    with open('encoder/'+col+'.json', 'r', encoding='utf-8') as f:
        dict = json.load(f)
    df[col] = df[col].map(dict)
df.to_excel('data/3Copy of Data.xlsx', sheet_name='Sheet1', index=False)