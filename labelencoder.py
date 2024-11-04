import json
import pandas as pd
columns = ['Occupation', 'Coverage', 'Insurer', 'Make', 'Model']
df1 = pd.read_excel('data/2MergerAWM.xlsx', usecols=columns)
df2 = pd.read_excel('data/2Copy of Data.xlsx', usecols=columns)
for col in columns:
    dict = {}
    for df in [df1, df2]:
        for value in df[col].unique():
            dict.setdefault(value, len(dict))
    with open('encoder/'+col+'.json', 'w', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)
    print(len(dict))