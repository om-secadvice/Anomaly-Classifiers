import numpy as np
import pandas as pd

train_atk=pd.read_csv('training_attack_types',dtype='object')
df_train=pd.read_csv('kddtrain.csv',dtype='object',header=None)
df_train_grouped=df_train.groupby(41)

df_test=pd.read_csv('kddtest.csv',dtype='object',header=None)
df_test_grouped=df_test.groupby(41)
ltest=list(df_test_grouped.groups.keys())
labels=train_atk['Name'].tolist()
labels.append('normal.')

df_test = pd.concat( [ df_test_grouped.get_group(group) for group in list(df_test_grouped.groups.keys()) if group in labels] )
df_test.to_csv('test.csv',header=None,index=False)
