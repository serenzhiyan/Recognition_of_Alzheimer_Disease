# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:52:32 2019

@author: Porthita
"""
import pandas as pd
#数据预处理

#导入数据
duration = pd.read_csv(r'C:\Users\Porthita\OneDrive\桌面\程序测试\duration.csv', encoding='utf-8')
linguistic = pd.read_csv(r'C:\Users\Porthita\OneDrive\桌面\程序测试\linguistic.csv', encoding='utf-8')
merged = pd.read_csv(r'C:\Users\Porthita\OneDrive\桌面\程序测试\merged.csv', encoding='utf-8')
egemaps_pre = pd.read_csv(r'C:\Users\Porthita\OneDrive\桌面\比赛\阿兹海默\data\egemaps_pre.csv', encoding='utf-8')

#查看数据的粗略情况
print(egemaps_pre.describe())

#处理缺失值
total = egemaps_pre.isnull().sum().sort_values(ascending=False)
print(total)

#查找负值
aa = linguistic[(linguistic['sum_B_correction']<0)].index.tolist()
bb = linguistic[(linguistic['sum_B_repeat']<0)].index.tolist()
aa = linguistic.loc[aa,'uuid']
bb = linguistic.loc[bb,'uuid']

#导入数据
one_egenmaps_csv = pd.read_csv(r'C:\Users\Porthita\OneDrive\桌面\比赛\阿兹海默\data\egemaps\P0001_0017.csv', sep = ';')
#确定每列有多少个零值
a = (one_egenmaps_csv == 0).sum(axis=0)

#获取列的名字
b = egenmaps.columns.tolist()









