# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 21:43:22 2019

@author: Porthita
"""

def merge_common(filepath_list, out_file_path):
    ''' 合并特征数据，根据uuid（是文件名也是被试的标识名）将多个特征文件合并
    类似于数据库表的JOIN操作
    参数:
        filepath_list : 所有需要合并的特征文件路径列表
        out_file_path : 合并后的特征文件路径
    '''
    data_all = pd.read_csv(filepath_list[0], encoding='utf-8')
    for fp in filepath_list[1:]:
        df = pd.read_csv(fp,  encoding='utf-8')
        data_all = pd.merge(data_all, df, how='inner', on='uuid')
    data_all.to_csv(out_file_path, index=False)



# 提取train数据集的特征
label_fp = r'C:\Users\Porthita\OneDrive\桌面\比赛\阿兹海默\data\1_preliminary_list_train.csv'
df = pd.read_csv(label_fp)
name_list = df.uuid
duration_fp = r'C:\Users\Porthita\OneDrive\桌面\程序测试\duration.csv'
extract_duration_features_main(name_list, duration_fp, "C:/Users/Porthita/OneDrive/桌面/比赛/阿兹海默/data/tsv/")
linguistic_fp = r'C:\Users\Porthita\OneDrive\桌面\程序测试\linguistic.csv'
extract_linguistic_features_main(name_list, linguistic_fp, "C:/Users/Porthita/OneDrive/桌面/比赛/阿兹海默/data/tsv/")
egemaps_fp = r'C:\Users\Porthita\OneDrive\桌面\程序测试\egemaps_pca.csv'
merged_features_fp =  r'C:\Users\Porthita\OneDrive\桌面\程序测试\merged.csv'
merge_common([duration_fp, linguistic_fp, egemaps_fp], merged_features_fp)
## 提取test数据集的特征
#label_fp = r'C:\Users\Porthita\OneDrive\桌面\比赛\阿兹海默\data\1_preliminary_list_test.csv'
#df = pd.read_csv(label_fp)
#name_list = df.uuid
#duration_fp =r'C:\Users\Porthita\OneDrive\桌面\程序测试\duration_test.csv'
#extract_duration_features_main(name_list, duration_fp, "C:/Users/Porthita/OneDrive/桌面/比赛/阿兹海默/data/tsv/")
#linguistic_fp = r'C:\Users\Porthita\OneDrive\桌面\程序测试\linguistic_test.csv'
#extract_linguistic_features_main(name_list, linguistic_fp, "C:/Users/Porthita/OneDrive/桌面/比赛/阿兹海默/data/tsv/")
#egemaps_fp =  r'C:\Users\Porthita\OneDrive\桌面\程序测试\egemaps_pca.csv'
#merged_features_fp =  r'C:\Users\Porthita\OneDrive\桌面\程序测试\merged_test.csv'
#merge_common([duration_fp, linguistic_fp, egemaps_fp], merged_features_fp)

