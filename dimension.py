import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
 
# =============================================================================
# feature = pd.read_csv(r'C:\Users\Porthita\OneDrive\桌面\程序测试\feature.csv',encoding='utf-8', index_col=0)
# train_id = pd.read_csv(r'C:\Users\Porthita\OneDrive\桌面\比赛\阿兹海默\data\1_preliminary_list_train.csv',encoding='utf-8')
# test_id = pd.read_csv(r'C:\Users\Porthita\OneDrive\桌面\比赛\阿兹海默\data\1_preliminary_list_test.csv',encoding='utf-8')
# feature_pca_train.to_csv(r'C:\Users\Porthita\OneDrive\桌面\程序测试\txt_pca_train.csv', index=False)
# feature_pca_test.to_csv(r'C:\Users\Porthita\OneDrive\桌面\程序测试\txt_pca_test.csv', index=False)
# 
# =============================================================================
def dimension_of_pca(feature_path,train_id_list,test_id_list,out_put_path):
    scaler = StandardScaler()
    df = pd.read_csv(feature_path,encoding='utf-8', index_col=0)
    fea = df.values
    uuid = df.index
    train_id = pd.read_csv(train_id_list,encoding='utf-8')
    dummies_label = pd.get_dummies(train_id['label'], prefix='label')
    Y = dummies_label.label_AD.values
    test_id = pd.read_csv(test_id_list,encoding='utf-8')
    train_id = train_id.drop(['label'],axis=1)
    df_id = pd.DataFrame({'uuid':uuid,
                           })
    n_column = df.shape[1]
    score = np.zeros(n_column)
    for i in range(1,n_column+1):
        if i<n_column:
            fea_pca = PCA(n_components=i).fit_transform(fea)
        else: fea_pca = fea
        fea_pca = pd.DataFrame(fea_pca)
        fea_pca = pd.merge(df_id, fea_pca,left_index=True,right_index=True)
        fea_pca_train = pd.merge(fea_pca,train_id,how = 'inner',on = 'uuid')
        fea_pca_train = fea_pca_train.drop(['uuid'],axis=1)
        X = fea_pca_train.values
        X = scaler.fit_transform(X)
        model = svm.LinearSVC(tol=1, class_weight='balanced')     
        score[i-1] = cross_val_score(model,X,Y, cv=10, scoring='recall').mean()
        print('维数',i)
        print(score[i-1])
    
    score = score.tolist()
    j = score.index(max(score))
    print('最佳维数：',j+1)
    fea_pca = PCA(n_components=j+1).fit_transform(fea)
    fea_pca = pd.DataFrame(fea_pca)
    fea_pca = pd.merge(df_id, fea_pca,left_index=True,right_index=True)
    fea_pca.to_csv(out_put_path, index=False)
# =============================================================================
#     fea_pca_train = pd.merge(fea_pca,train_id,how = 'inner',on = 'uuid')
#     fea_pca_train.to_csv(train_out_put_path, index=False)
#     fea_pca_test = pd.merge(fea_pca,test_id,how = 'inner',on = 'uuid')
#     fea_pca_test = fea_pca_test.drop(['label'],axis=1)
#     fea_pca_test.to_csv(test_out_put_path, index=False)
# =============================================================================
    

        



