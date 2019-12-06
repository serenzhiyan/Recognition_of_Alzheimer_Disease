# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 10:40:30 2019

@author: Porthita
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
#from  sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

def accuracy(predictions, Y):
    """ calculate accuracy """
    error = predictions - Y
    #print(error)
    acc = sum(error == 0) / len(error)
    return acc

def precision_recall_f1score(predictions, Y_true):
    """ calculate precision_recall_f1score
    二分类，建议使用F1分数作为最终评估标准
    """
    precision, recall, f1score, _ = precision_recall_fscore_support(Y_true, 
                                        predictions, pos_label=1, average='binary')
    return precision, recall, f1score

def get_XY(merged_features_fp, label_fp):
    """ 读取数据和标签并将其数值化，返回数值化的Numpy矩阵。
        参数：
            merged_features_fp：汇总后的特征文件路径
            label_fp：标签文件路径
        如果给定的文件名是None则对应返回None。
    """
    if merged_features_fp:
        data_train = pd.read_csv(merged_features_fp, encoding='utf-8', index_col=0)
        X = data_train.values
    else:
        X = None

    if label_fp:
        label_train = pd.read_csv(label_fp, encoding='utf-8', index_col=0)
        dummies_label = pd.get_dummies(label_train['label'], prefix='label')
        Y = dummies_label.label_AD.values
    else:
        Y = None
    return X, Y

def save_predict_result(test_predict):
    """ 保存预测结果
    参数:
        test_predict : 预测结果，一个Numpy数组
    """
    label_fp = r'C:\Users\Porthita\OneDrive\桌面\比赛\阿兹海默\data\1_preliminary_list_test.csv'
    Y_test_df = pd.read_csv(label_fp, encoding='utf-8', index_col=0)
    Y_test_df['pred_value'] = test_pred
    Y_test_df.loc[(Y_test_df.pred_value == 1), 'label'] = 'AD'
    Y_test_df.loc[(Y_test_df.pred_value == 0), 'label'] = 'CTRL'
    Y_test_df.drop(columns=['pred_value'], inplace=True)
    Y_test_df.to_csv(r'C:\Users\Porthita\OneDrive\桌面\程序测试\result5.csv')


merged_features_fp = r'C:\Users\Porthita\OneDrive\桌面\程序测试\txt_pca_train.csv'
label_fp = r'C:\Users\Porthita\OneDrive\桌面\比赛\阿兹海默\data\1_preliminary_list_train.csv'
X, Y = get_XY(merged_features_fp, label_fp)
scaler = StandardScaler()
X = scaler.fit_transform(X)
xx_train,xx_test,yy_train,yy_test = train_test_split(X,Y)
# =============================================================================
# merged_features_fp = r'C:\Users\Porthita\OneDrive\桌面\程序测试\feature_pca_test.csv'
# Xtest, _ = get_XY(merged_features_fp, None)
# =============================================================================
# =============================================================================
# #特征选择
# clf = ExtraTreesClassifier(n_estimators=100,criterion='gini',max_depth=4)
# clf.fit(XX,Y)
# index = np.flipud(np.argsort(clf.feature_importances_))
# score = clf.feature_importances_[index]
# fig,ax = plt.subplots(figsize=(20,8))
# plt.bar(range(len(index)),score,align='center')
# plt.xticks(range(len(index)),index)
# plt.title('importances of features')
# plt.show()
# sx = XX[:,index[0:10]]
# #分层抽样
# ss=StratifiedShuffleSplit(n_splits=5,test_size=0.25,train_size=0.75,random_state=5)#分成5组，测试比例为0.25，训练比例是0.75
# for train_index, test_index in ss.split(X, Y):
#     x_train, x_test = X[train_index], X[test_index]#训练集对应的值
#     y_train, y_test = Y[train_index], Y[test_index]#类别集对应的值
# 
# =============================================================================
model1 = LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10, 100], penalty='l2', solver='liblinear', cv=4)
model2 = svm.LinearSVC(tol=1, class_weight='balanced')     
model3 = LinearDiscriminantAnalysis()
model4 = DecisionTreeClassifier(criterion = 'entropy')

print (cross_val_score(model3,train_stack,y_train, cv=10, scoring='recall').mean())

model3.fit(x_train, y_train)
#print(model.score(x_test,y_test))
test_pred = model3.predict(x_test)
print('Accuracy:', accuracy(test_pred,y_test))
print('Precision: %f, Recall:%f, F1-score:%f' % precision_recall_f1score(test_pred,y_test))
test_pred = model.predict(Xtest)
save_predict_result(test_pred)

#stacking 模型
skf = StratifiedKFold(n_splits=5)
stack_model = [model1,model2,model3]  
ntrain = x_train.shape[0]  
ntest = x_test.shape[0]   ## 测试集样本数量
train_stack = np.zeros([ntrain,3]) ##  n表示n个模型
test_stack = np.zeros([ntest,3]) 
test = np.zeros([ntest,5])
for i,model in enumerate(stack_model):
    for j, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
        X_train, X_validate, label_train, label_validate = x_train[train_index, :], x_train[test_index, :], y_train[train_index], y_train[test_index]  
        model.fit(X_train,label_train)
        train_stack[test_index,i] = model.predict(X_validate)
        test[:,j] = model.predict(x_test)
    b = np.mean(test, axis=1)
    test_stack[:,i] = b
final_model  = model3  
final_model.fit(train_stack,y_train)
test_pred = final_model.predict(test_stack)
   
    
    
    
    
    
    
    
    
    
    
    
    