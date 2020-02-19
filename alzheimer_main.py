# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 21:22:02 2020

@author: 
"""

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

##################### 首先进行特征的提取 ############################

label_fp = r'C:\Users\...\data\egemaps_pre.csv'
df = pd.read_csv(label_fp)
name_list = df.uuid #所有的受试对象的ID列表
dir_path = "C:/Users/.../data/tsv/"


#提取所有对象的对话时间特征
duration_fp = r'C:\Users\...\duration.csv'
extract_duration_features_main(name_list, duration_fp, dir_path) 
#提取所有对象的文本语言特征
linguistic_fp = r'C:\Users\...\linguistic.csv'
extract_linguistic_features_main(name_list, linguistic_fp, dir_path) 
#提取所有对象的降维后的声学特征
egemaps_fp = r'C:\Users\...\data\egemaps_pre.csv'
train_id_list = r'C:\Users\...\data\1_preliminary_list_train.csv'
egemaps_pca_fp = r'C:\Users\...\egemaps_pca.csv'
dimension_of_pca(egemaps_fp,train_id_list,egemaps_pca_fp) 
#提取所有对象的自定义相似度特征
stopwords_fp = r'C:\Users\...\哈工大停用词表.txt'
out_filepath_AD = r'C:\Users\...\AD(停词表).txt'
get_samelabel_txt(stopwords_fp,train_id_list,AD,out_filepath_AD,dir_path)
out_filepath_CTRL = r'C:\Users\...\CTRL(停词表).txt'
get_samelabel_txt(stopwords_fp,train_id_list,CTRL,out_filepath_CTRL,dir_path)
tag1 = tags_extract(out_filepath_AD)
tag0 = tags_extract(out_filepath_CTRL)
#可以手动取出tag1和tag0中的共有成分和常见成分，然后添加到哈工大停用词表构成新停用词表
stopwords_fp = r'C:\Users\...\新停用词表.txt'
out_filepath_AD = r'C:\Users\...\AD(新停词表).txt'
get_samelabel_txt(stopwords_fp,train_id_list,AD,out_filepath_AD,dir_path)
out_filepath_CTRL = r'C:\Users\...\CTRL(新停词表).txt'
get_samelabel_txt(stopwords_fp,train_id_list,CTRL,out_filepath_CTRL,dir_path)
out_filepath = r'C:\Users\...\临时存储某个特定对象的谈话内容.txt'
similarity_def = r'C:\Users\...\similarity_def.csv'
comparation(out_filepath_AD,out_filepath_CTRL,egemaps_fp,out_filepath,similarity_def,dir_path)
#合并所有特征
merged_features_fp =  r'C:\Users\...\merged.csv'
merge = merge_common([duration_fp, linguistic_fp, egemaps_pca_fp,similarity_def], merged_features_fp)
#新加的两个特征
merge['ave_A_word'] = merge.sum_A_word/merge.A_speak_num
merge['ave_B_word'] = merge.sum_B_word/merge.B_speak_num

#分出训练集和测试集（比赛时给出的已知标签的数据和未知标签的数据）
train_id = pd.read_csv(r'C:\Users\...\data\1_preliminary_list_train.csv',encoding='utf-8')
train_id = train_id.drop(['label'],axis=1)
merge_train = pd.merge(train_id, merge, how='inner', on='uuid')
merge_train.to_csv(r'C:\Users\...\merge_train.csv', index=False)
test_id = pd.read_csv(r'C:\Users\...\data\1_preliminary_list_test.csv',encoding='utf-8')
test_id = test_id.drop(['label'],axis=1)
merge_test = pd.merge(test_id, merge, how='inner', on='uuid')
merge_test.to_csv(r'C:\Users\...\merge_test.csv', index=False)


##################### 接下来进行模型的训练 ############################

merged_features_fp = r'C:\Users\...\merge_train.csv'
label_fp = train_id_list
X, Y = get_XY(merged_features_fp, label_fp)
scaler = StandardScaler()
X = scaler.fit_transform(X)
xx_train,xx_test,yy_train,yy_test = train_test_split(X,Y)

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

#确定训练好模型后再进行预测
merged_features_fp = r'C:\Users\...\merge_test.csv'
label_fp = r'C:\Users\...\data\1_preliminary_list_test.csv'
X, Y = get_XY(merged_features_fp, label_fp)
scaler = StandardScaler()
X = scaler.fit_transform(X)
test_pred = final_model.predict(X)
out_predict_fp = r'C:\Users\...\predict_result.csv'
save_predict_result(test_pred,label_fp,out_predict_fp)
