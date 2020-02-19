# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
##################### 针对句子长度提取统计量 ############################

import pandas as pd
import numpy as np

def analysis_df(part_df):
    ''' 
    分析一个谈话的DataFrame，得到统计量：谈话的总时长、A讲话的次数、
    讲话时长的总和、均值和方差，同样还有B的对应统计量，以字典的形式返回
    '''
    total_rows = len(part_df)
    part_df = part_df.reset_index(drop=True)

    interview_start_time = part_df.loc[0, 'start_time']
    interview_end_time = part_df.loc[total_rows - 1, 'end_time']

    total_duration = interview_end_time-interview_start_time

    A_speak_num = (part_df.speaker == '<A>').sum()
    A_speak_duration_sum = part_df.duration[part_df.speaker == '<A>'].sum()
    A_speak_duration_mean = part_df.duration[part_df.speaker == '<A>'].mean()
    A_speak_duration_std = part_df.duration[part_df.speaker == '<A>'].std()

    B_speak_num = (part_df.speaker == '<B>').sum()
    B_speak_duration_sum  = part_df.duration[part_df.speaker == '<B>'].sum()
    B_speak_duration_mean = part_df.duration[part_df.speaker == '<B>'].mean()
    B_speak_duration_std  = part_df.duration[part_df.speaker == '<B>'].std()

    silence_duration_sum = total_duration - A_speak_duration_sum - B_speak_duration_sum

    result = {
        'total_duration'       : total_duration,
        'A_speak_num'          : A_speak_num,

        'A_speak_duration_mean': A_speak_duration_mean,
        'A_speak_duration_std' : A_speak_duration_std,

        'B_speak_num'          : B_speak_num,

        'B_speak_duration_mean': B_speak_duration_mean,
        'B_speak_duration_std' : B_speak_duration_std,

        # 语音总长占比 语音总长度的比值 语音平均长度的比值 语音方差的比值
        'A_speak_duration_proportion': A_speak_duration_sum / total_duration,
        'B_speak_duration_proportion': B_speak_duration_sum / total_duration,
        'slience_duration_proportion': silence_duration_sum / total_duration,
    }
    return result

def analysis_one_tsv(filename):
    '''分析一个人的TSV文件得到统计量
    TSV中需要包含这些列: start_time, end_time, speaker, value
    返回一个DataFrame
    '''
    tsv_df = pd.read_csv(open(filename, encoding='utf-8'), sep='\t')

    mask = (tsv_df.speaker == '<A>') | (tsv_df.speaker == '<B>') | (tsv_df.speaker == '<OVERLAP>')
    tsv_df = tsv_df[mask].reset_index(drop=True)
    '''
    生成了新的一列：每句话的时长
    '''
    tsv_df['duration'] = tsv_df.end_time - tsv_df.start_time

    result = analysis_df(tsv_df)
    return result


def extract_duration_features_main(name_list, duration_fp, dir_path):
    """ 针对句子时间长度提取统计量
    参数：
        name_list: 需要提取特征的人名列表
        duration_fp: 提取的特征文件存放路径
        dir_path:  TSV文件存放目录
    """
    uuids = pd.Series(name_list)
    df = pd.DataFrame({'uuid': uuids, 'total_duration': 0.0})
    #print('\nduration:')
    for index in df.index:
        name = df.at[index, 'uuid']
        #print("%s" % name, end=' ', flush=True)
        file_path = dir_path + name + '.tsv'
        result = analysis_one_tsv(file_path)
        for key in result:
            df.loc[index, key] = result[key]
    df.to_csv(duration_fp, index=False) #结果返回的是从对话中提取的所有name_list的对话特征。


##################### 针对谈话的内容提取一些文本特征 ############################

import re #正则表达式（进行文本过滤）
import jieba #分词包

#定义正则表达式的模式pattern
PATTERN_1 = re.compile('【.*?】')  # 方括号注释
PATTERN_2 = re.compile('&.')  # 语气词
PATTERN_3 = re.compile('(｛|｝)|(\{|\})')  # 语法错误
PATTERN_4 = re.compile('\(.*?\)|（.*?）')  # 重复修正
PATTERN_5 = re.compile('/')  # 重复修正
PATTERN_6 = re.compile('\?|？|，|\,|。')  # 标点符号 
PATTERN_7 = re.compile('//')

def get_origin_text(annotated_text):
    ''' 对有标注的文本进行处理，得到原始文本
    参数:
        annotated_text : 有标注的文本
    返回:
        origin_text : 删去人工注释的文本 
        num_filledpause : 语气词（有声停顿）的个数  
        num_repeat : 重复的次数
        num_correction : 修正的次数
        num_error : 语法错误的次数
    '''
    origin_text = PATTERN_1.sub('', annotated_text) #sub函数：使用''去替换annotated_text中匹配上的模式，
    origin_text, num_filledpause = PATTERN_2.subn('', origin_text) #subn的话还会返回替换次数
    origin_text = PATTERN_3.sub('', origin_text)
    origin_text = PATTERN_4.sub('', origin_text)
    origin_text, num_correction = PATTERN_7.subn('', origin_text)
    origin_text, num_repeat = PATTERN_5.subn('', origin_text)
    origin_text = PATTERN_6.sub(' ', origin_text)
    
    return origin_text, num_filledpause, num_correction, num_repeat

def text_segmentation_one_tsv(tsv_path，outfile_path=None):
    ''' 对一个人的谈话的文本进行分词
    参数:
        tsv_path:  TSV文件所在目录
        outfile_path: 输出文件目录
    返回:
        sum_filledpause : 语气词（有声停顿）的个数之和
        sum_correction : 修正的次数之和
        sum_repeat : 重复的次数之和
        sum_error : 语法错误的次数之和
    '''
    out_f = None
    if outfile_path is not None:
        out_f = open(outfile_path, 'w', encoding='utf-8')
    # process one tsv
    tsv_df = pd.read_csv(open(tsv_path, encoding='utf-8'), sep='\t')
    total_rows = len(tsv_df)
    sum_A_filledpause = 0
    sum_A_correction = 0
    sum_A_repeat = 0
    sum_A_word = 0
    sum_B_filledpause = 0
    sum_B_correction = 0
    sum_B_repeat = 0
    sum_B_word = 0
    for indx in range(total_rows):
        speaker, value = tsv_df.loc[indx, ['speaker', 'value']]
        if type(speaker) is not str:
            print('warning %s:%d line: speaker:%s' % (tsv_path, indx, str(speaker)))
            continue
        ori_text, num_filledpause, num_correction, num_repeat = get_origin_text(value)
        if ori_text == '':
            continue
        line_text = seg_sentence(ori_text)
        if outfile_path:
                out_f.write(line_text+',') 
        seg_list = jieba.cut(ori_text, cut_all=False, HMM=True)
        seg_list = list(seg_list)
        if speaker.strip() == '<A>':
            sum_A_filledpause += num_filledpause
            sum_A_correction += num_correction
            sum_A_repeat += num_repeat
            sum_A_word += len(seg_list)    
        elif speaker.strip() == '<B>':
            sum_B_filledpause += num_filledpause
            sum_B_correction += num_correction
            sum_B_repeat += num_repeat
            sum_B_word += len(seg_list)
    if outfile_path:
        out_f.close()
    result = {
            'sum_B_filledpause': sum_B_filledpause,
            'sum_B_correction' : sum_B_correction,
            'sum_B_repeat'     : sum_B_repeat,
            'sum_B_word'       : sum_B_word,
            'sum_A_filledpause': sum_A_filledpause,
            'sum_A_correction' : sum_A_correction,
            'sum_A_repeat'     : sum_A_repeat,
            'sum_A_word'       : sum_A_word,
            }
    return result

def extract_linguistic_features_main(name_list, out_put_file_path, dir_path):
    ''' 提取linguistic特征
    参数：
        name_list: 需要提取特征的人名列表
        out_put_file_path: 提取出的特征文件存放路径
        dir_path:  TSV文件存放目录
    '''

    uuids = pd.Series(name_list)
    df = pd.DataFrame({'uuid':uuids,
                       'sum_B_filledpause':0,
                       'sum_B_correction':0,
                       'sum_B_repeat':0,
                       'sum_B_word':0
                       'sum_A_filledpause':0,
                       'sum_A_correction':0,
                       'sum_A_repeat':0,
                       'sum_A_word':0})
    for index in df.index:
        name = df.at[index, 'uuid']
        file_path = dir_path + name + '.tsv'
        result = text_segmentation_one_tsv(file_path)
        for key in result:
            df.loc[index, key] = result[key]
    df.sum_B_filledpause = df.sum_B_filledpause/df.sum_B_word
    df.sum_B_correction  = df.sum_B_correction/df.sum_B_word
    df.sum_B_repeat      = df.sum_B_repeat/df.sum_B_word
    df.sum_A_filledpause = df.sum_A_filledpause/df.sum_A_word
    df.sum_A_correction  = df.sum_A_correction/df.sum_A_word
    df.sum_A_repeat      = df.sum_A_repeat/df.sum_A_word
    df.to_csv(out_put_file_path, index=False)

##################### 考虑对声学特征进行降维处理（或者是对最后汇总的所有特征统计量进行降维处理） ############################

from sklearn.decomposition import PCA #这里使用主成分分析法进行降维
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score
 
def dimension_of_pca(feature_path,train_id_list,out_put_path):
    scaler = StandardScaler()
    df = pd.read_csv(feature_path,encoding='utf-8', index_col=0)
    fea = df.values
    uuid = df.index
    train_id = pd.read_csv(train_id_list,encoding='utf-8')
    dummies_label = pd.get_dummies(train_id['label'], prefix='label')
    Y = dummies_label.label_AD.values
    train_id = train_id.drop(['label'],axis=1)
    df_id = pd.DataFrame({'uuid':uuid,
                           })
    n_column = df.shape[1]
    score = np.zeros(n_column)
    for i in range(1,n_column+1): #该for循环是为了寻找最佳的降维维数
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
        #print('维数',i)
        #print(score[i-1])
    
    score = score.tolist()
    j = score.index(max(score))
    #print('最佳维数：',j+1)
    fea_pca = PCA(n_components=j+1).fit_transform(fea)
    fea_pca = pd.DataFrame(fea_pca)
    fea_pca = pd.merge(df_id, fea_pca,left_index=True,right_index=True)
    fea_pca.to_csv(out_put_path, index=False)


##################### 自定义的特征：相似度（待预测样本与两类人群看图谈话内容的相似程度） ############################

import jieba.analyse
import math
import random      

# 创建停用词list  
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  
  
  
# 对句子进行分词  
def seg_sentence(sentence,stopwords_fp):  
    sentence_seged = jieba.cut(sentence.strip(), cut_all=False, HMM=True)  
    stopwords = stopwordslist(stopwords_fp)  # 这里写停用词的路径  
    outstr = ''  
    for word in sentence_seged:  
        if word not in stopwords:  
            if word != '\t':  
                outstr += word  
                outstr += " "  
    return outstr  

def txt_one_tsv1(tsv_path,stopwords_fp,outfile_path=None):
    out_f = None
    if outfile_path is not None:
        out_f = open(outfile_path, 'w', encoding='utf-8')
    tsv_df = pd.read_csv(open(tsv_path, encoding='utf-8'), sep='\t')
    mask = (tsv_df.speaker == '<A>') | (tsv_df.speaker == '<B>') 
    tsv_df = tsv_df[mask].reset_index(drop=True)
    total_rows = len(tsv_df)
    for indx in range(total_rows):
        speaker, value = tsv_df.loc[indx, ['speaker', 'value']]
        ori_text = get_origin_text(value)[0]
        if ori_text == '':
            continue
        line_text = seg_sentence(ori_text,stopwords_fp)
        if outfile_path:
            out_f.write(line_text+',')      
    if outfile_path:
        out_f.close()

def get_samelabel_txt(stopwords_fp,label_list_path,la_bel,out_filepath,dir_path):
    
    train_id = pd.read_csv(label_list_path,encoding='utf-8')
    ad = (train_id.label == la_bel)
    train_id = train_id[ad].reset_index(drop=True)
    for index in train_id.index:
        name = train_id.at[index, 'uuid']
        file_path = dir_path + name + '.tsv'
        txt_one_tsv1(file_path,stopwords_fp,out_filepath)


#提取关键词
def tags_extract(txt_path):
    content = open(txt_path, 'r',encoding='utf-8').read()
    tags = jieba.analyse.extract_tags(content)
    return tags

def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a,b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down

def similarity(s1,s2):
    key=list(set(s1+s2))
    keyLen=len(key)
    keyValue=0
    sk1=[keyValue]*keyLen
    sk2=[keyValue]*keyLen
    for index,keyElement in enumerate(key):
        if keyElement in s1:
            sk1[index]=sk1[index]+1
        if keyElement in s2:
            sk2[index]=sk2[index]+1 
    dist = cos_dist(sk1,sk2)
    return dist
    

def comparation(out_filepath_AD,out_filepath_CTRL,label_list_path,out_filepath,output_result_filepath,dir_path):
    
    train_id = pd.read_csv(label_list_path,encoding='utf-8')
    tag1 = tags_extract(out_filepath_AD)
    tag0 = tags_extract(out_filepath_CTRL)
    train_id['siml_AD'] = 0
    train_id['siml_CTRL'] = 0
    for index in train_id.index:
        name = train_id.at[index, 'uuid']
        file_path = dir_path + name + '.tsv'
        txt_one_tsv1(file_path, out_filepath)
        tag = tags_extract(out_filepath)
        a1 = similarity(tag,tag1)
        a0 = similarity(tag,tag0)
        train_id.loc[index,'siml_AD'] = a1
        train_id.loc[index,'siml_CTRL'] = a0
    train_id.to_csv(output_result_filepath,index=False)
    return train_id
 
##################### 合并所有特征的函数 ############################

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
    return data_all
    
##################### 评估模型好坏的函数 ############################



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

def save_predict_result(test_predict,test_list_fp,out_predict_fp):
    """ 保存预测结果
    参数:
        test_predict : 预测结果，一个Numpy数组
    """
    label_fp = test_list_fp
    Y_test_df = pd.read_csv(label_fp, encoding='utf-8', index_col=0)
    Y_test_df['pred_value'] = test_pred
    Y_test_df.loc[(Y_test_df.pred_value == 1), 'label'] = 'AD'
    Y_test_df.loc[(Y_test_df.pred_value == 0), 'label'] = 'CTRL'
    Y_test_df.drop(columns=['pred_value'], inplace=True)
    Y_test_df.to_csv(out_predict_fp)



