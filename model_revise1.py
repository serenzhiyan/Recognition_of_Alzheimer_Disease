# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:06:38 2019

@author: Porthita
"""

import pandas as pd
import numpy as np
import re
import jieba

def analysis_df(part_df):
    ''' 分析一个DataFrame得到统计量，以字典的形式返回
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
    print('\nduration:')
    for index in df.index:
        name = df.at[index, 'uuid']
        print("%s" % name, end=' ', flush=True)
        file_path = dir_path + name + '.tsv'
        result = analysis_one_tsv(file_path)
        for key in result:
            df.loc[index, key] = result[key]
    df.to_csv(duration_fp, index=False)


def get_origin_text(annotated_text):
    PATTERN_1 = re.compile('【.*?】')  # 方括号注释
    PATTERN_2 = re.compile('&.')  # 语气词
    PATTERN_3 = re.compile('(｛|｝)|(\{|\})')  # 语法错误
    PATTERN_4 = re.compile('\(.*?\)|（.*?）')  # 重复修正
    PATTERN_5 = re.compile('/')  # 重复修正
    PATTERN_6 = re.compile('\?|？|，|。')  # 标点符号 
    PATTERN_7 = re.compile('//')
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
    origin_text = PATTERN_1.sub('', annotated_text)
    origin_text, num_filledpause = PATTERN_2.subn('', origin_text)
    origin_text = PATTERN_3.sub('', origin_text)
    origin_text = PATTERN_4.sub('', origin_text)
    origin_text, num_correction = PATTERN_7.subn('', origin_text)
    origin_text, num_repeat = PATTERN_5.subn('', origin_text)
    origin_text = PATTERN_6.sub(' ', origin_text)
    return origin_text, num_filledpause, num_correction, num_repeat

def text_segmentation_one_tsv(tsv_path, outfile_path=None):
    ''' 对一个人的文本进行分词
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
    # tsv_df['text_seg'] = None
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
        seg_list = jieba.cut(ori_text, cut_all=False, HMM=True)
        seg_list = list(seg_list)
        if speaker.strip() == '<A>':
            sum_A_filledpause += num_filledpause
            sum_A_correction += num_correction
            sum_A_repeat += num_repeat
            sum_A_word += len(seg_list)
            if outfile_path:
                 out_f.write(' '.join(seg_list)+'\n')
        elif speaker.strip() == '<B>':
            sum_B_filledpause += num_filledpause
            sum_B_correction += num_correction
            sum_B_repeat += num_repeat
            sum_B_word += len(seg_list)
            if outfile_path:
                out_f.write(' '.join(seg_list)+'\n')
# =============================================================================
#         speaker, value = tsv_df.loc[indx, ['speaker', 'value']]
#         if speaker.strip() == '<B>':
#             ori_text, num_filledpause, num_correction, num_repeat, = get_origin_text(value)
#             if ori_text == '':
#                 continue
#             seg_list = jieba.cut(ori_text, cut_all=False, HMM=True)
#             seg_list = list(seg_list)
#             sum_B_filledpause += num_filledpause
#             sum_B_correction += num_correction
#             sum_B_repeat += num_repeat
#             sum_B_word += len(seg_list)
#             if outfile_path:
#                 out_f.write(' '.join(seg_list)+'\n')
#         # if outfile_path:
#         #     out_f.write(' '.join(seg_list)+'\n')
#         # tsv_df.loc[indx, 'text_seg'] = '/'.join(seg_list)
#         # tsv_df.to_csv(outtsv_path, encoding='utf-8', sep='\t', index=Flase)
# =============================================================================
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
                       'sum_B_word':0})
    for index in df.index:
        name = df.at[index, 'uuid']
        file_path = dir_path + name + '.tsv'
        result = text_segmentation_one_tsv(file_path)
        for key in result:
            # print(key, index)
            df.loc[index, key] = result[key]
    # print(df.info())
    df.sum_B_filledpause = df.sum_B_filledpause/df.sum_B_word
    df.sum_B_correction  = df.sum_B_correction/df.sum_B_word
    df.sum_B_repeat      = df.sum_B_repeat/df.sum_B_word
    df.sum_A_filledpause = df.sum_A_filledpause/df.sum_A_word
    df.sum_A_correction  = df.sum_A_correction/df.sum_A_word
    df.sum_A_repeat      = df.sum_A_repeat/df.sum_A_word
    df.to_csv(out_put_file_path, index=False)


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
    
label_fp = r'C:\Users\Porthita\OneDrive\桌面\比赛\阿兹海默\data\egemaps_pre.csv'
df = pd.read_csv(label_fp)
name_list = df.uuid
duration_fp = r'C:\Users\Porthita\OneDrive\桌面\程序测试\duration.csv'
extract_duration_features_main(name_list, duration_fp, "C:/Users/Porthita/OneDrive/桌面/比赛/阿兹海默/data/tsv/")
linguistic_fp = r'C:\Users\Porthita\OneDrive\桌面\程序测试\linguistic.csv'
extract_linguistic_features_main(name_list, linguistic_fp, "C:/Users/Porthita/OneDrive/桌面/比赛/阿兹海默/data/tsv/")
egemaps_fp = r'C:\Users\Porthita\OneDrive\桌面\比赛\阿兹海默\data\egemaps_pre.csv'
merged_features_fp =  r'C:\Users\Porthita\OneDrive\桌面\程序测试\merged.csv'
merge = merge_common([duration_fp, linguistic_fp, egemaps_fp], merged_features_fp)
train_id = pd.read_csv(r'C:\Users\Porthita\OneDrive\桌面\比赛\阿兹海默\data\1_preliminary_list_train.csv',encoding='utf-8')
train_id = train_id.drop(['label'],axis=1)
merge_train = pd.merge(train_id, merge, how='inner', on='uuid')
merge_train.to_csv(r'C:\Users\Porthita\OneDrive\桌面\程序测试\merge_train.csv', index=False)
test_id = pd.read_csv(r'C:\Users\Porthita\OneDrive\桌面\比赛\阿兹海默\data\1_preliminary_list_test.csv',encoding='utf-8')
test_id = test_id.drop(['label'],axis=1)
merge_test = pd.merge(test_id, merge, how='inner', on='uuid')
merge_test.to_csv(r'C:\Users\Porthita\OneDrive\桌面\程序测试\merge_test.csv', index=False)

duration = pd.read_csv(r'C:\Users\Porthita\OneDrive\桌面\程序测试\duration.csv',encoding='utf-8')
linguistic = pd.read_csv(r'C:\Users\Porthita\OneDrive\桌面\程序测试\linguistic.csv',encoding='utf-8')
egenmaps = pd.read_csv(r'C:\Users\Porthita\OneDrive\桌面\程序测试\egemaps_cut.csv',encoding='utf-8')
tag_similarity = pd.read_csv(r'C:\Users\Porthita\OneDrive\桌面\程序测试\tags_similarity.csv',encoding='utf-8')
merge = pd.merge(duration, egenmaps, how='inner', on='uuid')
merge['ave_A_word'] = merge.sum_A_word/merge.A_speak_num
merge['ave_B_word'] = merge.sum_B_word/merge.B_speak_num
merge_train = pd.merge(merge, train_id, how='inner', on='uuid')
dummies_label = pd.get_dummies(train_id['label'], prefix='label')
Y = dummies_label.label_AD.values
merge_train = merge_train.drop(['uuid'],axis=1)
X = merge_train.values