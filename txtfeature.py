# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 19:33:06 2019

@author: Porthita
"""

import pandas as pd
import numpy as np
import re
import jieba

PATTERN_1 = re.compile('【.*?】')  # 方括号注释
PATTERN_2 = re.compile('&.')  # 语气词
PATTERN_3 = re.compile('(｛|｝)|(\{|\})')  # 语法错误
PATTERN_4 = re.compile('\(.*?\)|（.*?）')  # 重复修正
PATTERN_5 = re.compile('/')  # 重复修正
PATTERN_6 = re.compile('\?|？|，|。')  # 标点符号
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
    origin_text = PATTERN_1.sub('', annotated_text)
    origin_text, num_filledpause = PATTERN_2.subn('', origin_text)
    origin_text = PATTERN_3.sub('', origin_text)
    origin_text, num_correction_repeat = PATTERN_4.subn('', origin_text)
    origin_text, num_slash = PATTERN_5.subn('', origin_text)
    origin_text = PATTERN_6.sub(' ', origin_text)
    num_correction = num_slash - num_correction_repeat
    num_repeat = num_correction_repeat - num_correction
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
    sum_B_filledpause = 0
    sum_B_correction = 0
    sum_B_repeat = 0
    sum_B_word = 0
    for indx in range(total_rows):
        speaker, value = tsv_df.loc[indx, ['speaker', 'value']]
        if speaker.strip() == '<B>':
            ori_text, num_filledpause, num_correction, num_repeat, = get_origin_text(value)
            if ori_text == '':
                continue
            seg_list = jieba.cut(ori_text, cut_all=False, HMM=True)
            seg_list = list(seg_list)
            sum_B_filledpause += num_filledpause
            sum_B_correction += num_correction
            sum_B_repeat += num_repeat
            sum_B_word += len(seg_list)
            if outfile_path:
                out_f.write(' '.join(seg_list)+'\n')
        # if outfile_path:
        #     out_f.write(' '.join(seg_list)+'\n')
        # tsv_df.loc[indx, 'text_seg'] = '/'.join(seg_list)
        # tsv_df.to_csv(outtsv_path, encoding='utf-8', sep='\t', index=Flase)
    if outfile_path:
        out_f.close()
    result = {
            'sum_B_filledpause': sum_B_filledpause,
            'sum_B_correction' : sum_B_correction,
            'sum_B_repeat'     : sum_B_repeat,
            'sum_B_word'       : sum_B_word
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
    df.to_csv(out_put_file_path, index=False)