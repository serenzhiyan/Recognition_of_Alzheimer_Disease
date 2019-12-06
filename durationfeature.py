# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 19:36:03 2019

@author: Porthita
"""
import pandas as pd
import numpy as np

##################### 针对句子长度提取统计量 ############################
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
