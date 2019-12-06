# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:24:44 2019

@author: Porthita
"""

        
#text_segmentation_one_tsv(r'C:\Users\Porthita\OneDrive\桌面\比赛\阿兹海默\data\tsv\P0001_0017.tsv',r'C:\Users\Porthita\OneDrive\桌面\程序测试\对话汇总.txt')
import pandas as pd
import numpy as np
import re
import jieba
import jieba.analyse
import math
import random


def get_origin_text1(annotated_text):
    PATTERN_1 = re.compile('【.*?】')  # 方括号注释
    PATTERN_2 = re.compile('&.')  # 语气词
    PATTERN_3 = re.compile('(｛|｝)|(\{|\})')  # 语法错误
    PATTERN_4 = re.compile('\(.*?\)|（.*?）')  # 重复修正
    PATTERN_5 = re.compile('/')  # 重复修正
    PATTERN_6 = re.compile('\?|？|，|\,|。')  # 标点符号 
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
    return origin_text

def txt_one_tsv1(tsv_path, outfile_path=None):
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
    mask = (tsv_df.speaker == '<A>') | (tsv_df.speaker == '<B>') 
    tsv_df = tsv_df[mask].reset_index(drop=True)
    total_rows = len(tsv_df)
    for indx in range(total_rows):
        speaker, value = tsv_df.loc[indx, ['speaker', 'value']]
        ori_text = get_origin_text1(value)
        if ori_text == '':
            continue
        line_text = seg_sentence(ori_text)
#        seg_list = jieba.cut(ori_text, cut_all=False, HMM=True)
#        seg_list = list(seg_list)
        if outfile_path:
            out_f.write(line_text+',') 
#                 out_f.write(' '.join(seg_list)+',')       
    if outfile_path:
        out_f.close()
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
    
# 创建停用词list  
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  
  
  
# 对句子进行分词  
def seg_sentence(sentence):  
    sentence_seged = jieba.cut(sentence.strip(), cut_all=False, HMM=True)  
    stopwords = stopwordslist(r'C:\Users\Porthita\OneDrive\桌面\比赛\哈工大停用词表.txt')  # 这里加载停用词的路径  
    outstr = ''  
    for word in sentence_seged:  
        if word not in stopwords:  
            if word != '\t':  
                outstr += word  
                outstr += " "  
    return outstr  

def get_samelabel_txt(label_list_path,la_bel,out_filepath):
    dir_path = "C:/Users/Porthita/OneDrive/桌面/比赛/阿兹海默/data/tsv/"
    train_id = pd.read_csv(label_list_path,encoding='utf-8')
    ad = (train_id.label == la_bel)
    train_id = train_id[ad].reset_index(drop=True)
    for index in train_id.index:
        name = train_id.at[index, 'uuid']
        file_path = dir_path + name + '.tsv'
        txt_one_tsv1(file_path, out_filepath)

def comparation(label_list_path,out_filepath,output_result_filepath):
    dir_path = "C:/Users/Porthita/OneDrive/桌面/比赛/阿兹海默/data/tsv/"
    train_id = pd.read_csv(label_list_path,encoding='utf-8')
    tag1 = tags_extract(r'C:\Users\Porthita\OneDrive\桌面\程序测试\AD(新停词表).txt')
    tag0 = tags_extract(r'C:\Users\Porthita\OneDrive\桌面\程序测试\CTRL(新停词表).txt')
#    c = random.sample(range(0,179),50)
#    lis = train_id.loc[c]
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
# =============================================================================
#         if a1>a0 :
#             lis.loc[i,'test_label'] = 'AD'
#         elif a1<a0:
#             lis.loc[i,'test_label'] = 'CTRL'
#         else:
#             lis.loc[i,'test_label'] = 'none'
#             
#     return lis
#         
# =============================================================================
#comparation(r'C:\Users\Porthita\OneDrive\桌面\比赛\阿兹海默\data\1_preliminary_list_train.csv',r'C:\Users\Porthita\OneDrive\桌面\程序测试\out_file.txt')

# =============================================================================
# result1['test_label'] = 0
# for index in result1.index:
#     if (result1.loc[index,'siml_AD'] > result1.loc[index,'siml_CTRL']):
#         result1.loc[index,'test_label'] = 'AD'
#     elif (result1.loc[index,'siml_AD'] < result1.loc[index,'siml_CTRL']):
#         result1.loc[index,'test_label'] = 'CTRL'
#     else:
#         result1.loc[index,'test_label'] = 'none'
# 
# =============================================================================
# =============================================================================
# #求交集
# tmp = [val for val in tag0 if val in tag1]
# 
# =============================================================================
