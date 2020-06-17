# -*- coding: UTF-8 -*-
import sys
import jieba.posseg as pseg
import jieba
import argparse

def segment(string):
    """
    输入为要分词的句子，输出分词后和标注后的结果
    :param string: the string that need to be segment
    :return: 分词后的结果，用一个列表表示，列表的每一个元素是一个元组，元组第一维表示分好后的词，第二维表示词性。
    """
    jieba.load_userdict("dict/userdict.txt")
    jieba.load_userdict('dict/disease_union3.txt')
    jieba.load_userdict('dict/drug.txt')
    jieba.load_userdict('dict/symptom.txt')
    segment_words = pseg.cut(str(string))
    segment_list = []
    for i in segment_words:
        segment_list.append((i.word, i.flag))
    return segment_list

if __name__ == '__main__':
    jieba.load_userdict("dict/userdict.txt")
    jieba.load_userdict('dict/disease_union3.txt')
    jieba.load_userdict('dict/drug.txt')
    jieba.load_userdict('dict/symptom.txt')
    print(segment('圣琼美食百汇自助餐厅国贸店的电话是多少'))
    print(segment('有慢性胃炎，'))

    t = pseg.POSTokenizer()
    segment_words = t.cut('圣琼美食百汇自助餐厅(国贸店)的电话是多少')

    segment_list = []
    for i in segment_words:
        segment_list.append((i.word, i.flag))

    print(segment_list)
