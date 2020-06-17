# -*- coding: UTF-8 -*-
import sys
import jieba.posseg as pseg
import jieba
import argparse
from MySQLUtil import MySQLUtil

jieba.load_userdict("dict/userdict.txt")
jieba.load_userdict('dict/disease_union3.txt')
jieba.load_userdict('dict/drug.txt')
jieba.load_userdict('dict/symptom.txt')

mysqlLogin = logIn = {'host': '118.194.132.112',
             'user': 'root',
             'db':'DarkMatter',
             'passwd': 'begin@2015',
             'charset': 'utf8'}
             
table = 'EntityDict'
column = ("entityword", "frequency", "part_of_speech")
def getdict():
    mysqlHandler = MySQLUtil(**logIn)
    dataset = mysqlHandler.select(table=table)
    for record in dataset:
        print(record)


def segment(string):
    """
    输入为要分词的句子，输出分词后和标注后的结果
    :param string: the string that need to be segment
    :return: 分词后的结果，用一个列表表示，列表的每一个元素是一个元组，元组第一维表示分好后的词，第二维表示词性。
    """
    segment_words = pseg.cut(str(string))
    segment_list = []
    for i in segment_words:
        segment_list.append((i.word, i.flag))
    return segment_list

if __name__ == '__main__':
    getdict()
    # print(segment('圣琼美食百汇自助餐厅国贸店的电话是多少'))
    # print(segment('有慢性胃炎，'))

    # t = pseg.POSTokenizer()
    # segment_words = t.cut('圣琼美食百汇自助餐厅(国贸店)的电话是多少')

    # segment_list = []
    # for i in segment_words:
        # segment_list.append((i.word, i.flag))

    # print(segment_list)
