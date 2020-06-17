# -*- coding: UTF-8 -*-
import sys
import jieba.posseg as pseg
import jieba
import argparse

t = pseg.POSTokenizer()
jieba.initialize()

def segment_nodict(string):
    """
    输入为要分词的句子，输出分词后和标注后的结果
    :param string: the string that need to be segment
    :return: 分词后的结果，用一个列表表示，列表的每一个元素是一个元组，元组第一维表示分好后的词，第二维表示词性。
    """
    segment_words = t.cut(string)
    segment_list = []
    for i in segment_words:
        segment_list.append((i.word, i.flag))
    return segment_list


if __name__ == '__main__':
    with open('jiebapos_union3.txt', 'r') as f:
        lines = f.readlines()
    str = ''
    for index, item in enumerate(lines):
        if item != '\n':
            word = item.split('\t')[0].strip()
        else:
            word = item
        str += word
    seg_list_seg_again = segment_nodict(str)
    with open('jiebapos_union3_seg_again.txt', 'w') as f:
        for index, item in enumerate(seg_list_seg_again):
            if item[0] != '\n':
                f.write(item[0] + '\t' + item[1] + '\n')
            else:
                f.write('\r')
            if item[0] == '。' and index < len(seg_list_seg_again) - 1 and seg_list_seg_again[index + 1][0] != '\n':
                f.write('\n')
