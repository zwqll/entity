# -*- coding: UTF-8 -*-
from segment import segment
from segment_nodict import segment_nodict
import csv,codecs,os

entities = ['symp', 'disease', 'drug']
labels = ['SYM', 'DIS', 'DRU']
path = '../data/'
if not os.path.exists(path):
    os.makedirs(path)
    
def originalcsv2postxt(textdata):
    """
    对原始语料进行分词，并用jieba获取每个词的词性，每句话用空行分隔
    """
    seg_list = segment(textdata)

    with codecs.open('jiebapos_union3.txt', 'w','utf-8') as f:
        for index, item in enumerate(seg_list):
            if item[0] != '\r\n':
                f.write(item[0] + '\t' + item[1] + '\r\n')
            else:
                f.write('\r\n')
            if item[0] == u'。' and index < len(seg_list) - 1 and seg_list[index + 1][0] != '\r\n':
                f.write('\r\n')


def label(label, templines, i, word, pos):
    """
    工具方法，打标签
    """
    if label == 'O':
        templines[i] = word + '\t' + pos + '\t' + label + '\r\n'
    else:
        templines[i] = word + '\t' + pos + '\t' + label + '-' + labels[entities.index(pos)] + '\r\n'


def match_entities():
    """
    部分数据清洗与预处理，并匹配出词典中的实体词
    """
    temppos = []
    templines = []
    finallines = []
    with codecs.open('jiebapos_union3.txt', 'r') as f1, codecs.open('jiebaner_union3.txt', 'w') as f2:
        originallines = f1.readlines()
        for index, item in enumerate(originallines):
            if item != '\r\n' and item != '\n':
                templines.append(item)
            else:
                for i, j in enumerate(templines):
                    word = j.split('\t')[0]
                    if word in [' ', ' ', ' ']:
                        word = '，'
                    print j
                    pos = j.split('\t')[1].strip()
                    temppos.append(pos)
                    if pos in entities:
                        label('S', templines, i, word, pos)
                    else:
                        label('O', templines, i, word, pos)
                for i in templines:
                    finallines.append(i)
                finallines.append('\r\n')
                templines = []
                temppos = []
        for i in finallines:
            f2.write(i)


def nerlabel_for_sentence(len_tag_list, label_type):
    """
    根据二次分词返回的词的数量，给实体词打上NER标签
    """
    tag_list = []
    if len_tag_list > 2:
        tag_list.append('B-' + labels[label_type])
        for _ in range(len_tag_list - 2):
            tag_list.append('I-' + labels[label_type])
        tag_list.append('E-' + labels[label_type])
    elif len_tag_list == 2:
        tag_list.append('B-' + labels[label_type])
        tag_list.append('E-' + labels[label_type])
    elif len_tag_list == 1:
        tag_list.append('S-' + labels[label_type])
    return tag_list


def nerlabeling_seg_again():
    """
    二次分词函数入口，写入文件中
    """
    disease_dict = []
    with codecs.open('dict/disease_union3.txt', 'r','utf-8') as f1:
        lines = f1.readlines()
        for i in lines:
            disease_dict.append(i.split(' ')[0].strip())

    with codecs.open('jiebaner_union3.txt', 'r') as f1, codecs.open('jiebaner_union3_seg_again.txt', 'w','utf-8') as f2:
        lines = f1.readlines()
        for line in lines:
            if line == '\r\n':
                f2.write('\r\n')
            else:
                word = line.split('\t')[0].strip()
                pos = line.split('\t')[1].strip()
                tag = line.split('\t')[2].strip()
                if tag in ['S-'+labels[1]]:
                    seg_list_seg_again_S = segment_nodict(word)
                    tag_list = nerlabel_for_sentence(len(seg_list_seg_again_S), 1)
                    for index, item in enumerate(seg_list_seg_again_S):
                        f2.write(item[0] + '\t' + item[1] + '\t' +tag_list[index] + '\r\n')
                elif pos in [entities[0],entities[2]]:
                    seg_list_seg_again_S = segment_nodict(word)
                    tag_list = nerlabel_for_sentence(len(seg_list_seg_again_S), 1)
                    for index, item in enumerate(seg_list_seg_again_S):
                        f2.write(item[0] + '\t' + item[1] + '\t' + 'O' + '\r\n')
                else:
                    f2.write(line.decode('utf-8'))


def SplitDataset():
    """
    根据7：1.5：1.5的比例将数据集划分为训练集、验证集和测试集
    统计分词结果
    """
    with codecs.open('jiebaner_union3_seg_again.txt', 'r') as f11:
        line11 = f11.readlines()
        count = 0
        count_sentence_len = 0
        count_sentence_len_list = []
        DIS_worddict = {}
        DIS_wordlen_list = []
        with codecs.open(path + 'train_ner.txt', 'w') as f12, codecs.open(path + 'dev_ner.txt', 'w') as f13, codecs.open(path + 'test_ner.txt', 'w') as f14:
            sentence_count = 0
            for i in line11:
                if i == '\r\n':
                    sentence_count += 1
            for i in line11:
                if i != '\r\n':
                    tag = i.split('\t')[2].strip()
                    word = i.split('\t')[0].strip()
                    if tag in ['S-' + labels[1]]:
                        DIS_wordlen_list.append(len(word))
                        if word not in DIS_worddict:
                            DIS_worddict[word] = 1
                        else:
                            DIS_worddict[word] += 1
                count_sentence_len += 1
                if count < 0.7 * sentence_count:
                    f12.write(i)
                elif count >0.7 * sentence_count and count < 0.85 * sentence_count:
                    f13.write(i)
                else:
                    f14.write(i)
                if i == '\r\n':
                    count_sentence_len_list.append(count_sentence_len)
                    count_sentence_len = 0
                    count += 1
            print count
            print sorted(count_sentence_len_list)[::-10]
    with codecs.open(path + 'test_ner.txt', 'r') as f15, codecs.open(path + 'test_sentence.txt', 'w') as f16:
        line15 = f15.readlines()
        for i in line15:
            if i != '\r\n':
                f16.write(i.split('\t')[0].strip() + '___' + i.split('\t')[1].strip() + ' ')
            else:
                f16.write('\r\n\r\n')
    DIS_wordclasslen_list = []
    for i in DIS_worddict.items():
        DIS_wordclasslen_list.append(len(i[0]))
    print 'Number of length of word with S label: ' 
    for i in range(1,6):
        print 'length '+str(i)+': '+str(DIS_wordclasslen_list.count(i))+' wordcount: '+str(DIS_wordlen_list.count(i))
    print sorted(DIS_worddict.items(), key = lambda item:item[1],reverse = True) 


if __name__ == '__main__':
    originalcsv2postxt()
    match_entities()
    nerlabeling_seg_again()
    SplitDataset()
