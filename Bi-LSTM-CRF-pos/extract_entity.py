# -*- coding: utf-8 -*-
import xlrd
import jieba
import jieba.posseg as pseg

def segment(string):
    """
    输入为要分词的句子，输出分词后和标注后的结果
    :param string: the string that need to be segment
    :return: 分词后的结果，用一个列表表示，列表的每一个元素是一个元组，元组第一维表示分好后的词，第二维表示词性。
    """
    # jieba.load_userdict("userdict.txt")
    segment_words = pseg.cut(str(string))
    segment_list = []
    for i in segment_words:
        segment_list.append((i.word, i.flag))
    return segment_list


def is_float(s):
    try:
        float(s) # is a number(either integer or real)
        return True
    except:
        return False




if __name__ == '__main__':
    ENTITY_TYPES = ['CAR','OPT']

    data = xlrd.open_workbook('QA_D90_2entity.xlsx')
    table = data.sheets()[0] 
    cnt = 0
    excel_rows_list = []
    for i in range(1,table.nrows):
        if table.row_values(i)[1]:
            cnt += 1
            excel_rows_list.append((table.row_values(i)[0],table.row_values(i)[1],table.row_values(i)[2]))
            for word in table.row_values(i)[1].split('##'):
                jieba.add_word(word)
            if not is_float(table.row_values(i)[2]):
                for word in table.row_values(i)[2].split('##'):
                    jieba.add_word(word)
    print 'total lines:', cnt

    with open('train_ner.txt','w') as f_train, open('dev_ner.txt','w') as f_dev, open('test_ner.txt','w') as f_test, open('test_sentence.txt','w') as f_test_sentence:
        for i in excel_rows_list[:int(cnt*0.7)]:
            entities_list_CAR = []
            entities_list_OPT = []
            for entity in i[1].split('##'):
                entities_list_CAR.append(entity)
            if not is_float(i[2]):
                for entity in i[2].split('##'):
                    entities_list_OPT.append(entity)
            for item in segment(i[0].encode('utf8')):
                if item[0] != ' ':
                    if item[0] in entities_list_CAR:
                        f_train.write(item[0].encode('utf8') + '\t' + item[1].encode('utf8') + '\t' + 'S-CAR' + '\n')
                    elif item[0] in entities_list_OPT:
                        f_train.write(item[0].encode('utf8') + '\t' + item[1].encode('utf8') + '\t' + 'S-OPT' + '\n')
                    else:
                        f_train.write(item[0].encode('utf8')+ '\t' + item[1].encode('utf8') + '\t' + 'O' + '\n')
            f_train.write('\n')
        
        for i in excel_rows_list[int(cnt*0.7):int(cnt*0.9)]:
            entities_list_CAR = []
            entities_list_OPT = []
            for entity in i[1].split('##'):
                entities_list_CAR.append(entity)
            if not is_float(i[2]):
                for entity in i[2].split('##'):
                    entities_list_OPT.append(entity)
            for item in segment(i[0].encode('utf8')):
                if item[0] != ' ':
                    if item[0] in entities_list_CAR:
                        f_dev.write(item[0].encode('utf8') + '\t' + item[1].encode('utf8') + '\t' + 'S-CAR' + '\n')
                    elif item[0] in entities_list_OPT:
                        f_dev.write(item[0].encode('utf8') + '\t' + item[1].encode('utf8') + '\t' + 'S-OPT' + '\n')
                    else:
                        f_dev.write(item[0].encode('utf8')+ '\t' + item[1].encode('utf8') + '\t' + 'O' + '\n')
            f_dev.write('\n')
        
        for i in excel_rows_list[int(cnt*0.9):]:
            entities_list_CAR = []
            entities_list_OPT = []
            for entity in i[1].split('##'):
                entities_list_CAR.append(entity)
            if not is_float(i[2]):
                for entity in i[2].split('##'):
                    entities_list_OPT.append(entity)
            for item in segment(i[0].encode('utf8')):
                if item[0] != ' ':
                    if item[0] in entities_list_CAR:
                        f_test.write(item[0].encode('utf8') + '\t' + item[1].encode('utf8') + '\t' + 'S-CAR' + '\n')
                    elif item[0] in entities_list_OPT:
                        f_test.write(item[0].encode('utf8') + '\t' + item[1].encode('utf8') + '\t' + 'S-OPT' + '\n')
                    else:
                        f_test.write(item[0].encode('utf8')+ '\t' + item[1].encode('utf8') + '\t' + 'O' + '\n')
                    f_test_sentence.write(item[0].encode('utf8')+ '___' + item[1].encode('utf8') + ' ')
            f_test.write('\n')
            f_test_sentence.write('\n')
            