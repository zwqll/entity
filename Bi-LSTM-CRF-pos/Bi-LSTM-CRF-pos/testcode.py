# -*- coding:utf-8 -*-
import os,optparse,re,collections,codecs
import pandas as pd
import numpy as np
import csv, xlwt


# 手工特征匹配字符串的范围参数
MATCH_RANGE_SHU = 1
MATCH_RANGE_PART = 2
MATCH_RANGE_POSITION = 1


# 接收待统计文件的路径
optparser = optparse.OptionParser()
optparser.add_option(
    "-i", "--input", default="",
    help="Input file location"
)
opts = optparser.parse_args()[0]
assert opts.input


def prepare(chars, labels, seq_max_len, is_padding=True):
    """
    准备word和pos标签的数组
    句子长度不足seq_max_len的补0
    """
    X = []
    y = []
    tmp_x = []
    tmp_y = []

    for record in zip(chars, labels):
        c = record[0]
        l = record[1]
        # empty line
        if c == -1:
            if len(tmp_x) <= seq_max_len:
                X.append(tmp_x)
                y.append(tmp_y)
            tmp_x = []
            tmp_y = []
        else:
            tmp_x.append(c)
            tmp_y.append(l)
    if is_padding:
        X = np.array(padding(X, seq_max_len))
    else:
        X = np.array(X)
    y = np.array(padding(y, seq_max_len))

    return X, y


def padding(sample, seq_max_len):
    """
    句子长度不足seq_max_len的进行补0操作
    """
    for i in range(len(sample)):
        if len(sample[i]) < seq_max_len:
            sample[i] += [0 for _ in range(seq_max_len - len(sample[i]))]
    return sample


def loadMap(token2id_filepath):
    """
    读取char2id，label2id的映射文件
    """
    if not os.path.isfile(token2id_filepath):
        print "file not exist, building map" 
        buildMap()

    token2id = {}
    id2token = {}
    with codecs.open(token2id_filepath,'r','utf-8') as infile:
        for row in infile:
            row = row.rstrip()
            if row != '':
                token = row.split('\t')[0]
                token_id = int(row.split('\t')[1])
                token2id[token] = token_id
                id2token[token_id] = token
    return token2id, id2token


def getTest(test_path="test.in", is_validation=True, seq_max_len=200):
    """
    从测试文件中读取word和NER标签数据
    """
    char2id, id2char = loadMap("char2id")
    label2id, id2label = loadMap("label2id")
    chars = []
    labels = []
    with codecs.open(test_path,'r','utf-8') as f1:
        lines = f1.readlines()
        for i in lines:
            if i != '\r\n':
                char = i.split('\t')[0].strip()
                label = i.split('\t')[-1].strip()
                chars.append(char)
                labels.append(label)
            else:
                chars.append(np.nan)
                labels.append(np.nan)
    #df_test = pd.read_csv(test_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=["char", "label"])
    def mapFunc(x, char2id):
        if x is np.nan:
            return -1
        elif x not in char2id:
            return char2id["<NEW>"]
        else:
            return char2id[x]
    chars_map = []
    labels_map = []
    for i in chars:
        chars_map.append(mapFunc(i, char2id))
    for i in labels:
        labels_map.append(-1 if i is np.nan else label2id[i])
    #df_test["char_id"] = df_test.char.map(lambda x:mapFunc(x, char2id))
    #df_test["label_id"] = df_test.label.map(lambda x : -1 if str(x) == str(np.nan) else label2id[x])
    if is_validation:
        X_test, y_test = prepare(chars_map, labels_map, seq_max_len)
        #X_test, y_test = prepare(df_test["char_id"], df_test["label_id"], seq_max_len)
        return X_test, y_test
    else:
        df_test["char"] = df_test.char.map(lambda x : -1 if str(x) == str(np.nan) else x)
        X_test, _ = prepare(df_test["char_id"], df_test["char_id"], seq_max_len)
        X_test_str, _ = prepare(df_test["char"], df_test["char_id"], seq_max_len, is_padding=False)
        print "test size: %d" %(len(X_test))
        return X_test, X_test_str


def saveMap(id2char, id2label):
    """
    保存char2id，label2id的映射文件
    """
    with codecs.open("char2id", "w",'utf-8') as outfile:
        for idx in id2char:
            outfile.write(id2char[idx] + "\t" + str(idx)  + "\r\n")
    with codecs.open("label2id", "w", 'utf-8') as outfile:
        for idx in id2label:
            outfile.write(id2label[idx] + "\t" + str(idx) + "\r\n")
    print "saved map between token and id" 


def buildMap(train_path="train.in"):
    """
    根据训练数据建立id2char，id2label的映射文件
    """
    chars = []
    labels = []
    with codecs.open(train_path,'r','utf-8') as f1:
        lines = f1.readlines()
        for i in lines:
            if i != '\r\n':
                char = i.split('\t')[0].strip()
                label = i.split('\t')[-1].strip()
                if char not in chars:
                    chars.append(char)
                if label not in labels:
                    labels.append(label)
    #df_train = pd.read_csv(train_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=["char", "label"])
    #chars = list(set(df_train["char"][df_train["char"].notnull()]))
    #labels = list(set(df_train["label"][df_train["label"].notnull()]))
    char2id = dict(zip(chars, range(1, len(chars) + 1)))
    label2id = dict(zip(labels, range(1, len(labels) + 1)))
    id2char = dict(zip(range(1, len(chars) + 1), chars))
    id2label =  dict(zip(range(1, len(labels) + 1), labels))
    id2char[0] = "<PAD>"
    id2label[0] = "<PAD>"
    char2id["<PAD>"] = 0
    label2id["<PAD>"] = 0
    id2char[len(chars) + 1] = "<NEW>"
    char2id["<NEW>"] = len(chars) + 1

    saveMap(id2char, id2label)

    return char2id, id2char, label2id, id2label


def extractEntityIOB2(id2char, sentence_id, id2label, labels_id):
    """
    提取一个句子中的实体，该句子标注规范为IOB2
    """
    entitys = []
    entitys_SYM = []
    entitys_DIS = []
    entitys_DRU = []
    start_index_id = 0
    re_entity = re.compile(r'(B-(SYM|DIS|DRU)(I-(SYM|DIS|DRU))*)')
    #labels = ''.join([str(id2label[val]) for val in labels_id ])

    tags_list = []
    [tags_list.append(str(id2label[val])) for val in labels_id if val != 0]
    tags_list = iobes_iob(tags_list)
    tags = ''.join([str(i) for i in tags_list])
    m = re_entity.search(tags)
    dif = 0
    temp = 0
    while m:
        entity_labels = m.group()
        start_index_char = tags.find(entity_labels)
        dif = start_index_char - temp
        start_index_id += dif
        temp = start_index_char + len(entity_labels)
        if len(entity_labels) > 6:
            entitylen = entity_labels.count('I-') + 1
        else:
            entitylen = 1
        entity_id = sentence_id[start_index_id:start_index_id + entitylen]
        start_index_id += entitylen
        entity = ''.join([str(id2char[val].encode('utf-8')) for val in entity_id])
        if entity_labels[:5] in ['B-SYM']:
            entitys_SYM.append(entity)
        elif entity_labels[:5] in ['B-DIS']:
            entitys_DIS.append(entity)
        elif entity_labels[:5] in ['B-DRU']:
            entitys_DRU.append(entity)
        
        # replace the "BM*E" with "OO*O"
        tags = list(tags)
        tags[start_index_char: start_index_char + len(entity_labels)] = ['O' for i in range(len(entity_labels))]
        tags = ''.join(tags)
        m = re_entity.search(tags)
    return entitys_SYM, entitys_DIS, entitys_DRU


def iob2(tags):
    """
    检查标签序列是否为有效的IOB格式
    IOB1格式会被转化为IOB2格式
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        #if len(split) != 2 or split[0] not in ['I', 'B']:
            #return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    #return True


def iobes_iob(tags):
    """
    IOBES -> IOB2
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            print tag 
            raise Exception('Invalid format!')
    return new_tags


def extract_entityclass(predfile, dict):
    """
    从模型预测出来的文件中提取实体词的种类，并且经过人手工特征的筛选。
    将新词所在句子写入到html文件中，新词标红，
    并且标出了组成新词的word的词性，以逗号隔开。
    """
    entitys_SYM = []
    entitys_DIS = []
    entitys_DRU = []
    templines = []
    templabel = []
    tempword = []
    temppos = []
    tempwordlist = []
    line_number = 1
    is_all_in_dict = True
    is_contain_negative_word = False
    with codecs.open(predfile, 'r','utf-8') as f, codecs.open('data/new_entities_with_sentences_test_on_small_corpus.html', 'w', 'utf-8')as f1:
        f1.write('<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8" />\n </head>\n<body>\n')
        for index, item in enumerate(f.readlines()):
            if item != '\r\n':
                templines.append(item)
            else:
                for i,j in enumerate(templines):
                    word = j.split('\t')[0].strip()
                    pos = j.split('\t')[1].strip()
                    label = j.split('\t')[-1].strip()
                    templabel.append(label)
                    tempword.append(word)
                    temppos.append(pos)

                start_index_id = 0
                re_entity = re.compile(r'(B-(SYM|DIS|DRU)(I-(SYM|DIS|DRU))*)')
                templabel = iobes_iob(templabel)
                tags = ''.join([i for i in templabel])
                m = re_entity.search(tags)
                dif = 0
                temp = 0
                while m:
                    entity_labels = m.group()
                    start_index_char = tags.find(entity_labels)
                    dif = start_index_char - temp
                    start_index_id += dif
                    temp = start_index_char + len(entity_labels)
                    if len(entity_labels) > 6:
                        entitylen = entity_labels.count('I-') + 1
                    else:
                        entitylen = 1
                    entity = ''.join([i for i in tempword[start_index_id:start_index_id + entitylen]])

                    after_entity_str = ''.join(i for i in tempword[start_index_id + entitylen:start_index_id + entitylen + MATCH_RANGE_SHU])
                    before_entity_str = ''
                    for pos_index, pos_item in enumerate(reversed(temppos[start_index_id - MATCH_RANGE_PART:start_index_id])):
                        if pos_item == 'x':
                            before_entity_str = ''.join(i for i in tempword[start_index_id - pos_index:start_index_id])
                            break
                        else:
                            before_entity_str = ''.join(i for i in tempword[start_index_id - MATCH_RANGE_PART:start_index_id])
                    position_str = ''.join(i for i in tempword[start_index_id - MATCH_RANGE_PART:start_index_id])
                    match_index_shu = after_entity_str.find(u'术')
                    for part_item in part:
                        match_index_part = before_entity_str.find(part_item.decode('utf-8'))
                        if match_index_part >= 0:
                            if before_entity_str[match_index_part:] in dict:
                                before_entity_str = ''
                            entity = before_entity_str[match_index_part:] + entity
                            #print before_entity_str,entity 
                            break
                    if match_index_shu > 0:
                        entity = entity + after_entity_str[:match_index_shu+1]
                    pos = ''.join([i+',' for i in temppos[start_index_id:start_index_id + entitylen]])
                    if entitylen > 1:
                        for word in tempword[start_index_id:start_index_id + entitylen]:
                            if word not in dict:
                                is_all_in_dict = False
                    else:
                        is_all_in_dict = False

                    # if is_all_in_dict:
                    #     print entity 

                    if entity not in dict and tempword not in tempwordlist and pos not in pos_filter and is_all_in_dict == False:
                        tempwordlist.append(tempword)
                        f1.write(str(line_number)+'. ')
                        line_number += 1
                        f1.write(''.join([i for i in tempword[0:start_index_id]]))
                        f1.write('<b><font color="#FF0000">' + entity + pos + '</font></b>')
                        f1.write(''.join([i for i in tempword[start_index_id + entitylen:]]) + '<br />')

                    for negative_item in negative:
                        if entity.find(negative_item.decode('utf-8')) != -1:
                            is_contain_negative_word = True
                            break

                    start_index_id += entitylen
                    if pos not in pos_filter and is_all_in_dict == False and len(entity) != 1 and is_contain_negative_word == False:
                        if entity_labels[:5] in ['B-SYM']:
                            if entity not in entitys_SYM:
                                entitys_SYM.append(entity)
                        elif entity_labels[:5] in ['B-DIS']:
                            if entity not in entitys_DIS:
                                entitys_DIS.append(entity)
                        elif entity_labels[:5] in ['B-DRU']:
                            if entity not in entitys_DRU:
                                entitys_DRU.append(entity)

                    # replace the "BM*E" with "OO*O"
                    tags = list(tags)
                    tags[start_index_char: start_index_char + len(entity_labels)] = ['O' for i in range(len(entity_labels))]
                    tags = ''.join(tags)
                    m = re_entity.search(tags)

                templabel =[]
                tempword = []
                temppos = []
                templines = []
                is_all_in_dict = True
                is_contain_negative_word = False
        f1.write('</body>\n<html>\n')
    return entitys_SYM, entitys_DIS, entitys_DRU



if __name__ == '__main__':
    # 手工特征词典
    # part为身体部位词典，需要按字数倒序排列
    part = ['周围神经系统', '子宫内膜', '免疫系统', '男性生殖', '女性生殖', '血液血管', '支气管', '甲状腺', '肾上腺', '会阴部', '输精管', '上肢骨', '肠系膜', '输尿管', '前列腺', '其他骨', '输卵管', '下肢骨', '胰头', '锁骨', '胸骨', '截骨', '股骨', '胰腺', '腰部', '尿道', '脊柱', '大腿', '四肢', '淋巴', '食管', '胸部', '阴囊', '心理', '肌肉', '子宫', '肘部', '气管', '其他', '肩部', '阑尾', '足部', '皮肤', '肛门', '骨髓', '手部', '睾丸', '全身', '乳房', '臀部', '下肢', '上臂', '阴茎', '膀胱', '卵巢', '关节', '膝部', '颅骨', '颅脑', '咽喉', '前臂', '面部', '上肢', '外阴', '膈肌', '肋骨', '盆腔', '盆骨', '纵膈', '脊髓', '腹膜', '小腿', '心脏', '颈部', '腹部', '阴道', '头部', '结肠', '锁骨', '直肠', '肝胆', '胃肠', '开腹', '额顶', '腹', '脑', '鼻', '胃', '胆', '脾', '肺']
    position = ['左','右','前','甲','乙','丙']
    negative = ['未','不','伴']
    pos_filter = ['v,n,']    
    
    # 建立记录统计结果的文件
    f_result = codecs.open("result_"+opts.input, 'w', 'utf-8')
    
    # 读取字典文件
    dictlines_DIS = []
    with codecs.open("segment/dict/disease_union3.txt", 'r', 'utf-8') as f1:
        for i in f1.readlines():
            dictlines_DIS.append(i.split(' ')[0].strip())
    
    # 提取新词
    entityclass_SYM_true, entityclass_DIS_true, entityclass_DRU_true = extract_entityclass('data/test_ner.txt', dictlines_DIS)
    entityclass_SYM_pred, entityclass_DIS_pred, entityclass_DRU_pred = extract_entityclass(opts.input, dictlines_DIS)
    new_entities = set(entityclass_DIS_pred).difference(set(dictlines_DIS))
    
    # 将新词、未预测出的词以及新词种类的召回率写入统计结果文件
    f_result.write("New entities: " + str(len(new_entities))+'\r\n')
    for i in new_entities:
        f_result.write(i+'\r\n')
    f_result.write('\r\n\r\n' + "Missed entities:" + str(len(set(entityclass_DIS_true).difference(set(entityclass_DIS_pred)))) + '\r\n')
    for i in set(entityclass_DIS_true).difference(set(entityclass_DIS_pred)):
        f_result.write(i+'\r\n')
    f_result.write('\r\n'+"Hit disease type: " + str(len(set(entityclass_DIS_true)&set(entityclass_DIS_pred))) + "/" + str(len(entityclass_DIS_true))  + ", hit " + str(1.0*len(set(entityclass_DIS_true)&set(entityclass_DIS_pred))/len(set(entityclass_DIS_true))*100) + "%" + '\r\n')

    # 建立word、label与数字序号的映射表
    char2id, id2char, label2id, id2label = buildMap("data/train_ner.txt")

    # 从预测文件、测试文件中读取数据
    X, y_pred = getTest(opts.input,seq_max_len = 256)
    X, y_true = getTest("data/test_ner.txt",seq_max_len = 256)

    # 从预测文件、测试文件中提取实体、进行对比，以计算准确率、召回率和F1值
    entitys_SYM_true = []
    entitys_DIS_true = []
    entitys_DRU_true = []
    entitys_SYM_pred = []
    entitys_DIS_pred = []
    entitys_DRU_pred = []
    precision = -1.0
    recall = -1.0
    f1 = -1.0
    hit_num = 0
    pred_num = 0
    true_num = 0
    char2id, id2char = loadMap("char2id")
    label2id, id2label = loadMap("label2id")
    for i in range(len(y_true)):
        true_labels_SYM, true_labels_DIS, true_labels_DRU = extractEntityIOB2(id2char, X[i], id2label, y_true[i])
        pred_labels_SYM, pred_labels_DIS, pred_labels_DRU = extractEntityIOB2(id2char, X[i], id2label, y_pred[i])
        for i in true_labels_SYM:
            entitys_SYM_true.append(i)
        for i in true_labels_DIS:
            entitys_DIS_true.append(i)
        for i in true_labels_DRU:
            entitys_DRU_true.append(i)
        for i in pred_labels_SYM:
            entitys_SYM_pred.append(i)
        for i in pred_labels_DIS:
            entitys_DIS_pred.append(i)
        for i in pred_labels_DRU:
            entitys_DRU_pred.append(i)
        hit_num += len(set(true_labels_SYM)&set(pred_labels_SYM)) + len(set(true_labels_DIS)&set(pred_labels_DIS)) + len(set(true_labels_DRU)&set(pred_labels_DRU))
        pred_num += len(set(pred_labels_SYM)) + len(set(pred_labels_DIS))+ len(set(pred_labels_DRU))
        true_num += len(set(true_labels_SYM)) + len(set(true_labels_DIS)) + len(set(true_labels_DRU))

    if pred_num != 0:
        precision = 1.0 * hit_num / pred_num
    if true_num != 0:
        recall = 1.0 * hit_num / true_num
    if precision > 0 and recall > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    f_result.write('\r\nprecision: ' + str(precision) + '\r\nrecall: ' + str(recall) + '\r\nf1: ' + str(f1) + '\r\n')
    f_result.close()


    print "The result has been saved in: "+"result_"+opts.input

    #print collections.Counter(entitys_DIS_pred) 
    #wb = xlwt.Workbook()
    #ws = wb.add_sheet('Sheet1')
    #for index,item in enumerate(new_entities):
    #    ws.write(index, 0, item)
    #wb.save('data/new_entities_test_on_small_corpus.xls')
