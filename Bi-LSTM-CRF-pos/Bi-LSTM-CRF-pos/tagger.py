#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import time,codecs,optparse
from loader import prepare_sentence, prepare_sentence_, prepare_sentence_pos
from utils import create_input_batch, zero_digits
from model import Model
import tensorflow as tf

optparser = optparse.OptionParser()
# 训练好的模型所在路径
optparser.add_option(
    "-m", "--model", default="",
    help="Model location"
)
# 连接模型参数的checkpoint文件
optparser.add_option(
    "-s", "--saver", default="",
    help="tf checkpoint location"
)
# 输入文件，测试集数据，格式为一行一个句子，句子中词以空格隔开，每个词后跟着词性标签，以“————”分隔
optparser.add_option(
    "-i", "--input", default="",
    help="Input file location"
)
# 输出文件，模型对测试集进行预测，格式为一行一个单词，后面跟着预测出来的NER标签，以换行符分隔。不同句子之间以空行隔开
optparser.add_option(
    "-o", "--output", default="",
    help="Output file location"
)
# 此参数用来定义分隔符，暂时无用
optparser.add_option(
    "-d", "--delimiter", default="__",
    help="Delimiter to separate words from their tags"
)

opts = optparser.parse_args()[0]

# 检查参数是否有效
assert opts.delimiter
assert os.path.isdir(opts.model)
assert os.path.isdir(opts.saver)
assert os.path.isfile(opts.input)


def add_emptyline(truefile, predfile, outputfile):
    """
    由于模型输出的预测文件'noemptyline_'+opts.output中，句子之间没有空行，
    此函数与有空行的测试集文件data/test_ner.txt相比对，然后给预测文件opts.output添加空行
    """
    file = codecs.open(truefile,'r', 'utf-8')
    tempFile1 = file.readlines()
    file.close()

    file = codecs.open(predfile,'r', 'utf-8')
    tempFile2 = file.readlines()
    file.close()

    file = codecs.open(outputfile,"w", 'utf-8')
    temp_count = 0
    space_count = 0
    for i in range(len(tempFile1)):
        if tempFile1[i] == "\r\n": # in Linux it's \r\n, in Windows it's \n
            for j in range(temp_count,i):
                file.write('' + tempFile2[j-space_count].split('\t')[0] + '\t' + tempFile1[j].split('\t')[1] + '\t' + tempFile2[j-space_count].split('\t')[1].strip() +'\r\n')
            file.write('\r\n')
            temp_count = i+1
            space_count += 1
        else:
            pass

    file.close()


# 读取现有模型
print "Loading model..."
model = Model(model_path=opts.model)
parameters = model.parameters
parameters['is_train'] = 0
parameters['dropout'] = 0
batch_size = parameters['batch_size']
# 读取映射
word_to_id, char_to_id, tag_to_id, pos_tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag, model.id_to_pos]
]
tag_count = len(tag_to_id)
# 建立模型
cost, f_eval, _ = model.build(**parameters)


f_output = codecs.open('noemptyline_'+opts.output, 'w', 'utf-8')
start = time.time()
saver = tf.train.Saver()
print 'Tagging...'
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(opts.saver)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    test_data = []
    word_data = []
    with codecs.open(opts.input, 'r', 'utf-8') as f_input:
        for line in f_input:
            if line != '\r\n':
                words_ini = line.rstrip().split()
                # 若训练时选择将英文字母小写，此处将测试集所有英文字母小写
                if parameters['lower']:
                    line = line.lower()
                # 若训练时选择将数字全部替换为0,此处将测试集所有数字替换为0
                if parameters['zeros']:
                    line = zero_digits(line)

                words = []
                pos_tags = []
                words_and_pos_tags = line.rstrip().split()
                [words.append(i.split('___')[0]) for i in words_and_pos_tags]
                [pos_tags.append(i.split('___')[1]) for i in words_and_pos_tags]
                # 准备输入数据
                if parameters['char_dim']:
                    sentence = prepare_sentence(words, word_to_id, char_to_id,
                                                lower=parameters['lower'])
                else:
                    sentence = prepare_sentence_pos(words, pos_tags, word_to_id, pos_tag_to_id, lower=parameters['lower'])
                test_data.append(sentence)
                word_data.append(words)
                
            else:
                continue
    count = 0
    assert len(test_data) == len(word_data)
    while count < len(test_data):
        batch_data = []
        batch_words = []
        for i in xrange(batch_size):
            index = i + count
            if index >= len(test_data):
                break
            data = test_data[index]
            batch_data.append(test_data[index])
            batch_words.append(word_data[index])
        if len(batch_data) <= 0:
            break
        input_ = create_input_batch(batch_data, parameters)
        feed_dict_ = {}
        if parameters['char_dim']:
            feed_dict_[model.word_ids] = input_[0]
            feed_dict_[model.word_pos_ids] = input_[1]
            feed_dict_[model.char_for_ids] = input_[2]
            feed_dict_[model.char_rev_ids] = input_[3]
            feed_dict_[model.char_pos_ids] = input_[4]
        else:
            feed_dict_[model.word_ids] = input_[0]
            feed_dict_[model.word_pos_ids] = input_[1]
            feed_dict_[model.pos_ids] = input_[2]
	    
        f_scores = sess.run(f_eval, feed_dict=feed_dict_)
        # 解码
        if parameters['crf']:
            for x in xrange(len(batch_data)):
                f_score = f_scores[x]
                word_pos = input_[1][x] + 2 	# sentence length + 1
                y_pred = f_score[1:word_pos]	# tag ids, exclude the first and last transition tag
                words = batch_words[x]
                y_preds = [model.id_to_tag[pred] for pred in y_pred]
                assert len(words) == len(y_preds)
                # 输出预测的NER标签
                for w,y in zip(words, y_preds):
                    f_output.writelines(w+'\t'+y+"\r\n")
                #f_output.write('%s\n' % ' '.join('%s%s%s' % (w, opts.delimiter, y) for w, y in zip(words, y_preds)))
        else:
            f_score = f_scores.argmax(axis=-1)
            for x in xrange(len(batch_data)):
                word_pos = input_[1][x] + 1
                y_pred = f_score[x][0:word_pos]
                words = batch_words[x]
                y_preds = [model.id_to_tag[pred] for pred in y_pred]
                assert len(words) == len(y_preds)
                f_output.write('%s\n' % ' '.join('%s%s%s' % (w, opts.delimiter, y) for w, y in zip(words, y_preds)))
        count += len(batch_data)

print '---- %i lines tagged in %.4fs ----' % (count, time.time() - start)
f_output.close()
add_emptyline("data/test_ner.txt", 'noemptyline_'+opts.output, opts.output)
# 删除临时产生的无空行的测试文件
os.remove('noemptyline_'+opts.output)
