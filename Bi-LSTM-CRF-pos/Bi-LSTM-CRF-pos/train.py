#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import datetime, optparse
import numpy as np
from collections import OrderedDict
from utils import create_input_batch
from utils import models_path, models_saver_path, evaluate, iobes_iob
from loader import word_mapping, char_mapping, tag_mapping, pos_mapping
from loader import update_tag_scheme
from loader import prepare_dataset, prepare_dataset_
from model import Model
import tensorflow as tf
import loader


eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")

# 从命令行中读取部分参数
optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="",
    help="Test set location"
)
# 是否将英文字母统一变为小写
optparser.add_option(
    "-l", "--lower", default="0",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
# 是否将所有数字统一替换为0
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)
# 字符向量的维度
optparser.add_option(
    "-c", "--char_dim", default="0",
    type='int', help="Char embedding dimension"
)
# 字符向量在LSTM隐藏层的维度
optparser.add_option(
    "-C", "--char_lstm_dim", default="0",
    type='int', help="Char LSTM hidden layer size"
)
# 是否使用双向的LSTM来训练字符向量
optparser.add_option(
    "-b", "--char_bidirect", default="0",
    type='int', help="Use a bidirectional LSTM for chars"
)
# 词向量维度
optparser.add_option(
    "-w", "--word_dim", default="100",
    type='int', help="Token embedding dimension"
)
# 词向量在LSTM隐藏层的维度
optparser.add_option(
    "-W", "--word_lstm_dim", default="100",
    type='int', help="Token LSTM hidden layer size"
)
# 是否使用双向的LSTM来训练词向量
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
# 词性标签向量的维度
optparser.add_option(
    "-p", "--pos_dim", default="50",
    type='int', help="POS embedding dimension"
)
# 词性标签向量在LSTM隐藏层的维度
optparser.add_option(
    "-P", "--pos_lstm_dim", default="25",
    type='int', help="POS LSTM hidden layer size"
)
# 是否使用CRF层
optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
# Dropout的概率，若为0则不使用Dropout
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
# 选择梯度下降方法(SGD, Adadelta, Adam..)
optparser.add_option(
    "-L", "--lr_method", default="sgd",
    help="Learning method (SGD, Adadelta, Adam..)"
)
# 设定学习率
optparser.add_option(
    "-R", "--lr_rate", default="0.005",
    type='float', help="learning rate"
)
# 设定clip_norm，用来防止梯度爆炸
optparser.add_option(
    "-i", "--clip_norm", default="5.0",
    type='float', help="The clipping ratio"
)
# 选择训练或测试，此处一直设为1即可
optparser.add_option(
    "-r", "--mode", default="1",
    type='int', help="1 for Train and 0 for Test"
)
# 设定batch的大小，越小模型效果越好，训练耗时越长
optparser.add_option(
    "-G", "--batch_size", default="2",
    type='int', help="batch size"
)
# 设定是否要将训练集中未出现过的词随机替换为unknown对应的id，统一训练
optparser.add_option(
    "-g", "--singleton", default="0",
    type='float', help=" whether it needs to replace singletons by the unknown word or not"
)
# 设定用整个训练集训练多少次，即多少个epoch
optparser.add_option(
    "-E", "--epoch", default="100",
    type='int', help="number of epochs over the training set"
)
# 设定每隔多少次迭代，测试一下当前模型的效果
optparser.add_option(
    "-F", "--freq", default="1000", # 1==1000 20==5000
    type='int', help="evaluate on dev every freq_eval steps"
)
# 设定使用CPU训练还是GPU训练
optparser.add_option(
    "-Z", "--gpu_no", default="-1",
    type='int', help="whether using the cpu or gpu"
)
opts = optparser.parse_args()[0]

# 模型参数转化
parameters = OrderedDict()
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['char_dim'] = opts.char_dim
parameters['char_lstm_dim'] = opts.char_lstm_dim
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['pos_dim'] = opts.pos_dim
parameters['pos_lstm_dim'] = opts.pos_lstm_dim
parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['lr_method'] = opts.lr_method
parameters['lr_rate'] = opts.lr_rate
parameters['clip_norm'] = opts.clip_norm
parameters['is_train'] = opts.mode
#parameters['update'] = opts.update_scheme
parameters['batch_size'] = opts.batch_size

# 检查参数是否合法
assert os.path.isfile(opts.train)
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0

if not os.path.exists(models_path):
    os.makedirs(models_path)
if not os.path.exists(models_saver_path):
    os.makedirs(models_saver_path)

# 初始化模型
model = Model(parameters=parameters, models_path=models_path)
print "Model location: %s" % model.model_path

# 取参数设定
lower = parameters['lower']
zeros = parameters['zeros']
batch_size = parameters['batch_size']

# 从训练集、验证集及测试集中以句子级别为单位读取数据，形成一个序列
train_sentences = loader.load_sentences(opts.train, lower, zeros)
dev_sentences = loader.load_sentences(opts.dev, lower, zeros)
test_sentences = loader.load_sentences(opts.test, lower, zeros)

# 选择标注规范(IOB / IOBES)
#update_tag_scheme(train_sentences, 'iobes')
#update_tag_scheme(dev_sentences, 'iobes')
#update_tag_scheme(test_sentences, 'iobes')


# 用训练集建立词/字/NER标签/词性标签的词典和映射
dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
dico_words_train = dico_words
id_to_char = {}
if opts.char_dim:
    dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)
dico_pos_tags, pos_tag_to_id, id_to_pos_tag = pos_mapping(train_sentences)

n_tag = len(id_to_tag)
n_pos_tag = len(id_to_pos_tag)

# Index data
if opts.char_dim:
    train_data = prepare_dataset(
        train_sentences, word_to_id, char_to_id, tag_to_id, lower
    )
    dev_data = prepare_dataset(
        dev_sentences, word_to_id, char_to_id, tag_to_id, lower
    )
    test_data = prepare_dataset(
        test_sentences, word_to_id, char_to_id, tag_to_id, lower
    )
else:
    train_data = prepare_dataset_(
        train_sentences, word_to_id, tag_to_id, pos_tag_to_id, lower
    )
    dev_data = prepare_dataset_(
        dev_sentences, word_to_id, tag_to_id, pos_tag_to_id, lower
    )
    test_data = prepare_dataset_(
        test_sentences, word_to_id, tag_to_id, pos_tag_to_id, lower
    )
print "%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data))

# 把映射保存到本地
print 'Saving the mappings to disk...'
model.save_mappings(id_to_word, id_to_char, id_to_tag, id_to_pos_tag)

# 建立模型
if opts.gpu_no < 0:
    with tf.device("/cpu:0"):
        cost, tags_scores, train_op = model.build(**parameters)
else:
    with tf.device("/gpu:" + str(opts.gpu_no)):
        cost, tags_scores, train_op = model.build(**parameters)

#
# 训练网络
#
singletons = None
if opts.singleton:
    singletons = set([word_to_id[k] for k, v in dico_words_train.items() if v == 1])

n_epochs = opts.epoch  
freq_eval = opts.freq  
count = 0
best_dev = -np.inf
best_test = -np.inf
saver = tf.train.Saver()
start_time_all = datetime.datetime.now()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in xrange(n_epochs):
        epoch_costs = []
        epoch_accus = []
        epoch_sentence = []
        print "Starting epoch %i..." % epoch
        permutation_index = np.random.permutation(len(train_data))
        train_data_count = 0
        start_time_epoch = datetime.datetime.now()
        token_count = 0.0
        while train_data_count <= len(permutation_index):
            batch_data = []
            start_time = datetime.datetime.now()
            for i in xrange(batch_size):
                count += 1
                index = i + train_data_count
                if index >= len(permutation_index):
                    index %= len(permutation_index)
                batch_data.append(train_data[permutation_index[index]])
            input_ = create_input_batch(batch_data, parameters, n_tag, True, singletons)
            feed_dict_ = {}
            if parameters['char_dim']:
                # 模型暂时没用到字符向量，所以此部分无词性标注的新代码
                assert len(input_) == 8
                feed_dict_[model.word_ids] = input_[0]
                feed_dict_[model.word_pos_ids] = input_[1]
                feed_dict_[model.char_for_ids] = input_[2]
                feed_dict_[model.char_rev_ids] = input_[3]
                feed_dict_[model.char_pos_ids] = input_[4]
                feed_dict_[model.tag_ids] = input_tag  = input_[5]
                feed_dict_[model.tag_id_trans] = input_[6]
                feed_dict_[model.tag_id_index] = input_[7]
            else:
                # 模型进入此条件，将数据喂入feed_dict
                assert len(input_) == 6
                feed_dict_[model.word_ids] = input_[0]      # 词的id序号
                feed_dict_[model.word_pos_ids] = input_[1]  # 词的位置序号
                feed_dict_[model.pos_ids] = input_[2]       # 词性标注的id序号
                feed_dict_[model.tag_ids] = input_tag = input_[3]   # NER标签的id序号
                feed_dict_[model.tag_id_trans] = input_[4]          # NER标签转移矩阵的id序号
                feed_dict_[model.tag_id_index] = input_[5]          # NER标签id索引的序号

            # 将feed_dict喂入网络，计算损失值cost、预测标签id序号序列f_scores
            new_cost, f_scores, _ = sess.run([cost, tags_scores, train_op], feed_dict=feed_dict_)


            # 将数据在tensorboard可视化
            merged = tf.summary.merge_all() # 将所有涉及可视化的操作merge
            writer = tf.summary.FileWriter('models/loss_log/train_loss') # 将需要可视化的数据记录在本地某路径
            result = sess.run(merged, feed_dict=feed_dict_) # 在session中run可视化的merge，fetch的地方填merge变量
            writer.add_summary(result,epoch) # 每个epoch向本地写入一次可视化数据，以便tensorboard实时更新数据变化


            accus_batch = []
            sentence_batch = []
            if parameters['crf']:
                # 若模型添加了CRF层（模型默认添加CRF）
                for x in xrange(batch_size):
                    f_score = f_scores[x]
                    word_pos = input_[1][x] + 2
                    y_pred = f_score[1:word_pos]
                    y_real = input_tag[x][0:(word_pos-1)]

                    correct_prediction = np.equal(y_pred, y_real)
                    accus = np.array(correct_prediction).astype(float).sum()
                    accus_mean = np.array(correct_prediction).astype(float).mean()
                    accus_batch.append(accus)
                    if accus_mean < 1.0:
                        sentence_batch.append(0.0)
                    else:
                        sentence_batch.append(1.0)
                    token_count += (input_[1][x] + 1)
                sentence_val = np.array(sentence_batch).astype(float).mean()
            else:
                # 若模型没有添加CRF层（模型一直默认添加CRF层，此条件内代码比较旧）
                y_preds = f_scores.argmax(axis=-1)
                y_reals = np.array(input_tag).astype(np.int32)
                for x in xrange(batch_size):
                    word_pos = input_[1][x] + 1
                    y_pred = y_preds[x][0:word_pos]
                    y_real = y_reals[x][0:word_pos]
                    correct_prediction = np.equal(y_pred, y_real)
                    accus = np.array(correct_prediction).astype(float).sum()
                    accus_mean = np.array(correct_prediction).astype(float).mean()
                    accus_batch.append(accus)
                    if accus_mean < 1.0:
                        sentence_batch.append(0.0)
                    else:
                        sentence_batch.append(1.0)
                    token_count += word_pos
                sentence_val = np.array(sentence_batch).astype(float).mean()
            
            # 得到更新后的预测值后，计算准确率，评估此次梯度下降效果
            epoch_costs.append(new_cost)
            epoch_accus.extend(accus_batch)
            epoch_sentence.append(sentence_val)
            end_time = datetime.datetime.now()
            cost_time = (end_time - start_time).seconds
            if train_data_count % freq_eval == 0 and train_data_count > 0:
                assert token_count != 0.0
                token_accus_freq = np.sum(epoch_accus) / token_count
                print "%i, cost average: %f, accuracy average: %f, sentence accuracy avg: %f, cost time: %i" % (train_data_count, np.mean(epoch_costs), token_accus_freq, np.mean(epoch_sentence), cost_time)
            if train_data_count % freq_eval == 0 and train_data_count > 0:
                dev_score, dev_sentence_score  = evaluate(sess, tags_scores, model, parameters, dev_data, n_tag)
                test_score, test_sentence_score = evaluate(sess, tags_scores, model, parameters, test_data, n_tag)
                print "Score on dev: %.5f" % dev_score
                print "Score on test: %.5f" % test_score
                if dev_score > best_dev:
                    # 若此时模型在验证集上效果比之前在验证集上最好的效果好，则保存此次模型的参数
                    best_dev = dev_score
                    print "New best score on dev."
                    print "Saving model to disk..."
                    saver.save(sess, os.path.join(models_saver_path, 'model.ckpt'), global_step=count)
                if test_score > best_test:
                    # 只是查看一下此时模型在测试集上的效果怎么样，不做任何操作
                    best_test = test_score
                    print "New best score on test."
            train_data_count += batch_size
        assert token_count != 0.0


        token_accus_epoch = np.sum(epoch_accus) / token_count
        end_time_epoch = datetime.datetime.now()
        cost_time_epoch = (end_time_epoch - start_time_epoch).seconds
        print "Epoch %i done. Average cost: %f, Average accuracy: %f, Average sentence: %f, Cost time: %i" % (epoch, np.mean(epoch_costs), token_accus_epoch, np.mean(epoch_sentence), cost_time_epoch)
end_time_all = datetime.datetime.now()
cost_time_a = (end_time_all - start_time_all).seconds
print "Epoch %i done. Cost time: %i" % (n_epochs, cost_time_a)
