# 运行环境
1. python 2.7
2. tensorflow 1.0.1及以上

# 运行指令
1. 对语料进行预处理、分词、划分训练集、验证集、测试集
   python segment/seg_and_tag.py
2. 训练模型
   python train.py --train data/train_ner.txt --dev data/dev_ner.txt --test data/test_ner.txt
3. 用训练好的模型对测试语料进行标注
   python tagger.py --model models/lower=False,zeros=False,char_dim=0,char_lstm_dim=0,char_bidirect=False,word_dim=100,word_lstm_dim=100,word_bidirect=True,pos_dim=50,pos_lstm_dim=25,crf=True,dropout=0.5,lr_method=sgd,lr_rate=0.005,clip_norm=5.0,is_train=1,batch_size=32 --saver models/saver_pos50 --input data/test_sentence.txt --output aaa.txt
4. 统计预测结果
   python testcode.py --input aaa.txt
5. 用TensorBoard可视化数据(目前只对loss值做可视化，可视化一般操作流程在train.py的sess.run处，可视化变量loss声明在model.py最后)
   tensorboard --logdir=./models/loss_log/train_loss


# 文件目录
data: 数据文件夹。放置训练集、验证集、测试集数据，以及新词的html文件
	train_ner.txt: 训练集数据
	dev_ner.txt: 验证集数据
	test_ner.txt: 测试集数据
	test_sentence.txt: 适用于tagger.py输入格式的测试集数据，内容与test_ner.txt一样，只是数据格式不同
	new_entities_with_sentences_test_on_small_corpus.html: 由testcode.py生成，新词、词性标注及其所在句子的html页面，新词标红

models: 放置保存好的模型的文件夹
	saver: checkpoint文件及训练过程中保存过的模型参数
	以模型配置命名的model文件夹: 
		mappings.pkl: 标签、词语与数字id的映射
		parameters.pkl: 模型配置的变量值

segment: 分词、数据预处理相关文件
	dict: 词典文件夹。放置疾病、药品及症状词典
		disease_union3: 疾病词典
		drug.txt: 药品词典
		symptom.txt: 症状词典
		medicalrecord.csv: 原始csv语料文件，编码为utf-8
	seg_and_tag.py: 数据与处理、分词、划分数据集的脚本
	segment.py: jieba分词功能函数（加入了词典），用来做第一次分词，匹配出词典中的实体词
	segment_nodict.py: jieba分词功能函数（不加词典），用来做第二次分词，不加入词典对实体词进行切分
	剩余jiebaner.txt, jiebaner_union3.txt, jiebaner_union3_seg_again.txt, jiebapos.txt, jiebapos_union3.txt, jiebapos_union3_seg_again.txt多个文件: seg_and_tag.py产生的中间文件

train.py: 训练模型文件
tagger.py: 模型预测标注的文件，训练好模型后，使用此文件对测试数据进行标注
testcode.py: 模型预测标注后，对标注结果进行统计
model.py: 模型算法文件，建立了模型的各种结构
nn.py: 功能文件，定义了神经网络里各种结构，方便其他文件调用
loader.py: 对模型输入数据的数据预处理功能文件
utils.py: 对模型输入数据的数据预处理功能文件

char2id: 词与id序号对应的映射文件，由testcode.py产生
label2id: NER标签与id序号对应的映射文件，由testcode.py产生









