#coding=utf-8
import numpy as np
import tensorflow as tf
import random
import pickle
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize


"""
'I'm super man'
tokenize:
['I', ''m', 'super','man' ]
"""
from nltk.stem import WordNetLemmatizer

"""
词形还原(lemmatizer)，即把一个任何形式的英语单词还原到一般形式，与词根还原不同(stemmer)，后者是抽取一个单词的词根。
"""

pos_file = 'C:/Users/THINKPAD/Desktop/MyProject/text/pos.txt'
neg_file = 'C:/Users/THINKPAD/Desktop/MyProject/text/neg.txt'
test_file = 'C:/Users/THINKPAD/Desktop/MyProject/text/test.txt'

# 创建词汇表
def create_lexicon(pos_file, neg_file):
    lex = []

    # 读取文件
    def process_file(f):
        lines = list(open(f, "r",encoding= 'utf-8').readlines())
        lex = []
        #print("############################################lines:",lines)
        for line in lines:
            words = word_tokenize(line.lower())
            lex += words
        return lex

    lex += process_file(pos_file)
    lex += process_file(neg_file)
    print("lex:",len(lex))
    lemmatizer = WordNetLemmatizer()
    lex = [lemmatizer.lemmatize(word) for word in lex]  # 词形还原 (cats->cat)

    word_count = Counter(lex)
    #print("word_count:",word_count)
    # 去掉一些常用词,像the,a and等等，和一些不常用词; 这些词对判断一个评论是正面还是负面没有做任何贡献
    lex = []
    for word in word_count:
        if word_count[word] < 1000 and word_count[word] > 10:  
            lex.append(word)  # 齐普夫定律-使用Python验证文本的Zipf分布 http://blog.topspeedsnail.com/archives/9546
    return lex

lex = create_lexicon(pos_file, neg_file)
'''
file = open('lex.txt')
lines = file.readlines()
lex=[]
for line in lines:
    lex.append(line)
'''
# lex里保存了情感词词典。

# 把每条评论转换为向量, 转换原理：
# 假设lex为['woman', 'great', 'feel', 'actually', 'looking', 'latest', 'seen', 'is'] 当然实际上要大的多
# 评论'i think this movie is great' 转换为 [0,1,0,0,0,0,0,1], 把评论中出现的字在lex中标记，出现过的标记为1，其余标记为0
def normalize_dataset(lex):
    dataset = []

    # lex:词汇表；review:评论；clf:评论对应的分类，[0,1]代表负面评论 [1,0]代表正面评论
    def string_to_vector(lex, review, clf):
        words = word_tokenize(line.lower())
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        #首先词向量全部置零
        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大
        return [features, clf]

    lines = list(open(pos_file, "r",encoding= 'utf-8').readlines())
    for line in lines:
        one_sample = string_to_vector(lex, line, [1, 0])  # [array([ 0.,  1.,  0., ...,  0.,  0.,  0.]), [1,0]]
        dataset.append(one_sample)
        #print("line:::",one_sample)
    lines = list(open(neg_file, "r",encoding= 'utf-8').readlines())
    for line in lines:
        one_sample = string_to_vector(lex, line, [0, 1])  # [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), [0,1]]]
        dataset.append(one_sample)

    #print("len_dataset:",len(dataset))
    return dataset
def test_dataset(lex):
    dataset_test = []

    # lex:词汇表；review:评论；clf:评论对应的分类，[0,1]代表负面评论 [1,0]代表正面评论
    def string_to_vector(lex, review):
        words = word_tokenize(line.lower())
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        #首先词向量全部置零
        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大
        #print("line::::",features)
        return [features]

    lines = list(open(test_file, "r",encoding= 'utf-8').readlines())
    for line in lines:
        one_sample = string_to_vector(lex, line)  # [array([ 0.,  1.,  0., ...,  0.,  0.,  0.]), [1,0]]
        dataset_test.append(one_sample)
    #print("len_dataset:",len(dataset))
    return dataset_test

dataset = normalize_dataset(lex)
dataset_test = test_dataset(lex)
random.shuffle(dataset)
random.shuffle(dataset_test)

# 取样本中的10%做为测试数据
test_size = int(len(dataset) * 0.1)

dataset = np.array(dataset)
dataset_test = np.array(dataset_test)



train_dataset = dataset[:-test_size]
test_dataset = dataset[-test_size:]

# Feed-Forward Neural Network
# 定义每个层有多少'神经元''
n_input_layer = len(lex)  # 输入层
print("n_input_layer",n_input_layer)

n_layer_1 = 1000
n_layer_2 = 1000
n_layer_3 = 800
n_layer_4 = 600
n_layer_5 = 400
n_layer_6 = 200

n_output_layer = 2  # 输出层


# 定义待训练的神经网络
def neural_network(data):
    # 定义第一层"神经元"的权重值和偏置量
    # 指定张量的shape。那个形状自动成为变量的shape
    # 初始化矩阵里的值都是随机数
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
                   'b_': tf.Variable(tf.random_normal([n_layer_1]))}
    # 定义第二层"神经元"的权重和biases
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
                   'b_': tf.Variable(tf.random_normal([n_layer_2]))}
    # 定义第三层"神经元"的权重和biases
    layer_3_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_layer_3])),
                   'b_': tf.Variable(tf.random_normal([n_layer_3]))}
    # 定义第四层"神经元"的权重和biases
    layer_4_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_3, n_layer_4])),
                   'b_': tf.Variable(tf.random_normal([n_layer_4]))}
    # 定义第五层"神经元"的权重和biases
    layer_5_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_4, n_layer_5])),
                   'b_': tf.Variable(tf.random_normal([n_layer_5]))}
    # 定义第六层"神经元"的权重和biases
    layer_6_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_5, n_layer_6])),
                   'b_': tf.Variable(tf.random_normal([n_layer_6]))}
    # 定义输出层"神经元"的权重和biases
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_6, n_output_layer])),
                        'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    # w·x+b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)  # 激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)  # 激活函数
    layer_3 = tf.add(tf.matmul(layer_2, layer_3_w_b['w_']), layer_3_w_b['b_'])
    layer_3 = tf.nn.relu(layer_3)  # 激活函数
    layer_4 = tf.add(tf.matmul(layer_3, layer_4_w_b['w_']), layer_4_w_b['b_'])
    layer_4 = tf.nn.relu(layer_4)  # 激活函数
    layer_5 = tf.add(tf.matmul(layer_4, layer_5_w_b['w_']), layer_5_w_b['b_'])
    layer_5 = tf.nn.relu(layer_5)  # 激活函数
    layer_6 = tf.add(tf.matmul(layer_5, layer_6_w_b['w_']), layer_6_w_b['b_'])
    layer_6 = tf.nn.relu(layer_6)  # 激活函数
    layer_output = tf.add(tf.matmul(layer_6, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output


# 每次使用50条数据进行训练
batch_size = 50

X = tf.placeholder('float', [None, len(train_dataset[0][0])])
Y = tf.placeholder('float')
#为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值：

#计算操作单元用于实现反向传播算法和AdamOptimizer。然后，它返回给你的只是一个单一的操作，
#当运行这个操作时，它用AdamOptimizer训练模型，微调变量，不断减少成本。
def train_neural_network(X, Y):
    predict = neural_network(X)#得到预测结果（通过神经网络）
    #print("predict:::",predict)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, Y))#得到损失函数
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)  #使用最优化算法来使损失函数值最小

    epochs = 32 #32次整体迭代
    #在一个Session里面启动模型，并且初始化变量
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        epoch_loss = 0
        i = 0
        random.shuffle(train_dataset)#数据集随机
        train_x = dataset[:, 0]#train_x
        train_y = dataset[:, 1]#标签
        text_x = test_dataset[:, 0]#测试数据
        text_y = test_dataset[:, 1]#标签
        #训练过程
        for epoch in range(epochs):
            while i < len(train_x):
                start = i
                end = i + batch_size#训练了一个batch大小

                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
                #feed_dict的作用是给使用placeholder创建出来的tensor赋值
                _, c = session.run([optimizer, cost_func], feed_dict={X: list(batch_x), Y: list(batch_y)})
                epoch_loss += c#一个epoch内一个batch的损失
                i += batch_size#下一个开始位置
            #print(epoch, ' : ', epoch_loss)#输出  所有数据后的一次迭代
            #循环的每个步骤中，都通过神经网络模型得到预测值，然后计算出交叉熵，
            #用优化算法AdamOptimizer来使交叉熵更小，通过微调变量。
        #tf.argmax能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
        #由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，
        #比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，
        #而 tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal 来检测
        #我们的预测是否真实标签匹配(索引位置一样表示匹配)。，
        #是就返回true,不是就返回false,这样得到一个boolean数组。
        #tf.cast将boolean数组转成int数组，
        #最后求平均值，得到分类的准确率
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('准确率: ', accuracy.eval({X: list(text_x), Y: list(text_y)}))
train_neural_network(X, Y)
