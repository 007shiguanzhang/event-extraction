from data import DataLoad
from Batch import BatchGenerator
from model6 import Model
import tensorflow as tf
import numpy as np

np.set_printoptions(threshold=np.inf) 
def max_index(s):
        index_max = 0
        for ii in range(len(s)):
            if s[ii] > index_max:
                index_max = ii
        return index_max
if __name__ == '__main__':
    if_train = input("if train: (1/0)")
    if_train = True if if_train == str(1) else False
    data = DataLoad(if_train)
    length_words, length_tags = data.words
    data_train = None
    # 获取训练集
    if if_train:
        x_train, y_train = data.train
        data_train = BatchGenerator(x_train, y_train, shuffle=True)
        a, b = data_train.next_batch(10)
        print("train data ready")
    # 获取测试集
    x_test, y_test = data.test
    id2tag, id2word = data.id2tag, data.id2word
    id2 = (id2word, id2tag)
    # 构建输入数据类
    print('Creating the data generator ...')
    data_test = BatchGenerator(x_test, y_test, shuffle=False)
    print('Finished creating the data generator.')
    # building model
    model = Model(id2, length_words, length_tags, data_test, data_train, if_train)
