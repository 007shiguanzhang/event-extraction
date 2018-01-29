import codecs
import os
import pandas as pd
import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split

path = os.path.dirname(__file__)


class DataLoad(object):
    """
    loading data from test.in
    """

    def __init__(self, if_train=False):
        self._max_len = 0
        self.time_steps = time_steps = 24
        self.test_input = []
        self.all_words = []
        self.label_dic = ['O', 'B-SUB', 'M-SUB', 'E-SUB', 'B-OBJ', 'M-OBJ', 'E-OBJ', 'B-PRE', 'M-PRE', 'E-PRE',
                          'S-SUB', 'S-PRE', 'S-OBJ']
        # 模型训练的数据预处理
        train_x, train_label = [], []
        self.x_train2lstm, self.y_train2lstm = [], []
        data_test, data_train = None, None
        if if_train:
            path_train = u"train.in"
            # load data from train.in
            with codecs.open(os.path.join(path, path_train), 'r', "utf-8") as f:
                x, labes = [], []
                for line in f.readlines():
                    line_list = line.split()
                    if line_list:
                        if len(line_list) == 2:
                            x.append(line_list[0])
                            labes.append(line_list[1])
                    else:
                        train_x.append(x)
                        train_label.append(labes)
                        x, labes = [], []
            data_train = pd.DataFrame({"words": train_x, "tags": train_label}, index=range(len(train_x)))
            all_words = list(chain(*data_train['words'].values))
            self.all_words = pd.Series(all_words).value_counts()
            self.all_words = self.all_words.index
        else:
            # 从words.csv读取字典信息
            words_path = os.path.join(path, 'words.csv')
            dictionary = pd.read_csv(words_path, sep="\t", header=None, encoding="utf-8")
            self.all_words = dictionary[dictionary.columns[0]].values
            # loading data from test.in
            with codecs.open(os.path.join(path, u"test.in"), 'r', "utf-8") as f:
                t = []
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        for ch in line:
                            t.append(ch)
                    else:
                        self.test_input.append(t)
                        t = []
            data_test = pd.DataFrame({'test': self.test_input}, index=range(len(self.test_input)))

        set_ids = range(1, len(self.all_words) + 1)
        tag_ids = range(len(self.label_dic))
        # 3. 构建 words 和 tags 都转为数值 id 的映射
        self.word2id = pd.Series(set_ids, index=self.all_words)
        if if_train:
            new_words_path = "words-new.csv"
            self.word2id.to_csv(new_words_path, sep="\t", encoding="utf-8")
        self.id2word = pd.Series(self.all_words, index=set_ids)
        self.tag2id = pd.Series(tag_ids, index=self.label_dic)
        self.id2tag = pd.Series(self.label_dic, index=tag_ids)
        self.words = len(self.all_words) + 1, len(self.label_dic)

        def X_padding(words):
            """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
            ids = list()
            for w in words:
                if w in self.all_words:
                    ids.extend([self.word2id[w]])
                else:
                    ids.extend([0])
            if len(ids) <= time_steps:
                ids.extend([0] * (time_steps - len(ids)))
            else:
                ids.extend([0] * int((time_steps / 3) - len(ids) % (time_steps / 3)))
            return ids

        def y_padding(tags):
            """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
            ids = list(self.tag2id[tags])
            if len(ids) <= time_steps:
                ids.extend([0] * (time_steps - len(ids)))
            else:
                ids.extend([0] * int((time_steps / 3) - len(ids) % (time_steps / 3)))
            return ids

        if train_x:
            self.train_x = np.asarray(list(data_train["words"].apply(X_padding).values))
            self.train_label = np.asarray(list(data_train["tags"].apply(y_padding).values))
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.train_x, self.train_label, test_size=0.3,
                                                                                    random_state=42)
            self.x_train2lstm = self.split_train(self.x_train)
            self.y_train2lstm = self.split_train(self.y_train)
            self.x_test2lstm = []
            self.y_test2lstm = []
            for i in range(len(self.x_test)):
                if self.x_test[i]:
                    testdata = self.split_test(self.x_test[i])
                    self.x_test2lstm.append(testdata)
            for i in range(len(self.y_test)):
                if self.y_test[i]:
                    testdata = self.split_test(self.y_test[i])
                    self.y_test2lstm.append(testdata)
        else:
            data_test['x'] = data_test['test'].apply(X_padding)
            self.x_test = np.asarray(list(data_test['x'].values))
            # 输入语料生成
            self.x_test2lstm = []
            self.y_test2lstm = []
            for i in range(len(self.x_test)):
                testdata = self.split_test(self.x_test[i])
                self.x_test2lstm.append(testdata)
            print("句子总数：" + str(len(self.x_test)))

    @property
    def length(self):
        return self.words

    @property
    def test(self):
        return self.x_test2lstm, self.y_test2lstm

    @property
    def train(self):
        return self.x_train2lstm, self.y_train2lstm

    def split_test(self, inputs):
        result = []
        k = 0
        while k < len(inputs) - self.time_steps / 3:
            result.append(inputs[k:k + self.time_steps])
            k += int(self.time_steps / 3)
        result.pop()
        return result

    def split_train(self, inputs):
        result = []
        for ii in inputs:
            k = 0
            while k < len(ii) - self.time_steps:
                result.append(ii[k:k + self.time_steps])
                k += 1
            if result:
                result.pop()
            else:
                result = [ii + [0]*(self.time_steps-len(ii))]
        return result


if __name__ == "__main__":
    test_input = []


