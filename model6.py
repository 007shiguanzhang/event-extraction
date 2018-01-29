import tensorflow as tf
import numpy as np
import random
import codecs
import os
import csv
import re
from collections import namedtuple
import time
# 用于将打印完全显示
np.set_printoptions(threshold=np.inf)


class Model(object):
    def __init__(self, id2, vocab_size, class_num, data_test, data_train=None, if_train=False):

        # hyperparameters
        self.lr = 0.001                  # learning rate
        self.testing_iters = data_test.num_examples  # test 病例上限
        self.batch_size = 100  # 训练批次大小
        self.batch_test_size = 1  # 测试病例选取参数
        self.vocab_size = vocab_size  # 样本中不同字的个数，处理数据的时候得到
        self.n_inputs = self.embedding_size = 64
        self.n_steps = 24                # time steps
        self.n_hidden_units = 128        # neurons in hidden layer
        self.n_classes = class_num              #

        self.id2word, self.id2tag = id2  # 标号解码

        path = os.path.dirname(__file__)
    # placeholder
        x = tf.placeholder(tf.int32, [None, self.n_steps])
        y = tf.placeholder(tf.int32, [None, self.n_steps])
        batch_sizes = tf.placeholder(tf.int32, [])

    # 对 weights biases 初始值的定义
        weights = {
            # shape (64, 128)
            'in': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units])),
            # shape (128*2, class_num)
            'out': tf.Variable(tf.random_normal([self.n_hidden_units * 2, self.n_classes]))
        }
        biases = {
            # shape (128, )
            'in': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ])),
            # shape (class_num, )
            'out': tf.Variable(tf.constant(0.1, shape=[self.n_classes, ]))
        }
    # 获取预测结果，计算损失函数，梯度下降
        pred = self.lstm(x, weights, biases, batch_sizes)
        output = tf.reshape(pred, [batch_sizes, -1, self.n_classes])
        self.seq_length = tf.convert_to_tensor(self.batch_size * [self.n_steps], dtype=tf.int32)
        self.log_likelihood, self.transition_params = \
            tf.contrib.crf.crf_log_likelihood(output, y, self.seq_length)
        self.loss = tf.reduce_mean(-self.log_likelihood)
        train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    # 按字的正确率统计
        y0 = tf.reshape(y, [-1])
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.cast(y0, tf.int64))
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        print('Finished creating the lstm model.')

    # initial
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        config.gpu_options.allow_growth = False
        with tf.Session(config=config) as sess:
            # sess.run(init)
            saver = tf.train.Saver(max_to_keep=1)
            ckpt_path = r"./my-test/test-model"
            ckpt_path_new = r"./my-new-test/test-model"
            best_event = []
            best_pred_results = []
            show_original = []
            # train过程
            if if_train:
                # model init
                sess.run(init)
                # 训练部分,在每次迭代训练后测试验证集的acc和recall,获得最佳结果模型并输出
                epochs = int(input("训练的最大epoch次数:\n"))
                # 保存最佳参数
                best_recall, best_acc, best_f = 0, 0, 0
                for epoch in range(epochs):
                    step = 0
                    # 迭代训练
                    while step * self.batch_size < data_train.num_examples:
                        batch_xs, batch_ys = data_train.next_batch(self.batch_size)
                        batch_y = np.asarray(batch_ys)
                        feed_dict = {
                            x: batch_xs,
                            y: batch_y,
                            batch_sizes: self.batch_size,
                        }
                        # 启动训练
                        sess.run([train_op], feed_dict=feed_dict)
                        step += 1
                    # 验证集acc, recall设置
                    step2 = 0
                    p_ave, r_ave, f_ave = 0, 0, 0
                    s = 0
                    result_error_label, result_correct_label, label_result_label = [0, 0, 0], [0, 0, 0], [0, 0, 0]
                    while step2 * self.batch_test_size < self.testing_iters:
                        batch_x_test, batch_y_test = data_test.next_batch(self.batch_test_size)
                        batch_x_test = np.asarray(batch_x_test[0])
                        batch_y_test = np.asarray(batch_y_test[0])
                        l = len(batch_x_test)
                        feed_dict = {
                            x: batch_x_test,
                            y: batch_y_test,
                            batch_sizes: l,
                        }
                        xx = self.combine(batch_x_test.reshape(-1))
                        word = self.id2word[xx]
                        yy = self.combine(batch_y_test.reshape(-1))
                        tag = self.id2tag[yy]
                        fetches_test = [output, self.transition_params]
                        scores, transition_params = sess.run(fetches_test, feed_dict)
                        result_test = []
                        for index, ii in enumerate(scores):
                            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(ii, transition_params)
                            result_test += viterbi_sequence
                        pred_result = self.combine(result_test)
                        pred_tag = self.id2tag[pred_result]
                        label_result, _ = self.extract_non(word.values, tag.values)
                        pred_result, result_irregular = self.extract_non(word.values, pred_tag.values)
                        result_error = [temp for temp in pred_result if temp not in label_result]
                        result_correct = []
                        for temp in pred_result:
                            for cont in label_result:
                                if self.judgement(str(temp[0]), str(cont[0])) and temp not in result_correct:
                                    result_correct.append(temp)
                                    continue
                        # 统计抽取的个数
                        if len(result_correct) != 0:
                            if result_error:
                                for l in result_error:
                                    result_error_label[self.label(l[1]) - 1] += 1
                            if result_correct:
                                for l in result_correct:
                                    result_correct_label[self.label(l[1]) - 1] += 1
                            if label_result:
                                for l in label_result:
                                    label_result_label[self.label(l[1]) - 1] += 1
                            s += 1
                            p = len(result_correct) / (len(result_correct) + len(result_error))
                            p_ave += p
                            r = len(result_correct) / len(label_result)
                            r_ave += r
                            f = p * r * 2 / (p + r)
                            f_ave += f
                        step2 += 1
                    # 计算模型的acc,recall和f值
                    p_ave = p_ave / s
                    r_ave = r_ave / s
                    f_ave = p_ave * r_ave * 2 / (p_ave + r_ave)
                    p_sub = result_correct_label[0] / (result_correct_label[0] + result_error_label[0])
                    r_sub = result_correct_label[0] / label_result_label[0]
                    p_pre = result_correct_label[1] / (result_correct_label[1] + result_error_label[1])
                    r_pre = result_correct_label[1] / label_result_label[1]
                    p_obj = result_correct_label[2] / (result_correct_label[2] + result_error_label[2])
                    r_obj = result_correct_label[2] / label_result_label[2]

                    print("train epoch:", epoch)
                    print("训练集：" + '\n' + "平均正确率：" + str(p_ave) + "  平均召回率：" + str(r_ave) + "\t" +
                          "F1值:" + str(f_ave) + '\n')
                    print("主语：平均正确率：" + str(p_sub) + "  平均召回率：" + str(r_sub) + '\n')
                    print("谓语：平均正确率：" + str(p_pre) + "  平均召回率：" + str(r_pre) + '\n')
                    print("宾语：平均正确率：" + str(p_obj) + "  平均召回率：" + str(r_obj) + '\n')
                    if r_ave > best_recall:
                        best_recall = r_ave
                        best_acc = p_ave
                        best_f = f_ave
                        saver.save(sess, ckpt_path_new)
                print("best acc:{0}, best recall:{1}".format(best_acc, best_recall))
            else:
                # 测试部分
                # load model
                saver.restore(sess, ckpt_path)
                print("restore success")
                # working
                step2 = 0
                while step2 * self.batch_test_size < self.testing_iters:
                    batch_x_test, batch_y_test = data_test.next_batch(self.batch_test_size)
                    batch_x_test = np.asarray(batch_x_test[0])
                    l = len(batch_x_test)
                    feed_dict = {
                        x: batch_x_test,
                        batch_sizes: l,
                    }
                    xx = self.combine(batch_x_test.reshape(-1))
                    word = self.id2word[xx]
                    fetches_test = [output, self.transition_params]
                    scores, transition_params = sess.run(fetches_test, feed_dict)
                    result_test = []
                    for index, ii in enumerate(scores):
                        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(ii, transition_params)
                        result_test += viterbi_sequence
                    pred_result = self.combine(result_test)
                    pred_tag = self.id2tag[pred_result]
                    pred_result, result_irregular = self.extract_non(word.values, pred_tag.values)
                    original = "".join([str(f) for f in word.values])
                    event, error = self.stitching(pred_result, original)
                    best_event.append(event)
                    best_pred_results.append(pred_result)
                    show_original.append(original)
                    step2 += 1
                print("start writing")
                # 输出至CSV中
                path_w = "test-result.csv"
                if os.path.exists(path_w):
                    os.remove(path_w)
                with codecs.open(path_w, 'a', 'utf-8') as f:
                    f_csv = csv.writer(f)
                    headers = ["SUB1", "PRE1", "OBJ1", "SUB2", "PRE2", "OBJ2", "seq"]
                    f_csv.writerow(headers)
                    for index, i in enumerate(best_event):
                        if i:
                            if i.__len__() > 2:
                                for index_e, event_con2 in enumerate(i):
                                    event_con1 = i[index_e-1]
                                    event_con1 = re.split(r"->", event_con1)
                                    event_con2 = re.split(r"->", event_con2)
                                    if event_con1 == event_con2:
                                        event_con2 = ["", "", ""]
                                    seq = show_original[index].rstrip("nan")
                                    csv_row = event_con1+event_con2+[seq]
                                    f_csv.writerow(csv_row)
                            elif i.__len__() == 2:
                                event_con1 = i[0]
                                event_con2 = i[1]
                                event_con1 = re.split(r"->", event_con1)
                                event_con2 = re.split(r"->", event_con2)
                                seq = show_original[index].rstrip("nan")
                                csv_row = event_con1 + event_con2 + [seq]
                                f_csv.writerow(csv_row)
                            else:
                                event_con1 = i[0]
                                event_con1 = re.split(r"->", event_con1)
                                event_con2 = ["", "", ""]
                                seq = show_original[index].rstrip("nan")
                                csv_row = event_con1 + event_con2 + [seq]
                                # f_csv.writerow(csv_row)
                            # 输出到txt时的格式
                            # f.write("原文:" + show_original[index] + "\n")
                            # f.write("事件:")
                            # f.write(str(i) + "\n")
                            # # f.write("人工抽取:" + str(label_events[index]) + "\n")
                            # # f.write("------------------" + "\n")
                            # f.write("模型抽取词语:" + "\n")
                            # for t in best_pred_results[index]:
                            #     f.write("".join(t[0]) + "\t"+self.restore(str(t[1]))+"\n")
                            # f.write("------------------" + "\n")
        # 测试

    # lstm模型网络结构，参数：输入，权重，偏置，批次大小
    def lstm(self, X, weights, biases, batch_sizes):
        # X ==> (10 batches * 64 inputs, 24 steps)
        # 随机生成embedding矩阵，然后通过lookup查找对应词向量
        #        embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32)
        embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], initializer=tf.uniform_unit_scaling_initializer())
        X = tf.nn.embedding_lookup(embedding, X)
        X = tf.cast(X, tf.float32)
    # X ==> (10 batches * 24 steps, 64 inputs)
        X = tf.reshape(X, [-1, self.n_inputs])
    # X_in = W*X + b
        X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (10 batches, 24 steps, 128 hidden) 换回3维
        X_in = tf.reshape(X_in, [-1, self.n_steps, self.n_hidden_units])
    # 使用 双向 basic LSTM Cell.
        fw_cell = tf.contrib.rnn.GRUCell(self.n_hidden_units, activation=tf.nn.relu6)
        bw_cell = tf.contrib.rnn.GRUCell(self.n_hidden_units, activation=tf.nn.relu6)
        # fw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        # bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # 初始化全零 state
        fw_init_state = fw_cell.zero_state(batch_sizes, dtype=tf.float32)
        bw_init_state = bw_cell.zero_state(batch_sizes, dtype=tf.float32)

        outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, X_in, initial_state_fw=fw_init_state,
                                                            initial_state_bw=bw_init_state, dtype=tf.float32)
        fw_out, bw_out = outputs
        outputs = tf.concat([fw_out, bw_out], axis=2)
        outputs = tf.reshape(outputs, [-1, self.n_hidden_units * 2])
        results = tf.matmul(outputs, weights['out']) + biases['out']

        return results

    # 将label转化为需要的形式
    def restore(self, m):
        d = {"SUB": "主语", "PRE": "谓语", "OBJ": "宾语"}
        return d[m]

    def label(self, m):
        tag_dic = [u'主语', u'谓语', u'宾语']
        result = 0
        if m == 'SUB':
            result = 1
        elif m == 'PRE':
            result = 2
        elif m == 'OBJ':
            result = 3
        return result

    # 合并函数
    def combine(self, y):
        r = []
        result = []
        i = 0
        while i < len(y):
            r.append(y[i:i + 24])
            i += 24
    # 计算原病例长度
        l = 8 * (len(r) - 1) + 24
    # 开始合并
        if l == 24:
            rerult = r[0]
        else:
            for ii in range(l):
                # 前8位无重叠
                if ii < 8:
                    result.append(r[0][ii])
    # 8-16位有两组相同
                elif ii < 16:
                    m = [r[int(ii / 8) - 1][8 + int(ii % 8)], r[int(ii / 8)][int(ii % 8)]]
                    if m[0] == m[1]:
                        result.append(m[0])
                    else:
                        if m[0] * m[1] != 0:
                            result.append(m[random.randint(0, 1)])
                        else:
                            if m[0] != 0:
                                result.append(m[0])
                            else:
                                result.append(m[1])
    # 16-(l-16)有三组相同
                elif ii < l - 16:
                    n = [r[int(ii / 8) - 2][16 + int(ii % 8)], r[int(ii / 8) - 1][8 + int(ii % 8)], r[int(ii / 8)][int(ii % 8)]]
                    if n[0] * n[1] * n[2] != 0:
                        if n[0] == n[1]:
                            result.append(n[0])
                        elif n[1] == n[2]:
                            result.append(n[1])
                        elif n[0] == n[2]:
                            result.append(n[0])
                        else:
                            result.append(n[random.randint(0, 2)])
                    else:
                        if n[0] == 0:
                            if n[1] == 0:
                                result.append(n[2])
                            else:
                                if n[2] == 0:
                                    result.append(n[1])
                                else:
                                    result.append(n[random.randint(1, 2)])
                        else:
                            if n[1] == 0:
                                if n[2] == 0:
                                    result.append(n[0])
                                else:
                                    result.append(n[2 * random.randint(0, 1)])
                            else:
                                result.append(n[random.randint(0, 1)])
    # (l-16)-(l-8)有两组相同
                elif ii < l - 8:
                    m = [r[int(ii / 8) - 2][16 + int(ii % 8)], r[int(ii / 8) - 1][8 + int(ii % 8)]]
                    if m[0] == m[1]:
                        result.append(m[0])
                    else:
                        if m[0] * m[1] != 0:
                            result.append(m[random.randint(0, 1)])
                        else:
                            if m[0] != 0:
                                result.append(m[0])
                            else:
                                result.append(m[1])
    # (l-8)-l无重叠
                else:
                    result.append(r[int(ii / 8) - 2][16 + int(ii % 8)])
        return result

    # label 的one-hot
    def embedding_y(self, label):
        result = []
        for i in label:
            for j in i:
                z1 = [0] * self.n_classes
                z1[int(j)] = 1
                result.append(z1)
        return result

    # 获得完整的标注 即B-E或者S
    def extract_non(self, words, labels):
        i = 0
        words, labels = list(words), list(labels)
        temp, result, index_correct, result_irregular = [], [], [], []
        while i < len(labels):
            if labels[i][0] == 'S':
                result.append([words[i], labels[i][-3:], i])
                index_correct.append(i)
                i += 1
            elif labels[i][0] == 'B':
                k = i + 1
                if k >= len(labels):
                    break
                temp += [labels[i][0], labels[k][0]]
                while labels[k] != 'O' and labels[k][-3:] == labels[i][-3:] and labels[k][0] == "M":
                    k += 1
                    if k < len(labels):
                        temp += labels[k][0]
                    else:
                        break
                if temp == ['B'] + ['M'] * (len(temp) - 2) + ['E'] or temp == ['B', 'E']:
                    result.append([words[i:k + 1], labels[i][-3:], i])
                    index_correct.extend(list(range(i, k + 1)))
                temp = []
                i = k + 1
            else:
                i += 1
        ii = 0
        while ii < len(labels):
            if ii not in index_correct and labels[ii] != 'O':
                forwad, backward = ii, ii
                fw_temp, bw_temp = 3, 3
                while forwad - 1 >= 0 and forwad - 1 not in index_correct:
                    forwad -= 1
                    if labels[forwad] != 'O' and labels[forwad][-3:] == labels[ii][-3:]:
                        fw_temp += 1
                    if ii - forwad > fw_temp:
                        break
                while backward + 1 < len(labels) and backward + 1 not in index_correct:
                    backward += 1
                    if labels[backward] != 'O' and labels[backward][-3:] == labels[ii][-3:]:
                        bw_temp += 1
                    if backward - ii > bw_temp:
                        break
                result_irregular.append([words[forwad:backward + 1], labels[forwad:backward + 1]])
                ii = backward + 1
            else:
                ii += 1
        return result, result_irregular

    # 按照主谓宾的相对顺序组合事件，
    def stitching(self, pred_result, seq):
        i, k = 0, 0
        pred_result.sort(key=lambda k: k[2])
        length = len(pred_result)
        event = []
        error = 0
        threshold = 6
        while i < length:
            if pred_result[i][1] == "SUB":
                if i + 2 < length and pred_result[i + 1][1] == "PRE" and pred_result[i + 2][1] == "OBJ":
                    temp = [pred_result[i][0], pred_result[i + 1][0], pred_result[i + 2][0]]
                    if i-1 > 0 and pred_result[i-1][1] == "SUB" and abs(pred_result[i][2] - pred_result[i-1][2] - len(pred_result[i-1][0])) < threshold:
                        # temp = [list(pred_result[i - 1][0])+list(pred_result[i][0]), pred_result[i + 1][0],
                        # pred_result[i + 2][0]]
                        temp = [seq[pred_result[i-1][2]:pred_result[i][2]+len(pred_result[i][0])], pred_result[i + 1][0], pred_result[i + 2][0]]
                    if i+3 < length and pred_result[i+3][1] == "OBJ" and abs(pred_result[i+3][2]-pred_result[i+2][2]-len(pred_result[i+2][0]))<threshold:
                        # temp = [pred_result[i][0], pred_result[i + 1][0], list(pred_result[i+2][0])+list(pred_result[i+3][0])]
                        temp = [pred_result[i][0], pred_result[i + 1][0], seq[pred_result[i+2][2]:pred_result[i+3][2]+len(pred_result[i+3][0])]]
                    temps = []
                    for tem in temp:
                        try:
                            co = "".join(list(tem))
                            temps.append(co)
                        except :
                            co = "".join(str(temp))
                            temps.append(co)
                    event.append("->".join(temps))
                    i = i + 3
                    continue
                if i + 1 < len(pred_result) and pred_result[i + 1][1] == "PRE":
                    temp = [pred_result[i][0], pred_result[i + 1][0], ""]
                    if i-1 > 0 and pred_result[i-1][1] == "SUB" and abs(pred_result[i][2]-pred_result[i-1][2]-len(pred_result[i-1][0])) < threshold:
                        temp = [seq[pred_result[i-1][2]:pred_result[i][2]+len(pred_result[i][0])], pred_result[i + 1][0], ""]
                    temps = []
                    for tem in temp:
                        try:
                            co = "".join(list(tem))
                            temps.append(co)
                        except:
                            co = "".join(str(temp))
                            temps.append(co)
                    event.append("->".join(temps))
                    i += 2
                    continue
                if i + 1 < len(pred_result) and pred_result[i + 1][1] == "OBJ":
                    temp = [pred_result[i][0], "", pred_result[i + 1][0]]
                    if i-1 > 0 and pred_result[i-1][1] == "SUB" and abs(pred_result[i][2] - pred_result[i-1][2]-len(pred_result[i-1][0]))<threshold:
                        temp = [seq[pred_result[i-1][2]:pred_result[i][2]+len(pred_result[i][0])], "", pred_result[i + 1][0]]
                    if i + 2 < length and pred_result[i+2][1] == "OBJ" and abs(pred_result[i+2][2] - pred_result[i+1][2]-len(pred_result[i+1][0]))<threshold:
                        temp = [pred_result[i][0], "", seq[pred_result[i+1][2]:pred_result[i+2][2]+len(pred_result[i+2][0])]]
                    temps = []
                    for tem in temp:
                        try:
                            co = "".join(list(tem))
                            temps.append(co)
                        except:
                            co = "".join(str(temp))
                            temps.append(co)
                    event.append("->".join(temps))
                    i += 2
                    continue
            if pred_result[i][1] == "PRE":
                if i + 1 < length and pred_result[i + 1][1] == "OBJ":
                    temp = ["", pred_result[i][0], pred_result[i + 1][0]]
                    if i + 2 < length and pred_result[i+2][1] == "OBJ" and abs(pred_result[i+2][2] - pred_result[i+1][2]-len(pred_result[i+1][0]))<threshold:
                        temp = ["", pred_result[i][0], seq[pred_result[i+1][2]:pred_result[i+2][2]+len(pred_result[i+2][0])]]
                    temps = []
                    for tem in temp:
                        try:
                            co = "".join(list(tem))
                            temps.append(co)
                        except:
                            co = "".join(str(temp))
                            temps.append(co)
                    event.append("->".join(temps))
                    i += 2
                    continue
            i += 1
            error += 1
        return list(set(event)), error

    # 判断是否匹配
    @staticmethod
    def judgement(ev1, ev2):
        correct = list(set(ev1) & set(ev2)).__len__()
        if correct / len(ev2) >= 0.5:
            return True
        else:
            return False

    def judge(self, ev_list1, ev_list2):
        count = 0
        ev_list2 = ["".join(ev2.split("->")) for ev2 in ev_list2]
        for ev1 in ev_list1:
            for ev2 in ev_list2:
                if self.judgement(ev1, ev2):
                    count += 1
        return count


