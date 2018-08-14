import torch
import random
import collections

unkkey = '-unk-'
nullkey = '-NULL-'
paddingkey = '-padding-'
sep = 'SEP'
app = 'APP'

class instance():

    def __init__(self):

        self.words = []
        self.words_size = 0
        self.chars = []
        self.chars_size = 0
        self.bichars_left = []
        self.bichars_right = []
        self.bichars_size = 0

        self.gold = []
        self.pos = []
        self.gold_pos = []
        self.gold_seg = []
        self.gold_size = 0

        self.words_index = []

        self.chars_index = []
        self.bichars_left_index = []
        self.bichars_right_index = []

        self.static_chars_index = []
        self.static_bichars_left_index = []
        self.static_bichars_right_index = []

        self.pos_index = []
        self.gold_index = []


class Create_Alphabet():
    def __init__(self, min_freq=1, word_min_freq=1, char_min_freq=1, bichar_min_freq=1,):

        self.min_freq = min_freq
        self.word_min_freq = word_min_freq
        self.char_min_freq = char_min_freq
        self.bichar_min_freq = bichar_min_freq

        self.word_count = dict()
        self.char_count = dict()
        self.bichar_count = dict()
        self.pos_count = dict()

        self.word2idx = dict()
        self.idx2word = []
        self.char2idx = dict()
        self.idx2char = []
        self.bichar2idx = dict()
        self.idx2bichar = []

        self.pos2idx = dict()
        self.idx2pos = []
        self.label2idx = dict()
        self.idx2label = []

        self.word_UnkkID = self.loadWord2idAndId2Word(self.word2idx, self.idx2word, unkkey)
        self.char_UnkID = self.loadWord2idAndId2Word(self.char2idx, self.idx2char, unkkey)
        self.bichar_UnkID = self.loadWord2idAndId2Word(self.bichar2idx, self.idx2bichar, unkkey)
        self.pos_UnkID = self.loadWord2idAndId2Word(self.pos2idx, self.idx2pos, unkkey)

        # copy with the PaddingID
        self.word_PaddingID = self.loadWord2idAndId2Word(self.word2idx, self.idx2word, paddingkey)
        self.char_PaddingID = self.loadWord2idAndId2Word(self.char2idx, self.idx2char, paddingkey)
        self.bichar_PaddingID = self.loadWord2idAndId2Word(self.bichar2idx, self.idx2bichar, paddingkey)
        self.pos_PaddingID = self.loadWord2idAndId2Word(self.pos2idx, self.idx2pos, paddingkey)
        self.gold_PaddingID = self.loadWord2idAndId2Word(self.label2idx, self.idx2label, paddingkey)


        self.appID = self.loadWord2idAndId2Word(self.label2idx, self.idx2label, app)

        self.word_count[unkkey] = self.word_min_freq
        self.word_count[paddingkey] = self.word_min_freq
        self.char_count[unkkey] = self.char_min_freq
        self.char_count[paddingkey] = self.char_min_freq
        self.bichar_count[unkkey] = self.bichar_min_freq
        self.bichar_count[paddingkey] = self.bichar_min_freq
        self.pos_count[unkkey] = 1
        self.pos_count[paddingkey] = 1

    def createAlphabet(self, train_data=None, dev_data=None, test_data=None, debug_index=-1):
        print("start create Alphabet ...... ! ")
        assert train_data is not None

        datasets = []
        datasets.extend(train_data)
        print("the length of train data {}".format(len(datasets)))
        if dev_data is not None:
            print("the length of dev data {}".format(len(dev_data)))
            datasets.extend(dev_data)
        if test_data is not None:
            print("the length of test data {}".format(len(test_data)))
            datasets.extend(test_data)
        print("the length of data that create Alphabet {}".format(len(datasets)))

        for index, data in enumerate(datasets):
            # word
            for word, pos in zip(data.words, data.pos):
                if word not in self.word_count:
                    self.word_count[word] = 1
                    self.idx2word.append(word)
                    self.word2idx[word] = len(self.word2idx)
                else:
                    self.word_count[word] += 1

                if pos not in self.pos_count:
                    self.pos_count[pos] = 1
                    self.idx2pos.append(pos)
                    self.pos2idx[pos] = len(self.pos2idx)
                else:
                    self.pos_count[pos] += 1
            # char

            for char, bichar in zip(data.chars, data.bichars_left):
                if char not in self.char_count:
                    self.char_count[char] = 1
                    self.idx2char.append(char)
                    self.char2idx[char] = len(self.char2idx)
                else:
                    self.char_count[char] += 1

                if bichar not in self.bichar_count:
                    self.bichar_count[bichar] = 1
                    self.idx2bichar.append(bichar)
                    self.bichar2idx[bichar] = len(self.bichar2idx)
                else:
                    self.bichar_count[bichar] += 1
            bichar = data.bichars_right[-1]

            if bichar not in self.bichar_count:
                self.bichar_count[bichar] = 1
                self.idx2bichar.append(bichar)
                self.bichar2idx[bichar] = len(self.bichar2idx)
            else:
                self.bichar_count[bichar] += 1
            # label pos

            # copy with the gold "SEP#PN"
            for gold in data.gold:
                self.loadWord2idAndId2Word(self.label2idx, self.idx2label, gold)


        # copy with the unkID
        # copy the app seq ID

    def loadWord2idAndId2Word(self, x2idx, idx2x, data):
        if data in x2idx:
            return x2idx[data]
        else:
            idx2x.append(data)
            x2idx[data]=len(x2idx)
            return x2idx[data]


class Batch_Features:
    def __init__(self):

        self.batch_length = 0
        self.inst = None
        self.word_features = 0
        self.pos_features = 0
        self.char_features = 0
        self.bichar_left_features = 0
        self.bichar_right_features = 0
        self.static_char_features = 0
        self.static_bichar_left_features = 0
        self.static_bichar_right_features = 0
        self.gold_features = 0


class Eval:
    def __init__(self):
        self.predict_num = 0
        self.correct_num = 0
        self.gold_num = 0

        self.precision = 0
        self.recall = 0
        self.fscore = 0

    def clear(self):
        self.predict_num = 0
        self.correct_num = 0
        self.gold_num = 0

        self.precision = 0
        self.recall = 0
        self.fscore = 0

    def getFscore(self):
        if self.predict_num == 0:
            self.precision = 0
        else:
            self.precision = (self.correct_num / self.predict_num) * 100

        if self.gold_num == 0:
            self.recall = 0
        else:
            self.recall = (self.correct_num / self.gold_num) * 100

        if self.precision + self.recall == 0:
            self.fscore = 0
        else:
            self.fscore = 2 * (self.precision * self.recall) / (self.precision + self.recall)

        return self.precision, self.recall, self.fscore

    def acc(self):
        return self.correct_num / self.gold_num
