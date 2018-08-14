import itertools
import json

import collections
import os
import numpy as np
import codecs
import random
import torch.nn as nn
import torch.nn.init
import unicodedata
from torch.utils.data import Dataset
from model.Multi_Dataset import instance, Create_Alphabet, Batch_Features

unkkey = '-unk-'
nullkey = '-NULL-'
paddingkey = '-padding-'
sep = 'SEP'
app = 'APP'

class TransitionDataset_P(Dataset):

    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return len(self.data_tensor)

class TransitionDataset(Dataset):

    def __init__(self, sample, word_tensor, pos_tensor, char_tensor, bichar_left_tensor, bichar_right_tensor,
                 static_char_tensor, static_bichar_left_tensor, static_bichar_right_tensor, gold_tensor ):
        self.insts = sample
        self.word_tensor = word_tensor
        self.pos_tensor = pos_tensor
        self.char_tensor = char_tensor
        self.bichar_left_tensor = bichar_left_tensor
        self.bichar_right_tensor = bichar_right_tensor

        self.static_char_tensor = static_char_tensor
        self.static_bichar_left_tensor = static_bichar_left_tensor
        self.static_bichar_right_tensor = static_bichar_right_tensor

        self.gold_tensor = gold_tensor


zip = getattr(itertools, 'izip', zip)

def varible(tensor, gpu):
    if gpu >= 0 :
        return torch.autograd.Variable(tensor).cuda()
    else:
        return torch.autograd.Variable(tensor)


def xavier_init(gpu, *size):
    return nn.init.xavier_normal(varible(torch.FloatTensor(*size), gpu))


def init_varaible_zero(gpu, *size):
    return varible(torch.zeros(*size), gpu)

def to_scalar(var):

    return var.view(-1).data.tolist()[0]


def argmax(vec):

    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec, m_size):

    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
      
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M


def encode2char_safe(input_lines, char_dict):

    unk = char_dict['<u>']
    forw_lines = [list(map(lambda m: list(map(lambda t: char_dict.get(t, unk), m)), line)) for line in input_lines]
    return forw_lines


def encode_safe(input_lines, word_dict):

    lines = []
    for item in input_lines:
        if item in word_dict:
            lines.append(word_dict[item])
        else:
            lines.append(word_dict[unkkey])

    return lines

def encode_safe_predict(input_lines, word_dict, unk):
    lines = list(map(lambda t: list(map(lambda m: word_dict.get(m, unk), t)), input_lines))
    return lines

def encode(input_lines, word_dict):

    lines = list(map(lambda t: list(map(lambda m: word_dict[m], t)), input_lines))
    return lines


def read_corpus(lines, separator, shuffle=False):

    insts = []

    for index, line in enumerate(lines):
        # copy with "/n"
        line = line.strip()
        # init instance
        inst = instance()
        line = line.split(" ")
        count = 0
        for word_pos in line:
            # segment the word and pos in line
            split = word_pos.split(separator)
            if len(split) == 3:
                word = split[0]+'/'+split[1]
                label = split[2]
            elif len(split) == 2:
                word = split[0]
                label = split[1]
            else:
                print('err')
                continue
            word_length = len(word)
            inst.words.append(word)
            inst.gold_seg.append("[" + str(count) + "," + str(count + word_length) + "]")
            inst.gold_pos.append("[" + str(count) + "," + str(count + word_length) + "]" + label)
            count += word_length
            for i in range(word_length):
                inst.chars.append(word[i])
                if i == 0:
                    inst.gold.append('SEP' + "#" + label)
                    inst.pos.append(label)
                else:
                    inst.gold.append('APP')
        char_number = len(inst.chars)
        for i in range(char_number):
            # copy with the left bichars
            if i is 0:
                inst.bichars_left.append('-NULL-' + inst.chars[i])
            else:
                inst.bichars_left.append(inst.chars[i - 1] + inst.chars[i])
            # copy with the right bichars
            if i == char_number - 1:
                inst.bichars_right.append(inst.chars[i] + '-NULL-')
            else:
                inst.bichars_right.append(inst.chars[i] + inst.chars[i + 1])
        # char/word size
        inst.chars_size = len(inst.chars)
        inst.words_size = len(inst.words)
        inst.bichars_size = len(inst.bichars_left)
        inst.gold_size = len(inst.gold)
        # add one inst that represent one sentence into the list
        if not inst.words_size == len(inst.pos):
            print(inst.words_size)
        assert inst.words_size == len(inst.pos)
        insts.append(inst)
        if index == -1:
            break
    if shuffle is True:
        print("shuffle tha data......")
        random.shuffle(insts)
    # return all sentence in data
    # print(insts[-1].words)
    return insts

def read_corpus_perdict(lines):
    insts = []

    for index, line in enumerate(lines):
        # copy with "/n"
        line = line.strip()
        # init instance
        inst = instance()

        for char in line:
            inst.chars.append(char)

        char_number = len(inst.chars)
        for i in range(char_number):
            # copy with the left bichars
            if i is 0:
                inst.bichars_left.append('-NULL-' + inst.chars[i])
            else:
                inst.bichars_left.append(inst.chars[i - 1] + inst.chars[i])
            # copy with the right bichars
            if i == char_number - 1:
                inst.bichars_right.append(inst.chars[i] + '-NULL-')
            else:
                inst.bichars_right.append(inst.chars[i] + inst.chars[i + 1])
        # char/word size
        inst.chars_size = len(inst.chars)
        inst.bichars_size = len(inst.bichars_left)
        # add one inst that represent one sentence into the list
        insts.append(inst)
        if index == -1:
            break

    return insts


def generate_datesets(dataset_path, separator):
    insts_bucket = []
    for data_path in dataset_path:
        print("loading: ", data_path)
        with open(data_path, encoding="UTF-8") as f:
            lines = f.readlines()
            lines.sort(key=lambda x: len(x))
            insts = read_corpus(lines, separator)
            insts_bucket.append(insts)

    alphabet = Create_Alphabet()

    alphabet.createAlphabet(insts_bucket[0], insts_bucket[1], insts_bucket[2])
    static_alphabet = Create_Alphabet()
    static_alphabet.createAlphabet(insts_bucket[0],insts_bucket[1], insts_bucket[2])

    return insts_bucket[0],insts_bucket[1], insts_bucket[2], alphabet, static_alphabet

def create_one_batch(insts, alphabet, static_alphabet, batch_size, use_cuda, mode):
    max_word_size = -1
    max_char_size = -1
    for inst in insts:
        if mode == 't':
            if inst.words_size > max_word_size:
                max_word_size = inst.words_size
        if inst.chars_size > max_char_size:
            max_char_size = inst.chars_size

    if mode == 't':
        batch_word_features = []
        batch_pos_features = []

    batch_char_features = []
    batch_bichar_left_features = []
    batch_bichar_right_features = []

    batch_static_char_features = []
    batch_static_bichar_left_features = []
    batch_static_bichar_right_features = []

    batch_gold_features = []

    for inst in insts:
        if mode == 't':
            batch_word_features.append(inst.words_index + [alphabet.word2idx[paddingkey]] * (max_word_size - inst.words_size))
            batch_pos_features.append(inst.pos_index + [alphabet.pos2idx[paddingkey]] * (max_word_size - inst.words_size))

        batch_char_features.append(inst.chars_index + [alphabet.char2idx[paddingkey]] * (max_char_size - inst.chars_size))
        batch_bichar_left_features.append(inst.bichars_left_index + [alphabet.bichar2idx[paddingkey]] * (max_char_size - inst.bichars_size))
        batch_bichar_right_features.append(inst.bichars_right_index + [alphabet.bichar2idx[paddingkey]] * (max_char_size - inst.bichars_size))

        batch_static_char_features.append(inst.static_chars_index + [static_alphabet.char2idx[paddingkey]] * (max_char_size - inst.chars_size))
        batch_static_bichar_left_features.append(inst.static_bichars_left_index + [static_alphabet.bichar2idx[paddingkey]] * (max_char_size - inst.bichars_size))
        batch_static_bichar_right_features.append(inst.static_bichars_right_index + [static_alphabet.bichar2idx[paddingkey]] * (max_char_size - inst.bichars_size))

        batch_gold_features.extend(inst.gold_index + [alphabet.label2idx[paddingkey]] * (max_char_size - inst.gold_size))


    features = Batch_Features()
    features.batch_length = batch_size
    features.inst = insts

    if mode == 't':
        features.word_features = varible(torch.LongTensor(batch_word_features), use_cuda)
        features.pos_features = varible(torch.LongTensor(batch_pos_features), use_cuda)
    features.char_features = varible(torch.LongTensor(batch_char_features), use_cuda)
    features.static_char_features = varible(torch.LongTensor(batch_static_char_features), use_cuda)
    features.bichar_left_features = varible(torch.LongTensor(batch_bichar_left_features), use_cuda)
    features.static_bichar_left_features = varible(torch.LongTensor(batch_static_bichar_left_features),
                                                         use_cuda)
    features.bichar_right_features = varible(torch.LongTensor(batch_bichar_right_features), use_cuda)
    features.static_bichar_right_features = varible(torch.LongTensor(batch_static_bichar_right_features),
                                                          use_cuda)
    features.gold_features = varible(torch.LongTensor(batch_gold_features), use_cuda)

    return features

def create_batch(datasets, alphabet, static_alphabet, batch_size, use_cuda):

    batched_dataset =[[] for _ in range(len(datasets))]

    for index, dataset in enumerate(datasets):
        batch = []
        count_inst = 0
        for inst in dataset:

            inst.bichars_right_index = encode_safe(inst.bichars_right, alphabet.bichar2idx)
            inst.bichars_left_index = encode_safe(inst.bichars_left, alphabet.bichar2idx)
            inst.chars_index = encode_safe(inst.chars, alphabet.char2idx)

            inst.gold_index = encode_safe(inst.gold, alphabet.label2idx)
            inst.pos_index = encode_safe(inst.pos, alphabet.pos2idx)
            inst.words_index = encode_safe(inst.words, alphabet.word2idx)

            inst.static_bichars_right_index = encode_safe(inst.bichars_right, static_alphabet.bichar2idx)
            inst.static_bichars_left_index = encode_safe(inst.bichars_left, static_alphabet.bichar2idx)
            inst.static_chars_index = encode_safe(inst.chars, static_alphabet.char2idx)

            batch.append(inst)
            count_inst += 1
            if count_inst == batch_size or count_inst == len(dataset):
                one_batch = create_one_batch(batch, alphabet, static_alphabet, count_inst, use_cuda, mode='t')
                batched_dataset[index].append(one_batch)
                batch = []
                count_inst = 0

    return batched_dataset[0], batched_dataset[1], batched_dataset[2]


def create_batch_predict(dataset, alphabet, static_alphabet, batch_size, use_cuda):

    batched_dataset = []
    batch = []
    count_inst = 0
    for inst in dataset:

        inst.bichars_right_index = encode_safe(inst.bichars_right, alphabet.bichar2idx)
        inst.bichars_left_index = encode_safe(inst.bichars_left, alphabet.bichar2idx)
        inst.chars_index = encode_safe(inst.chars, alphabet.char2idx)

        inst.static_bichars_right_index = encode_safe(inst.bichars_right, static_alphabet.bichar2idx)
        inst.static_bichars_left_index = encode_safe(inst.bichars_left, static_alphabet.bichar2idx)
        inst.static_chars_index = encode_safe(inst.chars, static_alphabet.char2idx)

        batch.append(inst)
        count_inst += 1
        if count_inst == batch_size or count_inst == len(dataset):
            one_batch = create_one_batch(batch, alphabet, static_alphabet, count_inst, use_cuda, mode='p')
            batched_dataset.append(one_batch)
            batch = []
            count_inst = 0

    return batched_dataset

def generate_corpus_predict(data_path):

    print("loading: ", data_path)
    with open(data_path, encoding="UTF-8") as f:
        lines = f.readlines()
        # lines.sort(key=lambda x: len(x))
        insts = read_corpus_perdict(lines)


    return insts


def load_embedding_wlm(emb_file, delimiter, words2id, emb_len, shrink_to_train=False, shrink_to_corpus=False):

    print("loading embedding from :{}".format(emb_file))

    word_dict = dict()
    word_dict[unkkey] = 0
    word_dict[paddingkey] = 1


    rand_embedding_tensor = torch.FloatTensor(len(word_dict), emb_len)
    init_embedding(rand_embedding_tensor)

    indoc_embedding_array = list()
    indoc_word_array = [unkkey, paddingkey]
    outdoc_embedding_array = list()
    outdoc_word_array = list()

    for line in open(emb_file, 'r'):
        line = line.split(delimiter)
        if len(line) > 2:
            vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

            if shrink_to_train and line[0] not in words2id:
                continue

            if line[0] in words2id:
                indoc_embedding_array.append(vector)
                indoc_word_array.append(line[0])
                word_dict[line[0]] = len(word_dict)
            elif not shrink_to_corpus:
                outdoc_word_array.append(line[0])
                outdoc_embedding_array.append(vector)

    embedding_tensor_0 = torch.FloatTensor(np.asarray(indoc_embedding_array))
    if not shrink_to_corpus:
        embedding_tensor_1 = torch.FloatTensor(np.asarray(outdoc_embedding_array))
        word_emb_len = embedding_tensor_1.size(1)
        assert (word_emb_len == emb_len)
        for outdoc_word in outdoc_word_array:
            word_dict[outdoc_word] = len(word_dict)
        embedding_tensor = torch.cat([rand_embedding_tensor, embedding_tensor_0, embedding_tensor_1], 0)
        word_array = indoc_word_array + outdoc_word_array
    else:
        embedding_tensor = torch.cat([rand_embedding_tensor, embedding_tensor_0], 0)
        word_array = indoc_word_array

    print("iov count {}".format(len(indoc_word_array)))
    print("oov count {}".format(len(words2id) - len(indoc_word_array)))

    return embedding_tensor, word_array, word_dict


def construct_dataset_predict(input_features, word_dict, caseless):
    if caseless:
        input_features = list(map(lambda t: list(map(lambda x: x, t)), input_features))
    features = encode_safe_predict(input_features, word_dict, word_dict['<unk>'])
    feature_tensor = []
    for feature in features:
        feature_tensor.append(torch.LongTensor(feature))
    dataset = TransitionDataset_P(feature_tensor)

    return dataset


def save_checkpoint(state, track_list, filename):

    with open(filename+'.json', 'w') as f:
        json.dump(track_list, f)
    torch.save(state, filename+'.model')

def adjust_learning_rate(optimizer, lr):

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_embedding(input_embedding):

    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform(input_embedding, -bias, bias)

def init_linear(input_linear):

    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def init_lstm(input_lstm):

    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
    
    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def init_lstm_cell(input_lstm):

    weight = eval('input_lstm.weight_ih')
    bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    nn.init.uniform(weight, -bias, bias)
    weight = eval('input_lstm.weight_hh')
    bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    nn.init.uniform(weight, -bias, bias)

    if input_lstm.bias:
        weight = eval('input_lstm.bias_ih' )
        weight.data.zero_()
        weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        weight = eval('input_lstm.bias_hh')
        weight.data.zero_()
        weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def getMaxindex(decode_out_acc):
    # print("get max index ......")
    decode_out_list = decode_out_acc.data.tolist()
    return decode_out_list.index(max(decode_out_list))

def cal_train_acc(batch_features, train_eval, decoder_out, maxCharSize):
    # print("calculate the acc of train ......")
    train_eval.clear()
    pre_actions = []
    for id_batch in range(batch_features.batch_length):
        pre_action =[]
        inst = batch_features.inst[id_batch]
        for id_char in range(inst.chars_size):
            actionID = getMaxindex(decoder_out[id_batch * maxCharSize + id_char])
            pre_action.append(actionID)
            if actionID == inst.gold_index[id_char]:
                train_eval.correct_num += 1
        train_eval.gold_num += inst.chars_size
        pre_actions.append(pre_action)

    return pre_actions

def generate_ws_pos(char_sequence, action_sequence, idx2action):
    words = []
    pos = []
    for actionID in action_sequence:
        action = idx2action[actionID]
        if action.startswith("SEP"):
            words.append(char_sequence[actionID])
            pos.append(action.split("#"))
        else:
            words[-1] = words[-1]+char_sequence[actionID]

    return [words[id]+'/'+pos[id] for id in range(len(words))]
