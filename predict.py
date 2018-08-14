from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
import model.utils as utils
import model.evaluate as evaluate
from model.Multi_Model import MultiModel

import argparse
import json
import os
import sys
from tqdm import tqdm
import itertools
import functools

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluating Stack-LSTM')
    parser.add_argument('--load_arg', default='./checkpoint/transition_3.22_ws_pos.json',help='arg json file path')
    parser.add_argument('--load_check_point', default='./checkpoint/transition_3.22_ws_pos.model', help='checkpoint path')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id')
    parser.add_argument('--test_file', default='test.txt', help='path to test file, if set to none, would use test_file path in the checkpoint file')
    parser.add_argument('--test_file_out', default='test_out.txt', help='path to test file output, if set to none, would use test_file path in the checkpoint file')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size (10)')
    parser.add_argument('--rnn_hidden_dim', type=int, default=200, help='hidden dimension for encoder LSTMCell')
    parser.add_argument('--hidden_size', type=int, default=200, help='output dimension for encoder')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--epoch', type=int, default=50, help='maximum epoch number')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch idx')
    parser.add_argument('--char_embedding_dim', type=int, default=100, help='dimension for char embedding')
    parser.add_argument('--bichar_embedding_dim', type=int, default=100, help='dimension for bichar embedding')
    parser.add_argument('--word_embedding_dim', type=int, default=100, help='dimension for word embedding')
    parser.add_argument('--pos_embedding_dim', type=int, default=100, help='dimension for pos embedding')
    parser.add_argument('--layers', type=int, choices=['1', '2'],  default=1, help='number of lstm layers')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.75, help='decay ratio of learning rate')
    parser.add_argument('--load_opt', action='store_true', help='load optimizer from ')
    parser.add_argument('--update', choices=['sgd', 'adam'], default='adam', help='optimizer method')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--clip_grad', type=float, default=10.0, help='grad clip at')

    args = parser.parse_args()

    with open(args.load_arg, 'r') as f:
        jd = json.load(f)
    jd = jd['args']
    jd['batch_size'] = args.batch_size

    checkpoint_file = torch.load(args.load_check_point, map_location=lambda storage, loc: storage)
    alphabet = checkpoint_file['alphabet']
    static_alphabet = checkpoint_file['static_alphabet']
    char_map = dict()
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    # load corpus
    with codecs.open(args.test_file, 'r', 'utf-8') as f:
        test_lines = f.readlines()


    # converting format
    test_features = utils.generate_corpus_predict(args.test_file)
    test_dataset = utils.create_batch_predict(test_features, alphabet, static_alphabet, jd['batch_size'], args.gpu)

    args.embed_char_num = jd['embed_char_num']
    args.embed_bichar_num = jd['embed_bichar_num']
    args.static_embed_char_num = jd['static_embed_char_num']
    args.static_embed_bichar_num = jd['static_embed_bichar_num']

    args.label_size = jd['label_size']
    args.pos_size = jd['pos_size']
    # build model
    print("building model:")
    model = MultiModel(args, alphabet, static_alphabet)

    print('loading static model')
    model.load_state_dict(checkpoint_file['state_dict'])

    if args.gpu >= 0:
        if_cuda = True
        torch.cuda.set_device(args.gpu)
        model.cuda()
    else:
        if_cuda = False
    file_out = codecs.open(args.test_file_out, "w+", encoding="utf-8")
    print("start generating:")
    evaluate.generate_outcome(model, file_out, test_dataset, alphabet)


