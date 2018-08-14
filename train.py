from __future__ import print_function
import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
from model.Multi_Model import MultiModel
from model.Multi_Dataset import Eval
import model.utils as utils
import model.evaluate as evaluate

import argparse
import json
import os
import sys
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training transition-based NER system')
    parser.add_argument('--rand_embedding', action='store_true', help='random initialize word embedding')
    parser.add_argument('--char_emb_file', default='/data/disk1/zhanghuimeng/Embeddings/chinesegigawordv5.char.skipngram.200d.txt',
                        help='path to pre-trained embedding')
    parser.add_argument('--bichar_emb_file', default='/data/disk1/zhanghuimeng/Embeddings/chinesegigawordv5.bichar.skipngram.200d.txt',
                        help='path to pre-trained embedding')
    parser.add_argument('--word_emb_file', default='../../Embeddings/word.embedding.structured_skipngram.100d.txt',
                        help='path to pre-trained embedding')
    parser.add_argument('--train_file', default='./ctb60/train.ctb60.pos.hwc', help='path to training file')
    parser.add_argument('--dev_file', default='./ctb60/dev.ctb60.pos.hwc', help='path to development file')
    parser.add_argument('--test_file', default='./ctb60/test.ctb60.pos.hwc', help='path to test file')
    # parser.add_argument('--train_file', default='./ctb60/train.ctb60.pos.hwc', help='path to training file')
    # parser.add_argument('--dev_file', default='./ctb60/dev.ctb60.pos.hwc', help='path to development file')
    # parser.add_argument('--test_file', default='./ctb60/test.ctb60.pos.hwc', help='path to test file')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size (10)')
    parser.add_argument('--separator', default='_', help='separators for different dataset')
    parser.add_argument('--checkpoint', default='./checkpoint/ctb60', help='path to checkpoint prefix')
    parser.add_argument('--rnn_hidden_dim', type=int, default=200, help='hidden dimension for encoder LSTMCell')
    parser.add_argument('--hidden_size', type=int, default=200, help='output dimension for encoder')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--epoch', type=int, default=50, help='maximum epoch number')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch idx')
    parser.add_argument('--caseless', default=False, help='caseless or not')
    parser.add_argument('--char_embedding_dim', type=int, default=200, help='dimension for char embedding')
    parser.add_argument('--bichar_embedding_dim', type=int, default=200, help='dimension for bichar embedding')
    parser.add_argument('--word_embedding_dim', type=int, default=100, help='dimension for word embedding')
    parser.add_argument('--pos_embedding_dim', type=int, default=50, help='dimension for pos embedding')
    parser.add_argument('--layers', type=int, choices=['1', '2'],  default=1, help='number of lstm layers')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--singleton_rate', type=float, default=0.0, help='initial singleton rate')
    parser.add_argument('--lr_decay', type=float, default=0.75, help='decay ratio of learning rate')
    parser.add_argument('--load_check_point', default='', help='path of checkpoint')
    parser.add_argument('--load_opt', action='store_true', help='load optimizer from ')
    parser.add_argument('--update', choices=['sgd', 'adam'], default='adam', help='optimizer method')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train', help='mode selection')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--clip_grad', type=float, default=10.0, help='grad clip at')
    parser.add_argument('--mini_count', type=float, default=1, help='thresholds to replace rare words with <unk>')
    parser.add_argument('--eva_matrix', choices=['a', 'fa'], default='fa', help='use f1 and accuracy or accuracy alone')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stop')
    parser.add_argument('--least_iters', type=int, default=50, help='at least train how many epochs before stop')
    parser.add_argument('--shrink_embedding', action='store_true',
                        help='shrink the embedding dictionary to corpus (open this if pre-trained embedding dictionary is too large, but disable this may yield better results on external corpus)')
    args = parser.parse_args()

    print('setting:')
    print(args)

    # load corpus
    print('loading corpus')

    train_insts, dev_insts, test_insts, alphabet, static_alphabet = utils.generate_datesets([args.train_file, args.dev_file, args.test_file], args.separator)

    if args.load_check_point:
        if os.path.isfile(args.load_check_point):
            print("loading checkpoint: '{}'".format(args.load_check_point))
            checkpoint_file = torch.load(args.load_check_point)
            args.start_epoch = checkpoint_file['epoch']
            f_map = checkpoint_file['f_map']
            l_map = checkpoint_file['l_map']
        else:
            print("no checkpoint found at: '{}'".format(args.load_check_point))
    else:
        print('constructing coding table')


        if not args.rand_embedding:
            print("char size: '{}'".format(len(alphabet.char2idx)))
            print("bichar size: '{}'".format(len(alphabet.bichar2idx)))
            print("word size: '{}'".format(len(alphabet.word2idx)))
            print("pos size: '{}'".format(len(alphabet.pos2idx)))
            print("action size: '{}'".format(len(alphabet.label2idx)))

            print('loading embedding')

            pretrain_char_embedding_tensor, idx2char, char2idx = utils.load_embedding_wlm(args.char_emb_file, ' ', static_alphabet.char2idx,
                                                                      args.char_embedding_dim,
                                                                      shrink_to_corpus=args.shrink_embedding)
            static_alphabet.idx2char = idx2char
            static_alphabet.char2idx = char2idx

            pretrain_bichar_embedding_tensor, idx2bichar, bichar2idx  = utils.load_embedding_wlm(args.bichar_emb_file, ' ', static_alphabet.bichar2idx,
                                                                             args.char_embedding_dim,
                                                                             shrink_to_corpus=args.shrink_embedding)
            static_alphabet.idx2bishar = idx2bichar
            static_alphabet.bichar2idx = bichar2idx

            print("char embedding size: '{}'".format(pretrain_char_embedding_tensor.shape[0]))
            # print("bichar embedding size: '{}'".format(pretrain_bichar_embedding_tensor.shape(1)))
            # print("char embedding size: '{}'".format())

    # construct dataset
    batched_train, batched_dev, batched_test = utils.create_batch([train_insts, dev_insts, test_insts], alphabet,
                                                                  static_alphabet, args.batch_size, args.gpu)

    args.embed_char_num = len(alphabet.char2idx)
    args.embed_bichar_num = len(alphabet.bichar2idx)
    args.static_embed_char_num = len(static_alphabet.char2idx)
    args.static_embed_bichar_num = len(static_alphabet.bichar2idx)

    args.label_size = len(alphabet.label2idx)
    args.pos_size = len(alphabet.pos2idx)

    # build model
    print('building model')
    model = MultiModel(args, alphabet, static_alphabet)

    if args.load_check_point:
        model.load_state_dict(checkpoint_file['state_dict'])
    else:
        if not args.rand_embedding:
            model.load_pretrained_embedding(pretrain_char_embedding_tensor, pretrain_bichar_embedding_tensor)

        print('random initialization')
        model.init_params()

    if args.update == 'sgd':
        optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum)
    elif args.update == 'adam':
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    if args.load_check_point and args.load_opt:
        optimizer.load_state_dict(checkpoint_file['optimizer'])


    if args.gpu >= 0:
        if_cuda = True
        print('device: ' + str(args.gpu))
        torch.cuda.set_device(args.gpu)
        model.cuda()
    else:
        if_cuda = False

    tot_length = len(batched_train)
    ws_best_f1 = float('-inf')
    pos_best_f1 = float('-inf')
    best_acc = float('-inf')
    track_list = list()
    start_time = time.time()
    epoch_list = range(args.start_epoch, args.start_epoch + args.epoch)
    patience_count = 0

    for epoch_idx, args.start_epoch in enumerate(epoch_list):

        epoch_loss = 0
        random.shuffle(batched_train)
        model.train()
        train_eval = Eval()
        for batch_count, batch_features in enumerate(batched_train):

            model.zero_grad()  # zeroes the gradient of all parameters
            # loss, _, _ = model.forward(fea_v, ac_v, mode='train')
            decoder_out, loss = model.forward(batch_features, mode= 'train')
            maxCharSize = batch_features.char_features.size()[1]
            _ = utils.cal_train_acc(batch_features, train_eval, decoder_out, maxCharSize)
            sys.stdout.write("\rbatch_count = [{}] , loss is {:.6f} , (correct/ total_num) = acc ({} / {}) = "
                             "{:.6f}%".format(batch_count + 1, loss.data[0], train_eval.correct_num,
                                              train_eval.gold_num, train_eval.acc() * 100))
            loss.backward()
            nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, model.parameters()), args.clip_grad)
            optimizer.step()
            epoch_loss += utils.to_scalar(loss)

        # update lr
        utils.adjust_learning_rate(optimizer, args.lr / (1 + (args.start_epoch + 1) * args.lr_decay))
        end_time = time.time()
        print("\ntrain time cost: ", end_time - start_time, 's')

        ws_dev_pre, ws_dev_rec, ws_dev_f1, pos_dev_pre, pos_dev_rec, pos_dev_f1 = evaluate.calc_f1_score(model, batched_dev, args, alphabet)

        if ws_dev_f1 > ws_best_f1:
            patience_count = 0
            ws_best_f1 = ws_dev_f1

            ws_test_pre, ws_test_rec, ws_test_f1, pos_test_pre, pos_test_rec, pos_test_f1 = evaluate.calc_f1_score(model, batched_test, args, alphabet)

            track_list.append(
                {'loss': epoch_loss, 'ws_dev_f1': ws_dev_f1, 'pos_dev_f1': pos_dev_f1, 'ws_test_f1': ws_test_f1, 'pos_test_f1': pos_test_f1})

            print(
                '(loss: %.4f, epoch: %d, WS: dev F1 = %.4f, dev pre = %.4f, dev rec = %.4f, F1 on test = %.4f, pre on test = %.4f, rec on test = %.4f), saving...' %
                (epoch_loss,
                 args.start_epoch,
                 ws_dev_f1,
                 ws_dev_pre,
                 ws_dev_rec,
                 ws_test_f1,
                 ws_test_pre,
                 ws_test_rec))

            print(
                '(loss: %.4f, epoch: %d, POS: dev F1 = %.4f, dev pre = %.4f, dev rec = %.4f, F1 on test = %.4f, pre on test = %.4f, rec on test = %.4f), saving...' %
                (epoch_loss,
                 args.start_epoch,
                 pos_dev_f1,
                 pos_dev_pre,
                 pos_dev_rec,
                 pos_test_f1,
                 pos_test_pre,
                 pos_test_rec))

            try:
                utils.save_checkpoint({
                    'epoch': args.start_epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'alphabet': alphabet,
                    'static_alphabet': static_alphabet,
                }, {'track_list': track_list,
                    'args': vars(args)
                    }, args.checkpoint + 'ws_pos')
            except Exception as inst:
                print(inst)

        else:
            patience_count += 1
            print('(loss: %.4f, epoch: %d, ws dev F1 = %.4f, pos dev F1 = %.4f)' %
                  (epoch_loss,
                   args.start_epoch,
                   ws_dev_f1,
                   pos_dev_f1))
            track_list.append({'loss': epoch_loss, 'ws_dev_f1': ws_dev_f1, 'pos_dev_f1':pos_dev_f1})


        print('epoch: ' + str(args.start_epoch) + '\t in ' + str(args.epoch) + ' take: ' + str(
            time.time() - start_time) + ' s')

        if patience_count >= args.patience and args.start_epoch >= args.least_iters:
            break

    # printing summary
    print('setting:')
    print(args)

    # log_file.close()
