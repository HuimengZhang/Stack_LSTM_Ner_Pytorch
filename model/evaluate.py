import torch
import numpy as np
import itertools

import model.utils as utils
from model.Multi_Dataset import Eval


def calc_score(ner_model, dataset_loader, if_cuda):

    ner_model.eval()
    correct = 0
    total_act = 0
    for feature, label, action in itertools.chain.from_iterable(dataset_loader):  # feature : torch.Size([4, 17])
        fea_v, tg_v, ac_v = utils.repack_vb(if_cuda, feature, label, action)
        loss, pre_action = ner_model.forward(fea_v, ac_v)  # loss torch.Size([1, seq_len, action_size+1, action_size+1])
        for idx in range(len(pre_action)):
            if pre_action[idx] == ac_v.squeeze(0).data[idx]:
                correct += 1
        total_act += len(pre_action)

    acc = correct / float(total_act)

    return acc

def calc_f1_score(ner_model, batched_data, args, alphabet):

    ner_model.eval()
    gold_total_words = 0
    pre_total_words = 0
    ws_correct = 0
    pos_correct = 0
    test_eval = Eval()

    for batch_count, batch_features in enumerate(batched_data):

        decoder_out, _ = ner_model.forward(batch_features, mode='predict')  # loss torch.Size([1, seq_len, action_size+1, action_size+1])
        maxCharSize = batch_features.char_features.size()[1]
        pre_actions = utils.cal_train_acc(batch_features, test_eval, decoder_out, maxCharSize)

        gold_actions = batch_features.gold_features.view(args.batch_size, maxCharSize).data.tolist()
        for pre, gold in zip(pre_actions, gold_actions):
            gold_words_num, pre_words_num, correct_ws, correct_pos = to_cws_pos(gold, pre, alphabet.idx2label)
            gold_total_words += gold_words_num
            pre_total_words += pre_words_num
            ws_correct += correct_ws
            pos_correct += correct_pos

    test_eval.gold_num = gold_total_words
    test_eval.predict_num = pre_total_words
    test_eval.correct_num = ws_correct
    ws_pre, ws_rec, ws_f1 = test_eval.getFscore()
    test_eval.correct_num = pos_correct
    pos_pre, pos_rec, pos_f1 = test_eval.getFscore()
    return ws_pre, ws_rec, ws_f1, pos_pre, pos_rec, pos_f1

def to_cws_pos(real_action, predict_action, idx2action):
    flags = [False, False]
    wss = [[],[]]
    poss = [[],[]]
    actions = [real_action, predict_action]
    for idx in range(len(actions)):
        ws_start_pos = -1
        pos = None
        for ac_idx in range(len(actions[idx])):
            if idx2action[actions[idx][ac_idx]].startswith('S'):
                if ws_start_pos >= 0 and pos is not None:
                    wss[idx].append(str(ws_start_pos)+"-"+str(ac_idx-1))
                    poss[idx].append(str(ws_start_pos)+"-"+str(ac_idx-1)+"-"+pos)
                pos = idx2action[actions[idx][ac_idx]].split("#")[1]
                ws_start_pos = ac_idx
            elif idx2action[actions[idx][ac_idx]].startswith('-pad') and pos is not None:
                wss[idx].append(str(ws_start_pos) + "-" + str(ac_idx - 1))
                poss[idx].append(str(ws_start_pos) + "-" + str(ac_idx - 1) + "-" + pos)
                ws_start_pos = -1
                pos = None
        if ws_start_pos >= 0 and pos is not None:
            wss[idx].append(str(ws_start_pos)+"-"+str(ac_idx))
            poss[idx].append(str(ws_start_pos) + "-" + str(ac_idx) + "-" + pos)

    correct_ws = set(wss[0]) & set(wss[1])
    correct_pos = set(poss[0]) & set(poss[1])
    return len(wss[0]), len(wss[1]), len(correct_ws), len(correct_pos)


def generate_outcome(model, fileout, dataset_loader, alphabet):


    model.eval()


    for batch_count, batch_features in enumerate(dataset_loader):
        model.zero_grad()
        decoder_out, loss = model.forward(batch_features, mode='predict')
        maxCharSize = batch_features.char_features.size()[1]
        for id_batch in range(batch_features.batch_length):
            words = []
            pos = []
            for id_char in range(batch_features.inst[id_batch].chars_size):
                actionID = utils.getMaxindex(decoder_out[id_batch * maxCharSize + id_char])
                action = alphabet.idx2label[actionID]
                word = batch_features.inst[id_batch].chars[id_char]
                if action.startswith("SEP"):
                    words.append(word)
                    pos.append(action.split("#")[1])
                else:
                    words[-1] = words[-1] + word
            out = [words[id] + '/' + pos[id] for id in range(len(words))]
            print("%s\n" % (" ".join(out)))
            fileout.write("%s\n" % (" ".join(out)))