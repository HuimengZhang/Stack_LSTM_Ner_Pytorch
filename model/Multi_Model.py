# coding=utf-8
import torch.nn
import torch.nn as nn
import  torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import random
import time

import model.utils as utils

unkkey = '-unk-'
nullkey = '-NULL-'
paddingkey = '-padding-'
sep = 'SEP'
app = 'APP'

class MultiModel(nn.Module):

    def __init__(self, args, alphabet, static_alphabet):
        super(MultiModel, self).__init__()
        self.args = args
        self.alphabet = alphabet
        self.static_alphabet = static_alphabet
        # Encode
        # ------------------------------------------------------------------
        # random
        self.char_embed = nn.Embedding(self.args.embed_char_num, self.args.char_embedding_dim, padding_idx=self.alphabet.char_PaddingID)
        self.char_embed.weight.requires_grad = True

        self.bichar_embed = nn.Embedding(self.args.embed_bichar_num, self.args.bichar_embedding_dim, padding_idx=self.alphabet.bichar_PaddingID)
        self.bichar_embed.weight.requires_grad = True

        # fix the word embedding
        self.static_char_embed = nn.Embedding(self.args.static_embed_char_num, self.args.char_embedding_dim, padding_idx=self.static_alphabet.char_PaddingID)
        self.static_char_embed.weight.requires_grad = False

        self.static_bichar_embed = nn.Embedding(self.args.static_embed_bichar_num, self.args.bichar_embedding_dim, padding_idx=self.static_alphabet.bichar_PaddingID)
        self.static_bichar_embed.weight.requires_grad = False

        self.lstm_left = nn.LSTMCell(input_size=self.args.hidden_size, hidden_size=self.args.rnn_hidden_dim, bias=True)
        self.lstm_right = nn.LSTMCell(input_size=self.args.hidden_size, hidden_size=self.args.rnn_hidden_dim, bias=True)

        self.input_dim = (self.args.char_embedding_dim + self.args.bichar_embedding_dim) * 2
        self.liner = nn.Linear(in_features=self.input_dim, out_features=self.args.hidden_size, bias=True)

        self.dropout = nn.Dropout(self.args.dropout)
        self.dropout_embed = nn.Dropout(0.25)
        # ------------------------------------------------------------------

        self.word_pos_lstmcell = nn.LSTMCell(input_size=self.args.hidden_size, hidden_size=self.args.rnn_hidden_dim, bias=True)
        self.pos_embed = nn.Embedding(num_embeddings=self.args.pos_size, embedding_dim=self.args.pos_embedding_dim)
        self.pos_embed.weight.requires_grad = True
        self.lstm_2_output = nn.Linear(in_features=self.args.rnn_hidden_dim * 2 + self.args.hidden_size,
                                out_features=self.args.label_size, bias=False)

        self.lstm_initial = self.rand_init_rnn_hidden(1)

        self.char_2_rnn_input = nn.Linear(in_features=self.args.rnn_hidden_dim * 2 + self.args.pos_embedding_dim,
                                        out_features=self.args.hidden_size, bias=True)

        self.softmax = nn.LogSoftmax()

    def init_params(self):


        utils.init_embedding(self.char_embed.weight)
        utils.init_embedding(self.bichar_embed.weight)
        utils.init_embedding(self.pos_embed.weight)

        utils.init_lstm_cell(self.lstm_left)
        utils.init_lstm_cell(self.lstm_right)
        utils.init_lstm_cell(self.word_pos_lstmcell)

        utils.init_linear(self.liner)
        utils.init_linear(self.lstm_2_output)
        utils.init_linear(self.char_2_rnn_input)

    def rand_init_rnn_hidden(self, batch_size):

        return utils.varible(
                torch.randn(batch_size, self.args.rnn_hidden_dim), self.args.gpu), utils.varible(
                torch.randn(batch_size, self.args.rnn_hidden_dim), self.args.gpu)

    def load_pretrained_embedding(self, pre_char_word_embedding=None, pre_bichar_word_embedding=None):

        # load external word embedding
        # self.word_embeds.weight = nn.Parameter(pre_embeddings)
        if pre_char_word_embedding is not None:
            print("loading static char_Embedding")
            self.static_char_embed.weight = nn.Parameter(pre_char_word_embedding)
            for index in range(self.args.char_embedding_dim):
                self.static_char_embed.weight.data[self.static_alphabet.char_PaddingID][index] = 0
            self.static_char_embed.weight.requires_grad = False
        else:
            utils.init_embedding(self.static_char_embed.weight)

        if pre_bichar_word_embedding is not None:
            print("loading static bichar_Embedding")
            self.static_bichar_embed.weight = nn.Parameter(pre_bichar_word_embedding)
            for index in range(self.args.bichar_embedding_dim):
                self.static_bichar_embed.weight.data[self.static_alphabet.bichar_PaddingID][index] = 0
            self.static_bichar_embed.weight.requires_grad = False
        else:
            utils.init_embedding(self.static_bichar_embed.weight)

    def encode(self, features):

        max_len = features.char_features.shape[1]

        char_features = self.dropout_embed(self.char_embed(features.char_features))
        bichar_left_features = self.dropout_embed(self.bichar_embed(features.bichar_left_features))
        bichar_right_features = self.dropout_embed(self.bichar_embed(features.bichar_right_features))

        # fix the word embedding
        static_char_features = self.dropout_embed(self.static_char_embed(features.static_char_features))
        static_bichar_l_features = self.dropout_embed(self.static_bichar_embed(features.static_bichar_left_features))
        static_bichar_r_features = self.dropout_embed(self.static_bichar_embed(features.static_bichar_right_features))

        left_concat = torch.cat((char_features, static_char_features, bichar_left_features, static_bichar_l_features), 2)
        right_concat = torch.cat((char_features, static_char_features, bichar_right_features, static_bichar_r_features), 2)

        # non-linear
        #[batch_size, max_len, hidden_dim] --> [max_len, batch_size, hidden_dim] for LSTM
        left_concat_input = self.dropout_embed(torch.nn.functional.tanh(self.liner(left_concat))).permute(1, 0, 2)
        right_concat_input = self.dropout_embed(torch.nn.functional.tanh(self.liner(right_concat))).permute(1, 0, 2)


        # # init hidden cell
        # self.hidden = self.init_hidden_cell(batch_size=batch_length)
        # left_lstm_output, _ = self.lstm_left(left_concat_input)

        left_h, left_c = self.rand_init_rnn_hidden(features.batch_length)
        right_h, right_c = self.rand_init_rnn_hidden(features.batch_length)
        left_lstm_output = []
        right_lstm_output = []
        for idx, id_right in zip(range(max_len), reversed(range(max_len))):
            left_h, left_c = self.lstm_left(left_concat_input[idx], (left_h, left_c))
            right_h, right_c = self.lstm_right(right_concat_input[id_right], (right_h, right_c))
            left_h = self.dropout_embed(left_h)
            right_h = self.dropout_embed(right_h)
            left_lstm_output.append(left_h.view(features.batch_length, 1, self.args.rnn_hidden_dim))
            right_lstm_output.insert(0, right_h.view(features.batch_length, 1, self.args.rnn_hidden_dim))
        left_lstm_output = torch.cat(left_lstm_output, 1)
        right_lstm_output = torch.cat(right_lstm_output, 1)

        encoder_output = torch.cat((left_lstm_output, right_lstm_output), 2)

        return encoder_output

    def decode(self,features, encoder_out, mode): # encode_out [batch_size, max_len, hidden_size]

        batch_size = features.batch_length
        max_len = features.char_features.shape[1]
        encoder_out = encoder_out.permute(1, 0, 2)

        if mode == 'train':
            pos_embeds = self.dropout_embed(self.pos_embed(features.pos_features))
        # pos_embeds = pos_embeds.permute(1, 0, 2)
        start_index = []
        lstm_state = []
        batch_output = [[] for i in range(batch_size)]
        char_output = []

        for id_char in range(max_len):
            if id_char == 0:
                h_now, c_now = self.rand_init_rnn_hidden(batch_size)
            else:
                h, c = lstm_state[-1]
                word_rep = []

                for batch_idx, start_position in enumerate(start_index[-1]):

                    if start_position == -1:
                        padding = utils.varible(torch.zeros(1, 2 * self.args.rnn_hidden_dim), self.args.gpu)
                        pos = self.dropout_embed(self.pos_embed(utils.varible(torch.LongTensor([self.alphabet.pos_PaddingID]), self.args.gpu)))
                        word_rep.append(torch.cat((padding, pos), 1))
                    else:
                        encode = encoder_out.permute(1, 0, 2)
                        word = encode[batch_idx][start_position:id_char]
                        pos = batch_output[batch_idx][-1][1].unsqueeze(0)
                        word = word.unsqueeze(0).permute(0, 2, 1)
                        last_word_embed = F.avg_pool1d(word, word.size(2)).squeeze(2)
                        word_rep.append(torch.cat((last_word_embed, pos), 1))    # last_word_embed [1, 400]

                z = torch.cat([w for w in word_rep])
                z = self.dropout_embed(F.tanh(self.char_2_rnn_input(z)))
                h_now, c_now = self.word_pos_lstmcell(z, (h, c))
            lstm_state.append((h_now, c_now))
            v = torch.cat((h_now, encoder_out[id_char]), 1)

            output = self.lstm_2_output(v) # torch.Size([batch_size, label_size])
            if id_char is 0:
                for i in range(batch_size):
                    output.data[i][self.alphabet.appID] = -10e9
            if mode == 'train':
                start_index, batch_output = self.transition_move(batch_size, mode, id_char, start_index, batch_output, pos_embeds=pos_embeds, insts=features.inst)
            else:
                start_index, batch_output = self.transition_move(batch_size, mode, id_char, start_index, batch_output, predict=output)
            char_output.append(output.unsqueeze(1))
        decoder_out = torch.cat(char_output, 1)
        decoder_out = decoder_out.view(batch_size * max_len, -1)
        decoder_out = self.softmax(decoder_out)
        return decoder_out

    def transition_move(self, batch_size, mode, id_char, start_index, output, pos_embeds=None, insts=None, predict=None):
        start = []
        if mode == 'train':
            for batch_idx in range(len(insts)):
                if id_char < len(insts[batch_idx].gold):
                    action = insts[batch_idx].gold[id_char]
                    if action.startswith("SEP"):
                        current_pos = pos_embeds[batch_idx][len(output[batch_idx])]
                        output[batch_idx].append([1, current_pos])
                        start.append((id_char + 1) - len(output[batch_idx]))
                    elif action.startswith("APP"):
                        output[batch_idx][-1][0] += 1
                        start.append((id_char + 1) - len(output[batch_idx]))
                else:
                    start.append(-1)
        else:
            for batch_idx in range(batch_size):
                actionID = utils.getMaxindex(predict[batch_idx].view(self.args.label_size))
                action = self.alphabet.idx2label[actionID]
                if action.startswith("SEP"):
                    current_pos = self.alphabet.pos2idx[action.split('#')[1]]
                    current_pos = self.pos_embed(utils.varible(torch.LongTensor([current_pos]), self.args.gpu))
                    output[batch_idx].append([1, current_pos.squeeze(0)])
                    start.append((id_char + 1) - len(output[batch_idx]))
                elif action.startswith("APP"):
                    output[batch_idx][-1][0] += 1
                    start.append((id_char + 1) - len(output[batch_idx]))
                else:
                    start.append(-1)

        start_index.append(start)
        return start_index, output

    def forward(self, features, mode):

        encode_out = self.encode(features)
        decoder_out = self.decode(features, encode_out, mode=mode)
        loss = torch.nn.functional.nll_loss(decoder_out, features.gold_features)

        return decoder_out, loss
