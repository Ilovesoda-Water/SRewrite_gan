import torch, time, argparse, os, codecs, h5py, cPickle, random
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from scpn_utils import *
from torch.nn import DataParallel

reload(sys)
sys.setdefaultencoding('utf8')


# seq2seq w/ decoder attention
# transformation embeddings concatenated with decoder word inputs
# attention conditioned on transformation via bilinear product
class SCPN(nn.Module):
    def __init__(self, d_word, d_hid, d_nt, d_trans,
                 len_voc, len_trans_voc, use_input_parse):

        super(SCPN, self).__init__()
        self.d_word = d_word
        self.d_hid = d_hid
        self.d_trans = d_trans
        self.d_nt = d_nt + 1
        self.len_voc = len_voc
        self.len_trans_voc = len_trans_voc
        self.use_input_parse = use_input_parse

        # embeddings
        self.word_embs = nn.Embedding(len_voc, d_word)
        self.trans_embs = nn.Embedding(len_trans_voc, d_nt)

        # lstms
        if use_input_parse:
            self.encoder = nn.LSTM(d_word + d_trans, d_hid, num_layers=1, bidirectional=True, batch_first=True)
        else:
            self.encoder = nn.LSTM(d_word, d_hid, num_layers=1, bidirectional=True, batch_first=True)

        self.encoder_proj = nn.Linear(d_hid * 2, d_hid)
        self.decoder = nn.LSTM(d_word + d_hid, d_hid, num_layers=2, batch_first=True)
        self.trans_encoder = nn.LSTM(d_nt, d_trans, num_layers=1, batch_first=True)
        self.trans_hid_init = Variable(torch.zeros(1, 1, d_trans).cuda())
        self.trans_cell_init = Variable(torch.zeros(1, 1, d_trans).cuda())
        self.e_hid_init = Variable(torch.zeros(2, 1, d_hid).cuda())
        self.e_cell_init = Variable(torch.zeros(2, 1, d_hid).cuda())
        self.d_cell_init = Variable(torch.zeros(2, 1, d_hid).cuda())

        # output softmax
        self.out_dense_1 = nn.Linear(d_hid * 2, d_hid)
        self.out_dense_2 = nn.Linear(d_hid, len_voc)
        self.att_nonlin = nn.Softmax()
        self.out_nonlin = nn.LogSoftmax()

        # attention params
        self.att_parse_proj = nn.Linear(d_trans, d_hid)
        self.att_W = nn.Parameter(torch.Tensor(d_hid, d_hid).cuda())
        self.att_parse_W = nn.Parameter(torch.Tensor(d_hid, d_hid).cuda())
        nn.init.xavier_uniform(self.att_parse_W.data)
        nn.init.xavier_uniform(self.att_W.data)

        # copy prob params
        self.copy_hid_v = nn.Parameter(torch.Tensor(d_hid, 1).cuda())
        self.copy_att_v = nn.Parameter(torch.Tensor(d_hid, 1).cuda())
        self.copy_inp_v = nn.Parameter(torch.Tensor(d_word + d_hid, 1).cuda())
        nn.init.xavier_uniform(self.copy_hid_v.data)
        nn.init.xavier_uniform(self.copy_att_v.data)
        nn.init.xavier_uniform(self.copy_inp_v.data)

    # create matrix mask from length vector
    def compute_mask(self, lengths):
        max_len = torch.max(lengths)
        range_row = torch.arange(0, max_len).long().cuda()[None, :].expand(lengths.size()[0], max_len)
        mask = lengths[:, None].expand_as(range_row)
        mask = range_row < mask
        return Variable(mask.float().cuda())

    # masked softmax for attention
    def masked_softmax(self, vector, mask):
        result = torch.nn.functional.softmax(vector)
        result = result * mask
        result = result / (result.sum(dim=1, keepdim=True) + 1e-13)
        return result

    # compute masked attention over enc hiddens with bilinear product
    def compute_decoder_attention(self, hid_previous, enc_hids, in_lens):
        mask = self.compute_mask(in_lens)
        b_hn = hid_previous.mm(self.att_W)
        scores = b_hn[:, None, :] * enc_hids
        scores = torch.sum(scores, 2)
        scores = self.masked_softmax(scores, mask)
        return scores

    # compute masked attention over parse sequence with bilinear product
    def compute_transformation_attention(self, hid_previous, trans_embs, trans_lens):

        mask = self.compute_mask(trans_lens)
        b_hn = hid_previous.mm(self.att_parse_W)
        scores = b_hn[:, None, :] * trans_embs
        scores = torch.sum(scores, 2)
        scores = self.masked_softmax(scores, mask)
        return scores

    # return encoding for an input batch
    def encode_batch(self, inputs, trans, lengths):

        bsz, max_len = inputs.size()
        max_len = int(torch.max(lengths))  # data in different GPUs may have different max lengths
        in_embs = self.word_embs(inputs)
        lens, indices = torch.sort(lengths, 0, True)

        # concat word embs with trans hid
        if self.use_input_parse:
            in_embs = torch.cat([in_embs, trans.unsqueeze(1).expand(bsz, max_len, self.d_trans)], 2)

        e_hid_init = self.e_hid_init.expand(2, bsz, self.d_hid).contiguous()
        e_cell_init = self.e_cell_init.expand(2, bsz, self.d_hid).contiguous()
        all_hids, (enc_last_hid, _) = self.encoder(pack(in_embs[indices],
                                                        lens.tolist(), batch_first=True), (e_hid_init, e_cell_init))

        _, _indices = torch.sort(indices, 0)
        all_hids = unpack(all_hids, batch_first=True)[0][_indices]
        all_hids = self.encoder_proj(all_hids.view(-1, self.d_hid * 2)).view(bsz, max_len, self.d_hid)
        enc_last_hid = torch.cat([enc_last_hid[0], enc_last_hid[1]], 1)
        enc_last_hid = self.encoder_proj(enc_last_hid)[_indices]

        return all_hids, enc_last_hid

    # return encoding for an input batch
    def encode_transformations(self, trans, lengths, return_last=True):

        bsz, _ = trans.size()

        lens, indices = torch.sort(lengths, 0, True)
        bsz, p_len = trans.size()
        for i in range(bsz):
            for j in range(p_len):
                try:
                    if int(trans[i][j]) == 103:
                        print(trans[i])
                        trans[i][j] = 0
                        break
                except:
                    print(trans, i, j)

        in_embs = self.trans_embs(trans)
        t_hid_init = self.trans_hid_init.expand(1, bsz, self.d_trans).contiguous()
        t_cell_init = self.trans_cell_init.expand(1, bsz, self.d_trans).contiguous()
        all_hids, (enc_last_hid, _) = self.trans_encoder(pack(in_embs[indices],
                                                              lens.tolist(), batch_first=True),
                                                         (t_hid_init, t_cell_init))
        _, _indices = torch.sort(indices, 0)

        if return_last:
            return enc_last_hid.squeeze(0)[_indices]
        else:
            all_hids = unpack(all_hids, batch_first=True)[0]
            return all_hids[_indices]

    # decode one timestep
    def decode_step(self, idx, prev_words, prev_hid, prev_cell,
                    enc_hids, trans_embs, in_sent_lens, trans_lens, bsz, max_len):

        # initialize with zeros
        if idx == 0:
            word_input = Variable(torch.zeros(bsz, 1, self.d_word).cuda())

        # get previous ground truth word embed and concat w/ transformation emb
        else:
            word_input = self.word_embs(prev_words)
            word_input = word_input.view(bsz, 1, self.d_word)

        # concatenate w/ transformation embeddings
        trans_weights = self.compute_transformation_attention(prev_hid[1], trans_embs, trans_lens)
        trans_ctx = torch.sum(trans_weights[:, :, None] * trans_embs, 1)
        decoder_input = torch.cat([word_input, trans_ctx.unsqueeze(1)], 2)  # (bsz, 1, d_word+d_hid)

        # feed to decoder lstm
        _, (hn, cn) = self.decoder(decoder_input, (prev_hid, prev_cell))  # (2, bsz, d_hid)

        # compute attention for next time step and att weighted ave of encoder hiddens
        attn_weights = self.compute_decoder_attention(hn[1], enc_hids, in_sent_lens)  # (bsz, max_len)
        attn_ctx = torch.sum(attn_weights[:, :, None] * enc_hids, 1)  # (bsz, d_hid)

        # compute copy prob as function of lotsa shit
        p_copy = decoder_input.squeeze(1).mm(self.copy_inp_v)  # (bsz, 1)
        p_copy += attn_ctx.mm(self.copy_att_v)  # (bsz, 1)
        p_copy += hn[1].mm(self.copy_hid_v)  # (bsz, 1)
        p_copy = torch.sigmoid(p_copy).squeeze(1)  # (bsz)

        return hn, cn, attn_weights, attn_ctx, p_copy

    def forward(self, inputs, outputs, in_trans, out_trans,
                in_sent_lens, out_sent_lens, in_trans_lens, out_trans_lens, max_decode):

        bsz, max_len = inputs.size()
        actual_max_len = int(torch.max(in_sent_lens))  # data in different GPUs may have different max lengths

        # encode transformations
        in_trans_hids = None
        if self.use_input_parse:
            in_trans_hids = self.encode_transformations(in_trans, in_trans_lens)

        out_trans_hids = self.encode_transformations(out_trans, out_trans_lens, return_last=False)
        out_trans_hids = self.att_parse_proj(out_trans_hids)

        # encode input sentence
        enc_hids, enc_last_hid = self.encode_batch(inputs, in_trans_hids, in_sent_lens)

        # store decoder hiddens and attentions for copy

        decoder_states = Variable(torch.zeros(max_decode, bsz, self.d_hid * 2).cuda())
        decoder_copy_dists = Variable(torch.zeros(max_decode, bsz, self.len_voc).cuda())
        copy_probs = Variable(torch.zeros(max_decode, bsz).cuda())

        # initialize decoder hidden to last encoder hidden
        hn = enc_last_hid.unsqueeze(0).expand(2, bsz, self.d_hid).contiguous()
        cn = self.d_cell_init.expand(2, bsz, self.d_hid).contiguous()

        # loop til max_decode, do lstm tick using previous prediction
        for idx in range(max_decode):

            prev_words = None
            if idx > 0:
                prev_words = outputs[:, idx - 1]

            # concat prev word emb and trans emb and feed to decoder lstm
            hn, cn, attn_weights, attn_ctx, p_copy = self.decode_step(idx, prev_words,  ## add argument max_len
                                                                      hn, cn, enc_hids, out_trans_hids, in_sent_lens,
                                                                      out_trans_lens, bsz, max_len)

            # compute copy attn by scattering attn into vocab space
            vocab_scores = Variable(torch.zeros(bsz, self.len_voc).cuda())
            vocab_scores = vocab_scores.scatter_add_(1, inputs[:, :actual_max_len], attn_weights)  # (bsz, vocab_size)

            # store decoder hiddens and copy probs in log domain
            decoder_states[idx] = torch.cat([hn[1], attn_ctx], 1)  # (bsz, d_hid*2)
            decoder_copy_dists[idx] = torch.log(vocab_scores + 1e-20)  # (bsz, vocab_size)
            copy_probs[idx] = p_copy   # (bsz)

        # now do prediction over decoder states (reshape to 2d first)
        decoder_states = decoder_states.transpose(0, 1).contiguous().view(-1, self.d_hid * 2)  # (bsz*max_len, d_hid)
        decoder_preds = self.out_dense_1(decoder_states)
        decoder_preds = self.out_dense_2(decoder_preds)  # (bsz*max_len, vocab_size)
        decoder_preds = self.out_nonlin(decoder_preds)  # (bsz*max_len, vocab_size)
        decoder_copy_dists = decoder_copy_dists.transpose(0, 1).contiguous().view(-1, self.len_voc)  # (bsz*max_len, vocab_size)

        # merge copy dist and pred dist using copy probs
        copy_probs = copy_probs.view(-1)  # (bsz*max_len)
        final_dists = copy_probs[:, None] * decoder_copy_dists + \
                      (1. - copy_probs[:, None]) * decoder_preds
        return final_dists

    def greedy_search(self, inputs, in_trans, out_trans,
                in_sent_lens, out_sent_lens, in_trans_lens, out_trans_lens, max_decode=40):
        bsz, max_len = inputs.size()
        actual_max_len = int(torch.max(in_sent_lens))  # data in different GPUs may have different max lengths

        # encode transformations
        in_trans_hids = None
        if self.use_input_parse:
            in_trans_hids = self.encode_transformations(in_trans, in_trans_lens)

        out_trans_hids = self.encode_transformations(out_trans, out_trans_lens, return_last=False)
        out_trans_hids = self.att_parse_proj(out_trans_hids)

        # encode input sentence
        enc_hids, enc_last_hid = self.encode_batch(inputs, in_trans_hids, in_sent_lens)

        # initialize decoder hidden to last encoder hidden
        hn = enc_last_hid.unsqueeze(0).expand(2, bsz, self.d_hid).contiguous()
        cn = self.d_cell_init.expand(2, bsz, self.d_hid).contiguous()

        seq = Variable(torch.zeros(bsz, max_decode).long().cuda())
        seq_prob = Variable(torch.zeros(bsz, max_decode, self.len_voc).float().cuda())
        for idx in range(max_decode):
            prev_words = None
            if idx>0:
                prev_words = seq[:, idx-1]

            # concat prev word emb and trans emb and feed to decoder lstm
            hn, cn, attn_weights, attn_ctx, p_copy = self.decode_step(idx, prev_words,  ## add argument max_len
                                                                      hn, cn, enc_hids, out_trans_hids,
                                                                      in_sent_lens,
                                                                      out_trans_lens, bsz, max_len)

            # compute copy attn by scattering attn into vocab space
            vocab_scores = Variable(torch.zeros(bsz, self.len_voc).cuda())
            vocab_scores = vocab_scores.scatter_add_(1, inputs[:, :actual_max_len], attn_weights)  # (bsz, vocab_size)

            decoder_state = torch.cat([hn[1], attn_ctx], 1)  # (bsz, d_hid*2)
            decoder_pred = self.out_dense_1(decoder_state)
            decoder_pred = self.out_dense_2(decoder_pred)
            decoder_pred = self.out_nonlin(decoder_pred)
            decoder_copy_dist = torch.log(vocab_scores + 1e-20)  # (bsz, vocab_size)
            final_dist = p_copy[:, None] * decoder_copy_dist + \
                            (1. - p_copy[:, None]) * decoder_pred  # (bsz, vocab_size)

            _, top_indices = torch.sort(-final_dist, 1)
            seq[:, idx] = top_indices[:, 0]
            seq_prob[:, idx] = final_dist

        return seq, seq_prob.contiguous()

    def batchPGLoss(self, inputs, outputs, in_trans, out_trans,
                in_sent_lens, out_sent_lens, in_trans_lens, out_trans_lens, rewards, max_decode=40):
        bsz, max_len = inputs.size()
        actual_max_len = int(torch.max(in_sent_lens))  # data in different GPUs may have different max lengths

        # encode transformations
        in_trans_hids = None
        if self.use_input_parse:
            in_trans_hids = self.encode_transformations(in_trans, in_trans_lens)

        out_trans_hids = self.encode_transformations(out_trans, out_trans_lens, return_last=False)
        out_trans_hids = self.att_parse_proj(out_trans_hids)

        # encode input sentence
        enc_hids, enc_last_hid = self.encode_batch(inputs, in_trans_hids, in_sent_lens)

        # initialize decoder hidden to last encoder hidden
        hn = enc_last_hid.unsqueeze(0).expand(2, bsz, self.d_hid).contiguous()
        cn = self.d_cell_init.expand(2, bsz, self.d_hid).contiguous()

        loss = 0
        max_out_len = max(out_sent_lens)
        for idx in range(max_out_len):
            prev_words = None
            if idx > 0:
                prev_words = outputs[:, idx - 1]

            # concat prev word emb and trans emb and feed to decoder lstm
            hn, cn, attn_weights, attn_ctx, p_copy = self.decode_step(idx, prev_words,  ## add argument max_len
                                                                      hn, cn, enc_hids, out_trans_hids,
                                                                      in_sent_lens,
                                                                      out_trans_lens, bsz, max_len)
            # compute copy attn by scattering attn into vocab space
            vocab_scores = Variable(torch.zeros(bsz, self.len_voc).cuda())
            vocab_scores = vocab_scores.scatter_add_(1, inputs[:, :actual_max_len], attn_weights)  # (bsz, vocab_size)

            decoder_state = torch.cat([hn[1], attn_ctx], 1)  # (bsz, d_hid*2)
            decoder_pred = self.out_dense_1(decoder_state)
            decoder_pred = self.out_dense_2(decoder_pred)
            decoder_pred = self.out_nonlin(decoder_pred)
            decoder_copy_dist = torch.log(vocab_scores + 1e-20)  # (bsz, vocab_size)
            final_dist = p_copy[:, None] * decoder_copy_dist + \
                         (1. - p_copy[:, None]) * decoder_pred  # (bsz, vocab_size)

            for batch_index in range(bsz):
                if out_sent_lens[batch_index]>idx:
                    # print(batch_index, out_sent_lens[batch_index], idx)
                    loss += -final_dist[batch_index][outputs[batch_index][idx]] * rewards[batch_index]

        return loss/bsz
















    # beam search given a single input / transformation
    def beam_search(self, inputs, in_trans, out_trans, in_sent_lens, in_trans_lens,
                    out_trans_lens, eos_idx, beam_size=4, max_steps=40):

        bsz, max_len = inputs.size()

        # chop input
        inputs = inputs[:, :in_sent_lens[0]]

        # encode transformations
        in_trans_hids = None
        if self.use_input_parse:
            in_trans_hids = self.encode_transformations(in_trans, in_trans_lens)
        out_trans_hids = self.encode_transformations(out_trans, out_trans_lens, return_last=False)
        out_trans_hids = self.att_parse_proj(out_trans_hids)

        # encode input sentence
        enc_hids, enc_last_hid = self.encode_batch(inputs, in_trans_hids, in_sent_lens)

        # initialize decoder hidden to last encoder hidden
        hn = enc_last_hid.unsqueeze(0).expand(2, bsz, self.d_hid).contiguous()
        cn = self.d_cell_init.expand(2, bsz, self.d_hid).contiguous()

        # initialize beams
        beams = [(0.0, hn, cn, [])]
        nsteps = 0

        # loop til max_decode, do lstm tick using previous prediction
        while True:

            # loop over everything in the beam
            beam_candidates = []
            for b in beams:
                curr_prob, prev_h, prev_c, seq = b

                # start with last word in sequence, if eos end the beam
                if len(seq) > 0:
                    prev_word = seq[-1]
                    if prev_word == eos_idx:
                        beam_candidates.append(b)
                        continue

                        # load into torch var so we can do decoding
                    prev_word = Variable(torch.LongTensor([prev_word]).cuda())

                else:
                    prev_word = None

                # concat prev word emb and prev attn input and feed to decoder lstm
                hn, cn, attn_weights, attn_ctx, p_copy = self.decode_step(len(seq), prev_word,
                                                                          prev_h, prev_c, enc_hids, out_trans_hids,
                                                                          in_sent_lens, out_trans_lens, bsz, max_len)

                # compute copy attn by scattering attn into vocab space
                vocab_scores = Variable(torch.zeros(bsz, self.len_voc).cuda())
                vocab_scores = vocab_scores.scatter_add_(1, inputs, attn_weights)
                vocab_scores = torch.log(vocab_scores + 1e-20).squeeze() # vocab_size

                # compute prediction over vocab for a single time step
                pred_inp = torch.cat([hn[1], attn_ctx], 1)
                preds = self.out_dense_1(pred_inp)
                preds = self.out_dense_2(preds)
                preds = self.out_nonlin(preds).squeeze()

                final_preds = p_copy * vocab_scores + (1 - p_copy) * preds

                # sort in descending order (log domain)
                _, top_indices = torch.sort(-final_preds)

                # add top n candidates
                for z in range(beam_size):
                    word_idx = top_indices[z].data[0]
                    beam_candidates.append((curr_prob + final_preds[word_idx].data[0],
                                            hn, cn, seq + [word_idx]))
            beam_candidates.sort(reverse=True)
            beams = beam_candidates[:beam_size]

            nsteps += 1
            if nsteps > max_steps:
                break

        return beams

    # beam search given a single sentence and a batch of transformations
    def batch_beam_search(self, inputs, out_trans, in_sent_lens,
                          out_trans_lens, eos_idx, beam_size=5, max_steps=70):
        bsz, max_len = inputs.size()

        # chop input
        inputs = inputs[:, :in_sent_lens[0]]
        # encode transformations
        out_trans_hids = self.encode_transformations(out_trans, out_trans_lens, return_last=False)
        out_trans_hids = self.att_parse_proj(out_trans_hids)

        # encode input sentence
        enc_hids, enc_last_hid = self.encode_batch(inputs, None, in_sent_lens)

        # initialize decoder hidden to last encoder hidden
        hn = enc_last_hid.unsqueeze(0).expand(2, bsz, self.d_hid).contiguous()
        cn = self.d_cell_init

        # initialize beams (dictionary of batch_idx: beam params)
        beam_dict = OrderedDict()
        for b_idx in range(out_trans.size()[0]):
            beam_dict[b_idx] = [(0.0, hn, cn, [])]
        nsteps = 0

        # loop til max_decode, do lstm tick using previous prediction
        while True:

            # set up accumulators for predictions
            # assumption: all examples have same number of beams at each timestep
            prev_words = []
            prev_hs = []
            prev_cs = []

            for b_idx in beam_dict:

                beams = beam_dict[b_idx]
                # loop over everything in the beam
                beam_candidates = []
                for b in beams:
                    curr_prob, prev_h, prev_c, seq = b

                    # start with last word in sequence, if eos end the beam
                    if len(seq) > 0:
                        prev_words.append(seq[-1])

                    else:
                        prev_words = None

                    prev_hs.append(prev_h)
                    prev_cs.append(prev_c)

            # now batch decoder computations
            hs = torch.cat(prev_hs, 1)
            cs = torch.cat(prev_cs, 1)
            num_examples = hs.size()[1]
            if prev_words is not None:
                prev_words = Variable(torch.from_numpy(np.array(prev_words, dtype='int32')).long().cuda())

            # expand out parse states if necessary
            if num_examples != out_trans_hids.size()[0]:
                d1, d2, d3 = out_trans_hids.size()
                rep_factor = num_examples / d1
                curr_out = out_trans_hids.unsqueeze(1).expand(d1, rep_factor,
                                                              d2, d3).contiguous().view(-1, d2, d3)
                curr_out_lens = out_trans_lens.unsqueeze(1).expand(d1, rep_factor).contiguous().view(-1)

            else:
                curr_out = out_trans_hids
                curr_out_lens = out_trans_lens

            # expand out inputs and encoder hiddens
            _, in_len, hid_d = enc_hids.size()
            curr_enc_hids = enc_hids.expand(num_examples, in_len, hid_d)
            curr_enc_lens = in_sent_lens.expand(num_examples)
            curr_inputs = inputs.expand(num_examples, in_sent_lens[0])

            # concat prev word emb and prev attn input and feed to decoder lstm
            hn, cn, attn_weights, attn_ctx, p_copy = self.decode_step(nsteps, prev_words,
                                                                      hs, cs, curr_enc_hids, curr_out, curr_enc_lens,
                                                                      curr_out_lens, num_examples, max_len)

            # compute copy attn by scattering attn into vocab space
            vocab_scores = Variable(torch.zeros(num_examples, self.len_voc).cuda())
            vocab_scores = vocab_scores.scatter_add_(1, curr_inputs, attn_weights)
            vocab_scores = torch.log(vocab_scores + 1e-20).squeeze()

            # compute prediction over vocab for a single time step
            pred_inp = torch.cat([hn[1], attn_ctx], 1)
            preds = self.out_dense_1(pred_inp)
            preds = self.out_dense_2(preds)
            preds = self.out_nonlin(preds).squeeze()
            final_preds = p_copy[:, None] * vocab_scores + (1 - p_copy[:, None]) * preds

            # now loop over the examples and sort each separately
            for b_idx in beam_dict:
                beam_candidates = []

                # no words previously predicted
                if num_examples == len(beam_dict):
                    ex_hn = hn[:, b_idx, :].unsqueeze(1)
                    ex_cn = cn[:, b_idx, :].unsqueeze(1)
                    preds = final_preds[b_idx]
                    _, top_indices = torch.sort(-preds)
                    # add top n candidates
                    for z in range(beam_size):
                        word_idx = top_indices[z].data[0]
                        beam_candidates.append((preds[word_idx].data[0],
                                                ex_hn, ex_cn, [word_idx]))
                    beam_dict[b_idx] = beam_candidates

                else:
                    origin_beams = beam_dict[b_idx]
                    start = b_idx * beam_size
                    end = (b_idx + 1) * beam_size
                    ex_hn = hn[:, start:end, :]
                    ex_cn = cn[:, start:end, :]
                    ex_preds = final_preds[start:end]

                    for o_idx, ob in enumerate(origin_beams):
                        curr_prob, _, _, seq = ob

                        # if one of the beams is already complete, add it to candidates
                        if seq[-1] == eos_idx:
                            beam_candidates.append(ob)

                        preds = ex_preds[o_idx]
                        curr_hn = ex_hn[:, o_idx, :]
                        curr_cn = ex_cn[:, o_idx, :]
                        _, top_indices = torch.sort(-preds)
                        for z in range(beam_size):
                            word_idx = top_indices[z].data[0]

                            beam_candidates.append((curr_prob + float(preds[word_idx].cpu().data[0]),
                                                    curr_hn.unsqueeze(1), curr_cn.unsqueeze(1), seq + [word_idx]))

                    s_inds = np.argsort([x[0] for x in beam_candidates])[::-1]
                    beam_candidates = [beam_candidates[x] for x in s_inds]
                    beam_dict[b_idx] = beam_candidates[:beam_size]

            nsteps += 1
            if nsteps > max_steps:
                break

        return beam_dict

