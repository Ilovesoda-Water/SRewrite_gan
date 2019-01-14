import random, json
from torch.autograd import Variable
from scpn_utils import *
from torch import nn
def generate_samples(h5f, batches, vocab, rev_vocab, label_voc, generator, args, max_decode=40):
    inp = h5f['inputs']
    out = h5f['outputs']
    in_parses = h5f['input_parses']
    out_parses = h5f['output_parses']
    in_lens = h5f['in_lengths']
    out_lens = h5f['out_lengths']

    pos_val = []
    neg_val = []
    pos_len = []
    neg_len = []



    # print(out.shape)
    # for i in range(len(out)-1, 0, -1):
    #     if out_lens[i]==40:
    #         print('out with len 40: %s' % ' '.join([rev_vocab[w] for (j, w) in enumerate(out[i])]))
    #         break

    for b_idx, (start, end) in enumerate(batches):
        # read data from hdf5
        in_p = in_parses[start:end]
        out_p = out_parses[start:end]

        # get valid instances of transformations
        z = indexify_transformations_no_EOP(in_p, out_p, label_voc, args)
        if z == None:
            continue

        in_trans_np, out_trans_np, mismatch_inds, in_trans_len_np, out_trans_len_np = z

        # only store valid input instances
        inp_np = inp[start:end][mismatch_inds]
        out_np = out[start:end][mismatch_inds]
        in_len_np = in_lens[start:end][mismatch_inds]
        out_len_np = out_lens[start:end][mismatch_inds]
        curr_bsz = inp_np.shape[0]

        # chop input based on length of last instance (for encoder efficiency)
        max_in_len = int(in_len_np[-1])  ## input and output has been paded in the data file, the out ends with 'EOS'
        inp_np = inp_np[:, :max_in_len]  ## what if there is a input sentence's length is larger than max_in_len

        # compute max output length and chop output (for decoder efficiency)
        max_out_len = int(np.amax(out_len_np))
        out_np = out_np[:, :max_out_len]

        # downsample if input sentences are too short
        if args.short_batch_downsampling_freq > 0. and max_in_len < args.short_batch_threshold:
            if np.random.rand() < args.short_batch_downsampling_freq:
                continue

        # randomly invert 50% of batches (to remove NMT bias)
        swap = random.random() > 0.5
        if swap:
            inp_np, out_np = out_np, inp_np
            in_trans_np, out_trans_np = out_trans_np, in_trans_np
            max_in_len, max_out_len = max_out_len, max_in_len
            in_len_np, out_len_np = out_len_np, in_len_np
            in_trans_len_np, out_trans_len_np = out_trans_len_np, in_trans_len_np

        # torchify input
        curr_inp = Variable(torch.from_numpy(inp_np.astype('int32')).long().cuda())
        curr_out = Variable(torch.from_numpy(out_np.astype('int32')).long().cuda())
        in_trans = Variable(torch.from_numpy(in_trans_np).long().cuda())
        out_trans = Variable(torch.from_numpy(out_trans_np).long().cuda())
        in_sent_lens = torch.from_numpy(in_len_np).long().cuda()
        out_sent_lens = torch.from_numpy(out_len_np).long().cuda()
        in_trans_lens = torch.from_numpy(in_trans_len_np).long().cuda()
        out_trans_lens = torch.from_numpy(out_trans_len_np).long().cuda()

        # forward
        preds, _ = generator.greedy_search(curr_inp, in_trans, out_trans,
                          in_sent_lens, out_sent_lens, in_trans_lens, out_trans_lens, max_decode)

        preds = preds.cpu().data.numpy()

        # pred_outputs = generator(curr_inp, curr_out, in_trans, out_trans,
        #                          in_sent_lens, out_sent_lens, in_trans_lens, out_trans_lens, max_out_len)
        # pred_outputs = pred_outputs.view(curr_bsz, max_out_len, -1).cpu().data.numpy()
        # pred_outputs = np.argmax(pred_outputs, -1)

        pos_vals = []
        neg_vals = []
        pos_lens = []
        neg_lens = []
        for i in range(curr_bsz):
            try:
                eos_out = np.where(out_np[i] == vocab['EOS'])[0][0]
                eos_pred = np.where(preds[i] == vocab['EOS'])[0][0]
                # print('gt output: %s' % ' '.join([rev_vocab[w] for (j, w) in enumerate(out_np[i, :(eos_out+1)]) \
                #                 if j < out_len_np[i]]))
                # print('greedy: %s' % ' '.join([rev_vocab[w.item()] for (j, w) in enumerate(preds[i])]))
                # print('pred output: %s' % ' '.join([rev_vocab[w.item()] for (j, w) in enumerate(pred_outputs[i])]))

                pos_vals.append(out_np[i, :(eos_out+1)])
                neg_vals.append(preds[i, :(eos_pred+1)])
                pos_lens.append(eos_out+1)
                neg_lens.append(eos_pred+1)
            except:
                eos_out = np.where(out_np[i] == vocab['EOS'])[0][0]
                pos_vals.append(out_np[i, :(eos_out+1)])
                neg_vals.append(preds[i])
                pos_lens.append(eos_out+1)
                neg_lens.append(max_decode)
                # print('greedy search error')
        if len(pos_vals)>0:
            pos_val.append(pos_vals)
            neg_val.append(neg_vals)
            pos_len.append(pos_lens)
            neg_len.append(neg_lens)

    return pos_val, neg_val, pos_len, neg_len

def prepar_discriminator_data(pos_samples, neg_samples, pos_lens, neg_lens):
    num = len(pos_samples)
    inp = []
    target = []
    len_seq = []

    for i in range(num):
        inp_batch = pos_samples[i] + neg_samples[i]
        len_batch = pos_lens[i] + neg_lens[i]
        max_len = 0
        for _, seq in enumerate(inp_batch):
            if (len(seq)>max_len):
                max_len = len(seq)
        inp_batch_np = np.zeros((len(inp_batch), max_len), 'int32')
        for idx, seq in enumerate(inp_batch):
            inp_batch_np[idx, :len(seq)] = seq
        len_batch_np = np.array(len_batch, 'int32')
        target_batch_np = np.ones(len(pos_samples[i])+len(neg_samples[i]))
        target_batch_np[len(pos_samples[i]):] = 0

        #shuffle
        shuffle_idx = np.random.permutation(len(pos_samples[i])+len(neg_samples[i]))
        inp_batch_np = inp_batch_np[shuffle_idx]
        len_batch_np = len_batch_np[shuffle_idx]
        target_batch_np = target_batch_np[shuffle_idx]
        inp.append(inp_batch_np)
        len_seq.append(len_batch_np)
        target.append(target_batch_np)

    return inp, len_seq, target


def prepar_discriminator_data_json(start, args):
    with open(args.dis_data_path) as f:
        pos_val, neg_val = json.load(f)
        inp = []
        target = []
        for i in range(start, start+64*2000, 64):
            max_len = 0
            inp_batch = pos_val[i:i+args.dis_batch_size] + neg_val[i:i+args.dis_batch_size]
            for _,seq in enumerate(inp_batch):
                if max_len < len(seq):
                    max_len = len(seq)
            inp_batch_np = np.zeros((len(inp_batch), max_len), 'int32')
            for idx, seq in enumerate(inp_batch):
                inp_batch_np[idx, :len(seq)] = seq
            target_batch_np = np.ones(len(inp_batch))
            target_batch_np[args.dis_batch_size:] = 0

            # shuffle
            shuffle_idx = np.random.permutation(len(inp_batch))
            inp_batch_np = inp_batch_np[shuffle_idx]
            target_batch_np = target_batch_np[shuffle_idx]
            inp.append(inp_batch_np)
            target.append(target_batch_np)
        return inp, target

def gen_gen_data_adv(h5f, batches, vocab, rev_vocab, label_voc, generator, args, p_mode='val', max_decode=40, log = None):

    total_loss = 0.
    criterion = nn.NLLLoss(ignore_index=0)

    inp = h5f['inputs']
    out = h5f['outputs']
    in_parses = h5f['input_parses']
    out_parses = h5f['output_parses']
    in_lens = h5f['in_lengths']
    out_lens = h5f['out_lengths']

    inp_nps = []
    out_nps = []
    in_trans_nps = []
    out_trans_nps = []
    in_len_nps = []
    out_len_nps = []
    in_trans_len_nps = []
    out_trans_len_nps = []

    for b_idx, (start, end) in enumerate(batches):
        # read data from hdf5
        in_p = in_parses[start:end]
        out_p = out_parses[start:end]

        # get valid instances of transformations
        z = indexify_transformations_no_EOP(in_p, out_p, label_voc, args)
        if z == None:
            continue

        in_trans_np, out_trans_np, mismatch_inds, in_trans_len_np, out_trans_len_np = z

        # only store valid input instances
        inp_np = inp[start:end][mismatch_inds]
        out_np = out[start:end][mismatch_inds]
        in_len_np = in_lens[start:end][mismatch_inds]
        out_len_np = out_lens[start:end][mismatch_inds]
        curr_bsz = inp_np.shape[0]

        # chop input based on length of last instance (for encoder efficiency)
        max_in_len = int(in_len_np[-1])  ## input and output has been paded in the data file, the out ends with 'EOS'
        inp_np = inp_np[:, :max_in_len]  ## what if there is a input sentence's length is larger than max_in_len

        # compute max output length and chop output (for decoder efficiency)
        max_out_len = int(np.amax(out_len_np))
        out_np = out_np[:, :max_out_len]

        # downsample if input sentences are too short
        if args.short_batch_downsampling_freq > 0. and max_in_len < args.short_batch_threshold:
            if np.random.rand() < args.short_batch_downsampling_freq:
                continue

        # randomly invert 50% of batches (to remove NMT bias)
        swap = random.random() > 0.5
        if swap and p_mode == 'train':
            inp_np, out_np = out_np, inp_np
            in_trans_np, out_trans_np = out_trans_np, in_trans_np
            max_in_len, max_out_len = max_out_len, max_in_len
            in_len_np, out_len_np = out_len_np, in_len_np
            in_trans_len_np, out_trans_len_np = out_trans_len_np, in_trans_len_np

        inp_nps.append(inp_np)
        in_trans_nps.append(in_trans_np)
        out_trans_nps.append(out_trans_np)
        in_len_nps.append(in_len_np)
        in_trans_len_nps.append(in_trans_len_np)
        out_trans_len_nps.append(out_trans_len_np)

        # torchify input
        curr_inp = Variable(torch.from_numpy(inp_np.astype('int32')).long().cuda())
        curr_out = Variable(torch.from_numpy(out_np.astype('int32')).long().cuda())
        in_trans = Variable(torch.from_numpy(in_trans_np).long().cuda())
        out_trans = Variable(torch.from_numpy(out_trans_np).long().cuda())
        in_sent_lens = torch.from_numpy(in_len_np).long().cuda()
        out_sent_lens = torch.from_numpy(out_len_np).long().cuda()
        in_trans_lens = torch.from_numpy(in_trans_len_np).long().cuda()
        out_trans_lens = torch.from_numpy(out_trans_len_np).long().cuda()

        # forward
        preds, pred_probs = generator.greedy_search(curr_inp, in_trans, out_trans, in_sent_lens, out_sent_lens, in_trans_lens, out_trans_lens, max_decode)
        preds = preds.cpu().data.numpy()

        if p_mode == 'val':
            pred_probs = pred_probs[:, :max_out_len, :].contiguous()
            loss = criterion(pred_probs.view(-1, len(vocab)), curr_out.view(-1))
            total_loss += loss.item()

        pred_out_list = []
        pred_out_len_list = []

        for i in range(curr_bsz):
            try:
                eos_pred = np.where(preds[i] == vocab['EOS'])[0][0]
                if p_mode == 'val' and i<10 and log is not None:
                    eos_out = np.where(out_np[i] == vocab['EOS'])[0][0]
                    log.write('gt output: {}\n'.format(' '.join([rev_vocab[w] for (j, w) in enumerate(out_np[i, :(eos_out+1)]) \
                                    if j < out_len_np[i]])))
                    log.write('greedy: {}\n\n'.format(' '.join([rev_vocab[w.item()] for (j, w) in enumerate(preds[i, :(eos_pred+1)])])))
                    # print('gt output: %s' % ' '.join([rev_vocab[w] for (j, w) in enumerate(out_np[i, :(eos_out+1)]) \
                    #                 if j < out_len_np[i]]))
                    # print('greedy: %s' % ' '.join([rev_vocab[w.item()] for (j, w) in enumerate(preds[i, :(eos_pred+1)])]))
                # print('pred output: %s' % ' '.join([rev_vocab[w.item()] for (j, w) in enumerate(pred_outputs[i])]))

                pred_out_list.append(preds[i, :(eos_pred+1)])
                pred_out_len_list.append(eos_pred+1)
            except:
                if p_mode == 'val' and i<10 and log is not None:
                    eos_out = np.where(out_np[i] == vocab['EOS'])[0][0]
                    log.write('gt output: {}\n'.format(
                        ' '.join([rev_vocab[w] for (j, w) in enumerate(out_np[i, :(eos_out + 1)]) \
                                  if j < out_len_np[i]])))
                    log.write('greedy: {}\n\n'.format(
                        ' '.join([rev_vocab[w.item()] for (j, w) in enumerate(preds[i, :(eos_pred + 1)])])))
                    # print('gt output: %s' % ' '.join([rev_vocab[w] for (j, w) in enumerate(out_np[i, :(eos_out + 1)]) \
                    #                                   if j < out_len_np[i]]))
                    # print('greedy: %s' % ' '.join([rev_vocab[w.item()] for (j, w) in enumerate(preds[i])]))
                pred_out_list.append(preds[i])
                pred_out_len_list.append(max_decode)
                # print('greedy search error')
        if len(pred_out_list)>0:
            max_len = max(pred_out_len_list)
            pred_out_np = np.zeros((len(pred_out_list), max_len), 'int32')
            for idx, seq in enumerate(pred_out_list):
                pred_out_np[idx, :len(seq)] = seq
            pred_out_len_np = np.array(pred_out_len_list, 'int32')
            out_nps.append(pred_out_np)
            out_len_nps.append(pred_out_len_np)
    if log is not None:
        log.flush()
    if p_mode == 'train':
        return inp_nps, out_nps, in_len_nps, out_len_nps, in_trans_nps, out_trans_nps, in_trans_len_nps, out_trans_len_nps
    elif p_mode == 'val':
        return total_loss/(b_idx + 1.0)






