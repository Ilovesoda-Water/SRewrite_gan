import torch, time, argparse, os, codecs, h5py, cPickle, random
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from scpn_utils import get_wordmap
import generator, discriminator, helper

def train_discriminator():
    inp = h5f['inputs']

    minibatches = [(start, start + args.dis_batch_size) \
                   for start in range(0, inp.shape[0], args.dis_batch_size)][:-1]  ## return a list[(0,2),(2,4)]
    random.shuffle(minibatches)

    # generating a small validation set before training
    dev_minibatches = minibatches[:args.dis_dev_batches]
    train_batches = minibatches[args.dis_dev_batches:]
    pos_val, neg_val, pos_len, neg_len = helper.generate_samples(h5f, dev_minibatches, vocab, rev_vocab, label_voc, gen, args)
    val_inp, val_len, val_target = helper.prepar_discriminator_data(pos_val, neg_val, pos_len, neg_len)

    ## loss function and optimizer
    loss_fn = nn.BCELoss()
    dis_optimizer = optim.Adagrad(dis.parameters())
    max_val_acc = 0.

    for d_step in range(args.d_steps):
        random.shuffle(train_batches)

        train_index = random.sample(range(len(train_batches)), args.dis_train_batches)
        train_mini_batches = [train_batches[i] for i in train_index]
        pos_train, neg_train, pos_len, neg_len = helper.generate_samples(h5f, train_mini_batches, vocab, rev_vocab, label_voc, gen, args)
        train_inp, train_len, train_target = helper.prepar_discriminator_data(pos_train, neg_train, pos_len, neg_len)

        for epoch in range(args.dis_epochs):

            total_loss = 0.
            stime = time.time()
            num_batch = 0
            # shuffle
            shuffle_idx = random.sample(range(args.dis_train_batches), args.dis_train_batches)
            train_inp_shuffle = [train_inp[i] for i in shuffle_idx]
            train_len_shuffle = [train_len[i] for i in shuffle_idx]
            train_target_shuffle = [train_target[i] for i in shuffle_idx]

            for b_idx in range(args.dis_train_batches):

                train_inp_torch = Variable(torch.from_numpy(train_inp_shuffle[b_idx]).long().cuda())
                train_target_torch = Variable(torch.from_numpy(train_target_shuffle[b_idx]).cuda())
                train_len_torch = torch.from_numpy(train_len_shuffle[b_idx]).long().cuda()

                out = dis.batch_Classify(train_inp_torch, train_len_torch)

                # compute loss
                loss = loss_fn(out, train_target_torch.float())
                total_loss += loss.data[0]

                dis_optimizer.zero_grad()
                loss.backward()
                dis_optimizer.step()

                num_batch += 1
                if num_batch % args.save_freq == 0:
                    print('batch {} / {} in epoch {} step {}, average_loss: {} time: {}'.format(b_idx, len(train_batches),epoch,d_step,
                                                                                        total_loss/num_batch, time.time()-stime))
                    # dis_log.write('batch {} / {} in epoch {}, average_loss: {} time: {}\n'.format(b_idx, len(train_batches),d_step,
                    #                                                                     total_loss/num_batch, time.time()-stime))

                    acc = 0.
                    c = 0.
                    for val_idx in range(args.dis_dev_batches):
                        val_inp_torch = Variable(torch.from_numpy(val_inp[val_idx]).long().cuda())
                        val_target_torch = Variable(torch.from_numpy(val_target[val_idx]).long().cuda())
                        val_len_torch = torch.from_numpy(val_len[val_idx]).long().cuda()
                        val_pred = dis.batch_Classify(val_inp_torch, val_len_torch)
                        acc += torch.sum((val_pred>0.5)==(val_target_torch>0.5)).data[0]
                        c += len(val_inp[val_idx])
                    print('val accuracy {}'.format(float(acc)/c))
                    if float(acc)/c > max_val_acc:
                        max_val_acc = float(acc)/c
                        dis_log.write('max val acc{}\n'.format(max_val_acc))
                        torch.save({'state_dict': dis.state_dict(),
                                    'ep_loss': total_loss / num_batch,
                                    'config_args': args}, 'models/discriminator_gru_200batches_greedy.pt')
                    dis_log.write('batch {} / {} in epoch {}, average_loss: {}, val accuracy {}, time: {}\n'.format(b_idx, len(train_batches),epoch,
                                                                                        total_loss/num_batch, float(acc)/c, time.time()-stime))
                    num_batch = 0
                    total_loss = 0.0
                    stime = time.time()
    dis_log.close()
    print("ok")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Syntactically Controlled Paraphrase Network GAN')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU id')
    parser.add_argument('--data', type=str, default='../scpn-master/data/parsed_data.h5',
                        help='hdf5 location')
    parser.add_argument('--vocab', type=str, default='data/parse_vocab.pkl',
                        help='word vocabulary')
    parser.add_argument('--paragram_sl999', type=str, default='../emnlp2017-master/data/paragram_sl999_small.txt')
    parser.add_argument('--parse_vocab', type=str, default='data/ptb_tagset.txt',
                        help='tag vocabulary')
    parser.add_argument('--model', type=str, default='scpn2_continue.pt',
                        help='model save path')
    parser.add_argument('--dis_gru_model', type=str, default='models/discriminator_gru.pt',
                        help='discriminator gru model save path')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--min_sent_length', type=int, default=5,
                        help='min number of tokens per batch')
    parser.add_argument('--d_word', type=int, default=300,
                        help='word embedding dimension')
    parser.add_argument('--d_trans', type=int, default=128,
                        help='transformation hidden dimension')
    parser.add_argument('--d_nt', type=int, default=56,
                        help='nonterminal embedding dimension')
    parser.add_argument('--d_hid', type=int, default=512,
                        help='lstm hidden dimension')
    parser.add_argument('--n_epochs', type=int, default=15,
                        help='n_epochs')
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='learning rate')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='clip if grad norm exceeds this threshold')
    parser.add_argument('--save_freq', type=int, default=200,
                        help='how many minibatches to save model')
    parser.add_argument('--lr_decay_factor', type=int, default=0.5,
                        help='how much to decrease LR every epoch')
    parser.add_argument('--eval_mode', type=bool, default=False,
                        help='run beam search for some examples using a trained model')
    parser.add_argument('--init_trained_gen_model', type=int, default=1,
                        help='use pretrained scpn model')
    parser.add_argument('--init_trained_model_gen_path', type=str, default='models/scpn.pt',
                        help='pretrained scpn model path')
    parser.add_argument('--tree_dropout', type=float, default=0.,
                        help='dropout rate for dropping tree terminals')
    parser.add_argument('--tree_level_dropout', type=float, default=0.,
                        help='dropout rate for dropping entire levels of a tree')
    parser.add_argument('--short_batch_downsampling_freq', type=float, default=0.0,
                        help='dropout rate for dropping entire levels of a tree')
    parser.add_argument('--short_batch_threshold', type=int, default=20,
                        help='if sentences are shorter than this, they will be downsampled')
    parser.add_argument('--seed', type=int, default=1000,
                        help='random seed')
    parser.add_argument('--use_input_parse', type=int, default=0,
                        help='whether or not to use the input parse')
    parser.add_argument('--dev_batches', type=int, default=50,
                        help='how many minibatches to use for validation')
    parser.add_argument('--dis_dev_batches', type=int, default=50,
                        help='how many minibatches to use for discriminator validation')
    parser.add_argument('--scpn_pt_loss', type=int, default=0,
                        help='the train loss of scpn.pt')
    parser.add_argument('--dis_log', type=str, default='log/dis_log_200batches_greedy.txt',
                        help='discriminator log location')
    parser.add_argument('--d_steps', type=int, default=10000000,
                        help='for discriminator, every dstep we generate new train data')
    parser.add_argument('--dis_epochs', type=int, default=3,
                        help='for discriminator, we train the generated train data dis_epochs times')
    parser.add_argument('--dis_train_batches', type=int, default=200,
                        help='how many minibatches to use for discriminator training')
    parser.add_argument('--dis_batch_size', type=int, default=100,
                        help='discriminator training batch size')
    parser.add_argument('--dis_embedding_dim', type=int, default=300,
                        help='discriminator word embedding dimension')
    parser.add_argument('--dis_hidden_dim', type=int, default=512,
                        help='discriminator gru hidden dimension')
    parser.add_argument('--dis_data_path', type=str, default='data/dis_train.json',
                        help='discriminator training data location')

    args = parser.parse_args()

    batch_size = args.batch_size
    d_word = args.d_word
    d_hid = args.d_hid
    d_trans = args.d_trans
    d_nt = args.d_nt
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load data, word vocab, and parse vocab
    h5f = h5py.File(args.data, 'r')
    inp = h5f['inputs']
    out = h5f['outputs']
    in_parses = h5f['input_parses']
    out_parses = h5f['output_parses']
    in_lens = h5f['in_lengths']
    out_lens = h5f['out_lengths']

    vocab, rev_vocab = \
        cPickle.load(open(args.vocab, 'rb'))  ##get two dicts, vocab is word2id, rev_vocab is id2word

    tag_file = codecs.open(args.parse_vocab, 'r', 'utf-8')
    label_voc = {}  ##no EOP in label_voc
    for idx, line in enumerate(tag_file):
        line = line.strip()
        if line != 'EOP':
            label_voc[line] = idx
    rev_label_voc = dict((v, k) for (k, v) in label_voc.iteritems())

    len_voc = len(vocab)
    len_parse_voc = len(label_voc)

    # build generator network
    gen = generator.SCPN(d_word, d_hid, d_nt, d_trans,
                         len_voc, len_parse_voc, args.use_input_parse).cuda()

    # build discriminator network
    We_vocab = get_wordmap(vocab, args)
    dis = discriminator.Discriminator(args.dis_embedding_dim, args.dis_hidden_dim, len_voc, We_vocab).cuda()

    # GENERATOR TRAINING
    if args.init_trained_gen_model:
        gen.load_state_dict(torch.load(args.init_trained_model_gen_path)['state_dict'])
    else:
        pass

    # dis.load_state_dict(torch.load('models/discriminator_gru_adagrad.pt')['state_dict'])

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    dis_log = open(args.dis_log, 'w')
    dis_log.write('train discriminator...\n')
    train_discriminator()



