import torch, time, argparse, os, codecs, h5py, cPickle, random, sys
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from scpn_utils import get_wordmap
import generator, discriminator, helper

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Syntactically Controlled Paraphrase Network GAN')
    parser.add_argument('--gpu', type=str, default='1',
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
    parser.add_argument('--save_freq', type=int, default=500,
                        help='how many minibatches to save model')
    parser.add_argument('--dis_save_freq', type=int, default=200,
                        help='how many minibatches to save discriminator model while training')
    parser.add_argument('--lr_decay_factor', type=int, default=0.5,
                        help='how much to decrease LR every epoch')
    parser.add_argument('--eval_mode', type=bool, default=False,
                        help='run beam search for some examples using a trained model')
    parser.add_argument('--init_trained_gen_model', type=int, default=1,
                        help='use pretrained scpn model')
    parser.add_argument('--init_trained_dis_model', type=int, default=1,
                        help='use pretrained dis model')
    parser.add_argument('--init_trained_model_gen_path', type=str, default='models/scpn.pt',
                        help='pretrained scpn model path')
    parser.add_argument('--init_trained_model_dis_path', type=str, default='models/discriminator_gru_200batches_greedy.pt',
                        help='pretrained dis model path')
    parser.add_argument('--adv_gen_model_save_path', type=str, default='models/adv_gen_best_nll.pt')
    parser.add_argument('--adv_dis_model_save_path', type=str, default='models/adv_dis_best_nll.pt')
    parser.add_argument('--tree_dropout', type=float, default=0.,
                        help='dropout rate for dropping tree terminals')
    parser.add_argument('--tree_level_dropout', type=float, default=0.,
                        help='dropout rate for dropping entire levels of a tree')
    parser.add_argument('--short_batch_downsampling_freq', type=float, default=0.0,
                        help='dropout rate for dropping entire levels of a tree')
    parser.add_argument('--short_batch_threshold', type=int, default=20,
                        help='if sentences are shorter than this, they will be downsampled')
    parser.add_argument('--seed', type=int, default=1002,
                        help='random seed')
    parser.add_argument('--use_input_parse', type=int, default=0,
                        help='whether or not to use the input parse')
    parser.add_argument('--dev_batches', type=int, default=200,
                        help='how many minibatches to use for validation')
    parser.add_argument('--dis_dev_batches', type=int, default=20,
                        help='how many minibatches to use for discriminator validation')
    parser.add_argument('--scpn_pt_loss', type=int, default=0,
                        help='the train loss of scpn.pt')
    parser.add_argument('--dis_log', type=str, default='log/dis_log_adagrad.txt',
                        help='discriminator log location')
    parser.add_argument('--adv_gen_log', type=str, default='log/adv_gen_log_best_nll.txt',
                        help='adversarial generator log location')
    parser.add_argument('--adv_dis_log', type=str, default='log/adv_dis_log_best_nll.txt',
                        help='adversarial discriminator log location')
    parser.add_argument('--generated_sentence', type=str, default='log/gen_sen_best_nll.txt',
                        help='generated sentences before adversarial training')
    parser.add_argument('--d_steps', type=int, default=15,
                        help='for discriminator, every dstep we generate new train data')
    parser.add_argument('--dis_epochs', type=int, default=1,
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
    parser.add_argument('--adv_train_epochs', type=int, default=5000,
                        help='we adv train the generator adv train epochs times')
    parser.add_argument('--adv_dev_batches', type=int, default=5,
                        help='how many minibatches to use for adversarial validation')
    parser.add_argument('--adv_gen_train_batches', type=int, default=1,
                        help='how many minibatches to use for adversarial gen training')
    parser.add_argument('--adv_gen_batch_size', type=int, default=64,
                        help='adv gen batch size')
    parser.add_argument('--adv_dis_batch_size', type=int, default=100,
                        help='discriminator adversarial training batch size')
    parser.add_argument('--adv_dis_train_batches', type=int, default=5,
                        help='how many minibatches to use for discriminator adversarial training')
    parser.add_argument('--adv_d_steps', type=int, default=5,
                        help='for discriminator, every dstep we generate new adv train data')
    parser.add_argument('--adv_dis_epochs', type=int, default=3,
                        help='for discriminator, we adversarially train the generated train data dis_epochs times')
    parser.add_argument('--adv_dis_dev_batches', type=int, default=5,
                        help='how many minibatches to use for discriminator adversarial validation')
    parser.add_argument('--adv_dis_save_freq', type=int, default=5,
                        help='how many minibatches to save dis adv model')


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

    print(torch.load(args.init_trained_model_dis_path)['ep_loss'])
