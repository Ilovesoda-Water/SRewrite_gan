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


def train_discriminator(discriminator, generator, minibatches, dev_minibatches, dis_batch_size, dis_dev_batches, d_steps, dis_train_batches, dis_epochs, save_freq):

    train_batches = minibatches[dis_dev_batches:]
    pos_val, neg_val, pos_len, neg_len = helper.generate_samples(h5f, dev_minibatches, vocab, rev_vocab, label_voc, generator, args)
    val_inp, val_len, val_target = helper.prepar_discriminator_data(pos_val, neg_val, pos_len, neg_len)


    ## loss function and optimizer
    loss_fn = nn.BCELoss()
    dis_optimizer = optim.Adagrad(discriminator.parameters(), lr=5e-5)
    max_val_acc = 0.

    for d_step in range(d_steps):

        train_index = random.sample(range(len(train_batches)), dis_train_batches)
        train_mini_batches = [train_batches[i] for i in train_index]
        pos_train, neg_train, pos_len, neg_len = helper.generate_samples(h5f, train_mini_batches, vocab, rev_vocab, label_voc, generator, args)
        train_inp, train_len, train_target = helper.prepar_discriminator_data(pos_train, neg_train, pos_len, neg_len)

        for epoch in range(dis_epochs):

            total_loss = 0.
            stime = time.time()
            num_batch = 0
            # shuffle
            shuffle_idx = random.sample(range(dis_train_batches), dis_train_batches)
            train_inp_shuffle = [train_inp[i] for i in shuffle_idx]
            train_len_shuffle = [train_len[i] for i in shuffle_idx]
            train_target_shuffle = [train_target[i] for i in shuffle_idx]

            for b_idx in range(dis_train_batches):

                train_inp_torch = Variable(torch.from_numpy(train_inp_shuffle[b_idx]).long().cuda())
                train_target_torch = Variable(torch.from_numpy(train_target_shuffle[b_idx]).cuda())
                train_len_torch = torch.from_numpy(train_len_shuffle[b_idx]).long().cuda()

                out = discriminator.batch_Classify(train_inp_torch, train_len_torch)

                # compute loss
                loss = loss_fn(out, train_target_torch.float())
                total_loss += loss.data[0]

                dis_optimizer.zero_grad()
                loss.backward()
                dis_optimizer.step()

                num_batch += 1
                if num_batch % save_freq == 0:
                    print('batch {} / {} in epoch {} step {}, average_loss: {} time: {}'.format(b_idx, len(train_batches),epoch,d_step,
                                                                                        total_loss/num_batch, time.time()-stime))
                    # adv_dis_log.write('batch {} / {} in epoch {} step {}, average_loss: {} time: {}\n'.format(b_idx, len(train_batches),d_step,
                    #                                                                     total_loss/num_batch, time.time()-stime))

                    acc = 0.
                    c = 0.
                    for val_idx in range(dis_dev_batches):
                        val_inp_torch = Variable(torch.from_numpy(val_inp[val_idx]).long().cuda())
                        val_target_torch = Variable(torch.from_numpy(val_target[val_idx]).long().cuda())
                        val_len_torch = torch.from_numpy(val_len[val_idx]).long().cuda()
                        val_pred = discriminator.batch_Classify(val_inp_torch, val_len_torch)
                        acc += torch.sum((val_pred>0.5)==(val_target_torch>0.5)).data[0]
                        c += len(val_inp[val_idx])
                    print('val accuracy {}'.format(float(acc)/c))
                    # if float(acc)/c > max_val_acc:
                    #     max_val_acc = float(acc)/c
                    #     # dis_log.write('max val acc{}\n'.format(max_val_acc))
                    #     # torch.save({'state_dict': dis.state_dict(),
                    #     #             'ep_loss': total_loss / num_batch,
                    #     #             'config_args': args}, 'models/discriminator_gru_200batches_greedy.pt')
                    adv_dis_log.write('batch {} / {} in epoch {} step {}, average_loss: {}, val accuracy {}, time: {}\n'.format(b_idx, len(train_batches),epoch,d_step,
                                                                                        total_loss/num_batch, float(acc)/c, time.time()-stime))
                    adv_dis_log.flush()
                    num_batch = 0.
                    total_loss = 0.0
                    stime = time.time()

    # torch.save({'state_dict': discriminator.state_dict(),
    #             'val_acc': float(acc)/c,
    #             'config_args': args}, args.adv_dis_model_save_path)
    # dis_log.close()
    return float(acc)/c


def train_generator_PG(generator, minibatches, dev_minibatches, gen_opt, discriminator, args):

    train_batches = minibatches[args.adv_dev_batches:]

    train_index = random.sample(range(len(train_batches)), args.adv_gen_train_batches)
    train_mini_batches = [train_batches[i] for i in train_index]

    # helper.gen_gen_data_adv(h5f, dev_minibatches, vocab, rev_vocab,label_voc,generator,args, 'val', 40)

    inp_nps, out_nps, in_len_nps, out_len_nps, in_trans_nps, out_trans_nps, in_trans_len_nps, out_trans_len_nps = \
        helper.gen_gen_data_adv(h5f, train_mini_batches, vocab, rev_vocab,label_voc,generator,args, 'train', 40)

    for batch in range(args.adv_gen_train_batches):
        # pass
        train_inp_torch = Variable(torch.from_numpy(out_nps[batch].astype('int32')).long().cuda())
        train_len_torch = torch.from_numpy(out_len_nps[batch]).long().cuda()
        rewards = discriminator.batch_Classify(train_inp_torch, train_len_torch) # positive prob

        inp_tensor = Variable(torch.from_numpy(inp_nps[batch].astype('int32')).long().cuda())
        out_tensor = Variable(torch.from_numpy(out_nps[batch].astype('int32')).long().cuda())
        in_trans_tensor = Variable(torch.from_numpy(in_trans_nps[batch]).long().cuda())
        out_trans_tensor = Variable(torch.from_numpy(out_trans_nps[batch]).long().cuda())
        in_sent_lens_tensor = torch.from_numpy(in_len_nps[batch]).long().cuda()
        out_sent_lens_tensor = torch.from_numpy(out_len_nps[batch]).long().cuda()
        in_trans_lens_tensor = torch.from_numpy(in_trans_len_nps[batch]).long().cuda()
        out_trans_lens_tensor = torch.from_numpy(out_trans_len_nps[batch]).long().cuda()

        loss = generator.batchPGLoss(inp_tensor, out_tensor, in_trans_tensor, out_trans_tensor, in_sent_lens_tensor, out_sent_lens_tensor,
                        in_trans_lens_tensor, out_trans_lens_tensor, rewards, max_decode=40)
        print('generator adversarial training loss {}'.format(loss))
        adv_gen_log.write('generator adversarial training loss {}\n\n'.format(loss.item()))

        gen_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(generator.parameters(), args.grad_clip)
        gen_opt.step()

    dev_nll_loss = helper.gen_gen_data_adv(h5f, dev_minibatches, vocab, rev_vocab, label_voc, generator, args, p_mode='val',
                         max_decode=40, log=gen_sen)
    print('dev NLL loss {}'.format(dev_nll_loss))

    # torch.save({'state_dict': generator.state_dict(),
    #             'dev_nll_loss':dev_nll_loss,
    #             'config_args': args}, args.adv_gen_model_save_path)

    adv_gen_log.write('dev NLL loss {}\n'.format(dev_nll_loss))
    adv_gen_log.flush()
    return dev_nll_loss



if __name__ == '__main__':
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
    parser.add_argument('--adv_dis_train_batches', type=int, default=200,
                        help='how many minibatches to use for discriminator adversarial training')
    parser.add_argument('--adv_d_steps', type=int, default=5,
                        help='for discriminator, every dstep we generate new adv train data')
    parser.add_argument('--adv_dis_epochs', type=int, default=3,
                        help='for discriminator, we adversarially train the generated train data dis_epochs times')
    parser.add_argument('--adv_dis_dev_batches', type=int, default=50,
                        help='how many minibatches to use for discriminator adversarial validation')
    parser.add_argument('--adv_dis_save_freq', type=int, default=200,
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

    # GENERATOR TRAINING
    gen_optimizer = optim.Adam(gen.parameters(), lr=args.lr)
    if args.init_trained_gen_model:
        gen.load_state_dict(torch.load(args.init_trained_model_gen_path)['state_dict'])
    else:
        pass

    # dis.load_state_dict(torch.load('models/discriminator_gru_adagrad.pt')['state_dict'])

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    # dis_log = open(args.dis_log, 'w')
    # dis_log.write('train discriminator...\n')
    if args.init_trained_dis_model:
        dis.load_state_dict(torch.load(args.init_trained_model_dis_path)['state_dict'])

    else:
        train_discriminator(dis, gen, args.dis_batch_size, args.dis_dev_batches, args.d_steps, args.dis_train_batches,
                            args.dis_epochs, args.dis_save_freq)

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')

    adv_dis_log = open(args.adv_dis_log, 'w')
    adv_gen_log = open(args.adv_gen_log, 'w')
    gen_sen = open(args.generated_sentence, 'w')
    best_nll_loss = 10.0


    # data division for generator
    gen_minibatches = [(start, start + args.adv_gen_batch_size) \
                   for start in range(0, inp.shape[0], args.adv_gen_batch_size)][:-1]  ## return a list[(0,2),(2,4)]
    random.shuffle(gen_minibatches)
    gen_dev_minibatches = gen_minibatches[:args.adv_dev_batches]
    gen_sen.write('\n--------\nEPOCH %d\n--------\n\n' % (0))
    dev_nll_loss = helper.gen_gen_data_adv(h5f, gen_dev_minibatches, vocab, rev_vocab, label_voc, gen, args, 'val', 40, gen_sen)
    adv_gen_log.write('\n--------\nEPOCH %d\n--------\n\n' % (0))
    adv_gen_log.write('dev NLL loss {}\n'.format(dev_nll_loss))
    print('dev NLL loss {}'.format(dev_nll_loss))

    # data division for discriminator
    dis_minibatches = [(start, start + args.adv_dis_batch_size) \
                   for start in range(0, inp.shape[0], args.adv_dis_batch_size)][:-1]  ## return a list[(0,2),(2,4)]
    random.shuffle(dis_minibatches)
    dis_dev_minibatches = dis_minibatches[:args.adv_dis_dev_batches]

    for epoch in range(args.adv_train_epochs):
        print('\n--------\nEPOCH %d\n--------' % (epoch + 1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ')
        adv_gen_log.write('\n--------\nEPOCH %d\n--------\n\n' % (epoch + 1))
        gen_sen.write('\n--------\nEPOCH %d\n--------\n\n' % (epoch + 1))

        sys.stdout.flush()
        dev_nll_loss = train_generator_PG(gen, gen_minibatches, gen_dev_minibatches, gen_optimizer, dis, args)

        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        adv_dis_log.write('\n--------\nEPOCH %d\n--------\n\n' % (epoch + 1))
        acc = train_discriminator(dis, gen, dis_minibatches, dis_dev_minibatches, args.adv_dis_batch_size, args.adv_dis_dev_batches, args.adv_d_steps, args.adv_dis_train_batches,
                            args.adv_dis_epochs, args.adv_dis_save_freq)

        if dev_nll_loss<best_nll_loss:
            best_nll_loss = dev_nll_loss

            ##save model
            torch.save({'state_dict': gen.state_dict(),
                        'dev_nll_loss':dev_nll_loss,
                        'epoch':epoch+1,
                        'config_args': args}, args.adv_gen_model_save_path)
            torch.save({'state_dict': dis.state_dict(),
                        'val_acc': acc,
                        'epoch':epoch+1,
                        'config_args': args}, args.adv_dis_model_save_path)
            print('model save')

    adv_dis_log.close()
    adv_gen_log.close()
    gen_sen.close()




