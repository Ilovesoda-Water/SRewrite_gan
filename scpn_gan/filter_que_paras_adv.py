import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='models/adv_dis.pt')
    parser.add_argument('--filter_before', type=str, default='wiki_data/wiki_que_para_test_adv.txt')
    parser.add_argument('--paragram_sl999', type=str, default='../emnlp2017-master/data/paragram_sl999_small.txt')


