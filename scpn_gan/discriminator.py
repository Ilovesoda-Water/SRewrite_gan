import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence as pack

class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, We_vocab, gpu=True, dropout=0.):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.gpu = gpu

        # use pretrained embedding
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        pretrained_weight = np.array(We_vocab)
        self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_weight))

        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout, batch_first=True)
        self.gru2hidden = nn.Linear(2*2*hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init__hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, input, lengths, hidden):
        emb = self.embeddings(input)
        emb = emb

        lens, indices = torch.sort(lengths, 0, True)

        # _,hidden = self.gru(emb, hidden)
        _, hidden = self.gru(pack(emb[indices], lens.tolist(), batch_first=True), hidden)
        _, _indices = torch.sort(indices, 0)

        hidden = (hidden.permute(1, 0, 2))[_indices].contiguous()
        out = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))
        out = F.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)
        out = F.sigmoid(out)
        return out

    def batch_Classify(self, inp, lengths):
        h = self.init__hidden(inp.size()[0])
        out = self.forward(inp, lengths, h)
        return out.view(-1)
