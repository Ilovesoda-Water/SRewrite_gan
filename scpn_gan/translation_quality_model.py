import torch
import numpy as np
from torch import nn, autograd
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.functional as F

class translation_quality_model(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, vocab_size, We_vocab, gpu=True):
        super(translation_quality_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.gpu = gpu

        # use pretrained embedding
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        pretrained_weight = np.array(We_vocab)
        self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_weight))

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init__hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))
        c = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))
        if self.gpu:
            return h.cuda(), c.cuda()
        else:
            return h, c

    def forward(self, input, lengths, h_init, c_init):
        emb = self.embeddings(input)

        lens, indices = torch.sort(lengths, 0, True)

        _, (hidden, _) = self.lstm(pack(emb[indices], lens.tolist(), batch_first=True), (h_init, c_init))
        _, _indices = torch.sort(indices, 0)

        hidden = hidden.squeeze(0)[_indices]
        out = self.hidden2out(hidden)
        out = F.sigmoid(out)
        return out

    def batch_Classify(self, inp, lengths):
        h, c = self.init__hidden(inp.size()[0])
        out = self.forward(inp, lengths, h, c)
        return out.view(-1)
