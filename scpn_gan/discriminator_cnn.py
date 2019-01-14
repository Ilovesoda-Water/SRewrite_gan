import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Discriminator_CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_sizes, num_filters, num_classes=2, dropout=0.25):
        super(Discriminator_CNN, self).__init__()
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.convs = []
        for filter_size, num_filter in zip(filter_sizes, num_filters):
            conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=num_filter,
                    kernel_size=(filter_size,embedding_dim),
                    stride=1
                ),
                nn.ReLU()
            )
            self.convs.append(conv.cuda())
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.highway_lin_linear = nn.Linear(sum(self.num_filters), sum(self.num_filters))
        self.highway_gate_linear = nn.Linear(sum(self.num_filters), sum(self.num_filters))
        self.dropout_linear = nn.Dropout(p=dropout)
        self.dropout2out = nn.Linear(sum(self.num_filters),num_classes)

    def forward(self, inp, target):
        emb = self.embeddings(inp)
        emb = emb.unsqueeze(1)     # batch_size * 1 * seq_len * embedding_dim
        seq_len = inp.size()[1]
        pooled_outputs = []
        for i in range(len(self.filter_sizes)):
            h = self.convs[i](emb)
            pooled = nn.MaxPool2d((seq_len-self.filter_sizes[i]+1,1))(h)
            pooled_outputs.append(pooled)
        num_filters_total = sum(self.num_filters)
        h_pool = torch.cat(pooled_outputs,1).contiguous()
        h_pool_flat = h_pool.view(-1, num_filters_total)
        t = self.sigmoid(self.highway_gate_linear(h_pool_flat))
        g = self.relu(self.highway_lin_linear(h_pool_flat))
        h_highway = t * g + (1. - t) * h_pool_flat
        h_drop = self.dropout_linear(h_highway)
        scores = self.dropout2out(h_drop)
        return scores

