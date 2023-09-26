import torch, time, os, sys, numpy as np
from torch import nn, optim

class SpiralConv(nn.Module):
    def __init__(self, in_c, spiral_size, out_c, activation='elu', bias=True, device=None):
        super(SpiralConv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.device = device

        self.conv = nn.Linear(in_c * spiral_size, out_c, bias=bias)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.02)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x, spiral_adj):
        bsize, num_pts, feats = x.size()
        _, _, spiral_size = spiral_adj.size()

        spirals_index = spiral_adj.view(bsize * num_pts * spiral_size).long()  # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=self.device).view(-1, 1).repeat([1, num_pts * spiral_size]).view(
            -1).long()  # [0*numpt,1*numpt,etc.]
        spirals = x[batch_index, spirals_index, :].view(bsize * num_pts,
                                                        spiral_size * feats)  # [bsize*numpt, spiral*feats]

        out_feat = self.conv(spirals)
        out_feat = self.activation(out_feat)

        out_feat = out_feat.view(bsize, num_pts, self.out_c)
        zero_padding = torch.ones((1, x.size(1), 1), device=self.device)
        zero_padding[0, -1, 0] = 0.0
        out_feat = out_feat * zero_padding

        return out_feat