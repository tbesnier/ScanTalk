import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from transformers import Wav2Vec2Processor
from wav2vec import Wav2Vec2Model
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d, LeakyReLU, Dropout

import pdb

def MLP(channels, bias=False, nonlin=LeakyReLU(negative_slope=0.2)):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i], bias=bias), BatchNorm1d(channels[i]), nonlin)
        for i in range(1, len(channels))
    ])

def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralEnblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, down_transform):
        out = F.elu(self.conv(x))
        out = Pool(out, down_transform)
        return out


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralDeblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out))
        return out

class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))

        x = self.layer(x)
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)



class SpiralAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels,
                 spiral_indices, down_transform, up_transform):
        super(SpiralAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = self.down_transform[-1].size(0)

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx]))
            else:
                self.en_layers.append(
                    SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx]))
        self.en_layers.append(
            nn.Linear(self.num_vert * out_channels[-1], latent_channels))

        # decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_channels*2, self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx - 1],
                                  out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
            else:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx], out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
        self.de_layers.append(
            SpiralConv(out_channels[0], in_channels, self.spiral_indices[0]))

        #self.audio_embedding = #Seq(Lin(768, 1024, bias=False), Dropout(0.5), Lin(1024, self.latent_channels))
        #self.audio_embedding = nn.LSTM(input_size=768, hidden_size=1024, num_layers=3, batch_first=True , bidirectional=True)
        self.audio_embedding = nn.LSTM(input_size=768, hidden_size=self.latent_channels, num_layers=5, batch_first=True, bidirectional=True)
        #self.fc = Lin(1024, self.latent_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def encode(self, x):
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        return x

    def decode(self, x):
        num_layers = len(self.de_layers)
        num_features = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.up_transform[num_features - i])
            else:
                x = layer(x)
        return x

    def predict(self, audio, actor):

        pred_sequence = actor
        for i in range(audio.shape[1]):
            if i == 0:

                z = self.encode(actor - actor)
                z = torch.cat([z, self.audio_embedding(audio[:, i, :])], dim=1)
                x = self.decode(z) + actor
                pred_sequence = torch.vstack([pred_sequence, x])
            else:
                z = self.encode(x - actor)
                z = torch.cat([z, self.audio_embedding(audio[:, i, :])], dim=1)
                x = self.decode(z) + actor
                pred_sequence = torch.vstack([pred_sequence, x])

        return pred_sequence[1:, :, :]

    def predict_new(self, audio, actor):

        pred_sequence = actor
        for i in range(audio.shape[1]):
            #z = self.encode(actor)
            if i<5:
                audio_data = torch.mean(torch.stack([audio[:, i, :] for i in range(0, i + 5)]), axis=0)
            elif i>=5 and i+5<audio.shape[1]:
                audio_data = torch.mean(torch.stack([audio[:,i,:] for i in range(i-5, i+5)]), axis=0)
            else:
                audio_data = torch.mean(torch.stack([audio[:,i,:] for i in range(i-5, audio.shape[1])]), axis=0)

            audio_emb, _ = self.audio_embedding(audio_data.unsqueeze(0))
            #audio_emb = self.audio_embedding(audio[:,i,:])
            #z = torch.cat([z, audio_emb[0]], dim=1)
            z = audio_emb[0]
            x = self.decode(z) + actor
            pred_sequence = torch.vstack([pred_sequence, x])

        return pred_sequence[1:, :, :]

    def forward(self, audio, actor):
        audio_emb, _ = self.audio_embedding(audio.unsqueeze(0))
        #audio_emb = self.audio_embedding(audio)
        #z = self.encode(actor)
        #z = torch.cat([z, audio_emb[0]], dim=1)
        z = audio_emb[0]
        pred = self.decode(z)
        return pred + actor

