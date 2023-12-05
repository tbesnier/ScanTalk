import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d, LeakyReLU, Dropout
from wav2vec import Wav2Vec2Model
import sys
sys.path.append('/home/federico/Scrivania/ST/ScanTalk/model/diffusion-net/src')
import diffusion_net
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
    def __init__(self, in_channels, out_channels, indices, dim=1, init=False):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
                
        if init==True:
            nn.init.constant_(self.layer.weight, 0)
            nn.init.constant_(self.layer.bias, 0)
        else:
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




class DiffusionNetAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, spiral_indices, down_transform, up_transform):
        super(DiffusionNetAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = self.down_transform[-1].size(0)

        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_encoder.feature_extractor._freeze_parameters()


        # encoder
        self.encoder = diffusion_net.layers.DiffusionNet(C_in=in_channels,
                                          C_out=latent_channels,
                                          C_width=32, 
                                          N_block=2, 
                                          last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
                                          outputs_at='global_mean', 
                                          dropout=False)
        # decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_channels, self.num_vert * out_channels[-1]))
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
            SpiralConv(out_channels[0], in_channels, self.spiral_indices[0], init=True))
        
        self.audio_embedding = nn.Linear(768, latent_channels)
        self.lstm = nn.LSTM(input_size=latent_channels*2, hidden_size=int(latent_channels/2), num_layers=3, batch_first=True, bidirectional=True)

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

    def forward(self, audio, actor, vertices, mass, L, evals, evecs, gradX, gradY):
        hidden_states = self.audio_encoder(audio, frame_num=len(vertices)).last_hidden_state
        pred_sequence = actor
        audio_emb = self.audio_embedding(hidden_states)
        actor_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)
        actor_emb = actor_emb.expand(audio_emb.shape)
        latent, _ = self.lstm(torch.cat([audio_emb, actor_emb], dim=2))
        for k in range(latent.shape[1]):
            pred = self.decode(latent[:, k, :]) + actor
            pred_sequence = torch.vstack([pred_sequence, pred])
        return pred_sequence[1:, :, :]

    def predict(self, audio, actor, mass, L, evals, evecs, gradX, gradY):
        hidden_states = self.audio_encoder(audio).last_hidden_state
        pred_sequence = actor
        audio_emb = self.audio_embedding(hidden_states)
        actor_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)
        actor_emb = actor_emb.expand(audio_emb.shape)
        latent, _ = self.lstm(torch.cat([audio_emb, actor_emb], dim=2))
        for k in range(latent.shape[1]):
            pred = self.decode(latent[:, k, :]) + actor
            pred_sequence = torch.vstack([pred_sequence, pred])
        return pred_sequence[1:, :, :]

