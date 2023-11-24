import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d, LeakyReLU, Dropout
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
from .spiralnet import SpiralConv
from wav2vec import Wav2Vec2Model


import pdb

#def MLP(channels, bias=False, nonlin=LeakyReLU(negative_slope=0.2)):
#    return Seq(*[
#        Seq(Lin(channels[i - 1], channels[i], bias=bias), BatchNorm1d(channels[i]), nonlin)
#        for i in range(1, len(channels))
#    ])

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

        if init == True:
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

class PointNet2MSG(nn.Module):
    def __init__(self, normal_channel=True):
        super(PointNet2, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 128)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x


class PointNet2(nn.Module):
    def __init__(self, latent_channels, normal_channel=False):
        super(PointNet2, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=32, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, latent_channels)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        xyz = xyz.permute(0, 2, 1)
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x

class PointNet2Encoder(nn.Module):
    def __init__(self, latent_channels, normal_channel=False):
        nn.Module.__init__(self)
        self.ptnet2 = PointNet2(latent_channels, normal_channel)

    def forward(self, x):
        x = self.ptnet2(x)
        return x

class SpiralNet2Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels,
                 spiral_indices, down_transform, up_transform):
        super(SpiralNet2Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = self.down_transform[-1].size(0)
        # decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_channels + 6 * 16, self.num_vert * out_channels[-1]))
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

        # self.audio_embedding = nn.LSTM(input_size=768, hidden_size=1024, num_layers=3, batch_first=True , bidirectional=True)
        # self.audio_embedding = nn.LSTM(input_size=768, hidden_size=self.latent_channels, num_layers=5, batch_first=True, bidirectional=True)
        # self.fc = Lin(1024, self.latent_channels, bias=False)

        #self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def reset_parameters_to_zero(self):
        for name, param in self.named_parameters():
            nn.init.constant_(param, 0)

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

    def forward(self, x):
        #bsize = x.size(0)
        x_hat = self.decode(x)
        return x_hat

class PointNet2SpiralsAutoEncoder(nn.Module):
    def __init__(self, latent_channels, in_channels, out_channels,
                 spiral_indices, down_transform, up_transform, device="cuda:0"):
        nn.Module.__init__(self)
        self.device = device
        self.encode = PointNet2Encoder(latent_channels)
        self.decode = SpiralNet2Decoder(in_channels, out_channels, latent_channels, spiral_indices,
                                         down_transform, up_transform)
        
        self.latent_channels = latent_channels

        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_embedding = nn.Linear(768, self.latent_channels)
        self.lstm = nn.LSTM(input_size=self.latent_channels * 2, hidden_size=int(self.latent_channels / 2),
                            num_layers=5, batch_first=True, bidirectional=True)
    def get_latent(self, x):
        z = self.encode(x)
        return z

    # def predict_cat_audio(self, audio, actor, n=3):
    #
    #     pred_sequence = actor
    #     for i in range(audio.shape[1]):
    #         z = self.encode(actor)
    #         if i<n:
    #             audio_data = [audio[:, 0, :] for j in range(n - i)] + [audio[:,j,:] for j in range(0, i+n)]
    #         elif i>=n and i+n<audio.shape[1]:
    #             audio_data = [audio[:, k, :] for k in range(i - n, i + n)]
    #         else:
    #             audio_data = [audio[:, k, :] for k in range(i - n, audio.shape[1])] + (i + n - audio.shape[1])*[audio[:,-1,:]]
    #         audio_emb = self.audio_embedding(audio_data[0])
    #         for l in range(1,6):
    #             audio_emb = torch.cat([audio_emb, self.audio_embedding(audio_data[l])], dim=1)
    #         z = torch.cat([z, audio_emb], dim=1)
    #         x = self.decode(z) + actor
    #         pred_sequence = torch.vstack([pred_sequence, x])
    #
    #     return pred_sequence[1:, :, :]

    # def forward(self, audio, actor):
    #     z = self.encode(actor)
    #     #audio_emb, _ = self.audio_embedding(audio.unsqueeze(0))
    #     for i in range(audio.shape[1]):
    #         audio_emb = self.audio_embedding(audio[:,i,:])
    #         z = torch.cat([z, audio_emb], dim=1)
    #     #z = audio_emb[0]
    #     pred = self.decode(z)
    #     return pred + actor

    def forward(self, audio, actor, vertices):
        hidden_states = self.audio_encoder(audio, frame_num=len(vertices)).last_hidden_state
        pred_sequence = actor
        audio_emb = self.audio_embedding(hidden_states)
        actor_emb = self.encode(actor)
        actor_emb = actor_emb.expand(audio_emb.shape)
        latent = torch.cat([audio_emb, actor_emb], dim=2)
        for k in range(latent.shape[1]):
            pred = self.decode(latent[:, k, :]) + actor
            pred_sequence = torch.vstack([pred_sequence, pred])
        return pred_sequence[1:, :, :]

    def predict(self, audio, actor):
        hidden_states = self.audio_encoder(audio).last_hidden_state
        pred_sequence = actor
        audio_emb = self.audio_embedding(hidden_states)
        actor_emb = self.encode(actor)
        actor_emb = actor_emb.expand(audio_emb.shape)
        latent = torch.cat([audio_emb, actor_emb], dim=2)
        for k in range(latent.shape[1]):
            pred = self.decode(latent[:, k, :]) + actor
            pred_sequence = torch.vstack([pred_sequence, pred])
        return pred_sequence[1:, :, :]
