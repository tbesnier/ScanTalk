import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from .pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation
from .spiralnet import SpiralConv
from wav2vec import Wav2Vec2Model
from tqdm import tqdm

from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d, ReLU, LeakyReLU, Dropout, LayerNorm, GroupNorm, InstanceNorm1d

def MLP(channels, bias=False, normalization="batch", nonlin=LeakyReLU(negative_slope=0.2)):

    if normalization=="group":
        return Seq(*[
            Seq(Lin(channels[i - 1], channels[i], bias=bias), GroupNorm(channels[i]), nonlin)
            for i in range(1, len(channels))
        ])

    if normalization=="layer":
        return Seq(*[
            Seq(Lin(channels[i - 1], channels[i], bias=bias), LayerNorm(channels[i]), nonlin)
            for i in range(1, len(channels))
        ])

    if normalization=="batch":
        return Seq(*[
            Seq(Lin(channels[i - 1], channels[i], bias=bias), BatchNorm1d(channels[i]), nonlin)
            for i in range(1, len(channels))
        ])

    if normalization=="instance":
        return Seq(*[
            Seq(Lin(channels[i - 1], channels[i], bias=bias), InstanceNorm1d(channels[i]), nonlin)
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


class PointNet2(nn.Module):
    def __init__(self, latent_channels, per_point_features=False, normal_channel=True):
        super(PointNet2, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.per_point_features = per_point_features
        self.sa1 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=32, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + in_channel, mlp=[128, 128, 128])

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.InstanceNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.InstanceNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, latent_channels)

        self.conv1 = nn.Conv1d(128, 8, 1)
        self.bn0 = nn.InstanceNorm1d(8)

    def forward(self, xyz):
        B, C, N = xyz.shape
        xyz = xyz.permute(0, 2, 1)
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        #cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)
        l0_points = self.fp1(xyz, l1_xyz, xyz, l1_points)

        # FC layers
        per_point_feat = F.relu(self.bn0(self.conv1(l0_points)))

        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        #print(x.shape)
        #print(per_point_feat[0, :, 0])
        #print(per_point_feat[0, :, 1])
        if self.per_point_features:
            return x, per_point_feat
        else:
            return x


class PointNet2Encoder(nn.Module):
    def __init__(self, latent_channels, per_point_features=False, normal_channel=False):
        nn.Module.__init__(self)
        self.per_point_features = per_point_features
        self.ptnet2 = PointNet2(latent_channels, per_point_features, normal_channel)

    def forward(self, x):

        if self.per_point_features:
            x, per_point_feat = self.ptnet2(x)
            return x, per_point_feat
        else:
            return self.ptnet2(x)


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
        x_hat = self.decode(x)
        return x_hat


class NJFDecoder_Conv1d(nn.Module):
    def __init__(self, latent_channel, point_dim=3, device="cuda:0"):
        super(NJFDecoder_Conv1d, self).__init__()
        self.latent_channel = latent_channel
        self.point_dim = point_dim
        self.device = device
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv1d(latent_channel + point_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 64, 1)
        self.conv3 = nn.Conv1d(64, 32, 1)
        self.last_layer = nn.Conv1d(32, 3, 1)
        self.bn1 = nn.InstanceNorm1d(128)
        self.bn2 = nn.InstanceNorm1d(64)
        self.bn3 = nn.InstanceNorm1d(32)
        self.bn4 = nn.InstanceNorm1d(3)

        #nn.init.constant_(self.last_layer.weight, 0)
        #nn.init.constant_(self.last_layer.bias, 0)


    def decode(self, z, actor_per_point_feat):

        NV = actor_per_point_feat.shape[-1]
        latent_all = z.unsqueeze(-1).expand(-1, -1, NV)
        #actor = actor_per_point_feat.transpose(1, 2)
        x = torch.cat([actor_per_point_feat, latent_all], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.last_layer(x))
        print(self.last_layer.weight[0])

        return x.transpose(1, 2)

    def forward(self, x, actor_per_point_feat):
        x_hat = self.decode(x, actor_per_point_feat=actor_per_point_feat)
        return x_hat


class NJFDecoder(nn.Module):
    def __init__(self, latent_channel, point_dim=3, device="cuda:0", init=True):
        super(NJFDecoder, self).__init__()
        self.latent_channel = latent_channel
        self.point_dim = point_dim
        self.device = device
        hid_shape=64
        linear_layer = nn.Linear
        self.linears = [linear_layer(self.point_dim + self.latent_channel, hid_shape, bias=False)]
        for _ in range(4):
            self.linears.append(linear_layer(hid_shape, hid_shape, bias=False))

        self.gns = [nn.InstanceNorm1d(hid_shape) for _ in range(len(self.linears))]
        self.gns = nn.ModuleList(self.gns)

        self.linears = nn.ModuleList(self.linears)
        self.linear_out = linear_layer(hid_shape, 3)

        self.relu = nn.ReLU()

        if init == True:
            nn.init.constant_(self.linear_out.weight, 0)
            nn.init.constant_(self.linear_out.bias, 0)


    def decode(self, z, actor_per_point_feat):

        NV = actor_per_point_feat.shape[-1]
        latent_all = z.unsqueeze(1).expand(1, NV, -1)
        actor_per_point_feat = actor_per_point_feat.transpose(1,2)

        x = torch.cat([actor_per_point_feat, latent_all], dim=-1)
        out = x
        for _ in range(len(self.linears)):
            out = torch.transpose(self.relu(self.gns[_](torch.transpose(self.linears[_](out), -1, -2))), -1, -2)

        out = self.linear_out(out)
        return out

    def forward(self, x, actor_per_point_feat):
        x_hat = self.decode(x, actor_per_point_feat=actor_per_point_feat)
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
                            num_layers=3, batch_first=True, bidirectional=True)

    def get_latent(self, x):
        z = self.encode(x)
        return z

    def forward(self, audio, actor, vertices):
        hidden_states = self.audio_encoder(audio, frame_num=len(vertices)).last_hidden_state
        pred_sequence = actor
        audio_emb = self.audio_embedding(hidden_states)
        actor_emb = self.encode(actor)
        actor_emb = actor_emb.expand(audio_emb.shape)
        latent, _ = self.lstm(torch.cat([audio_emb, actor_emb], dim=2))
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
        latent, _ = self.lstm(torch.cat([audio_emb, actor_emb], dim=2))
        for k in range(latent.shape[1]):
            pred = self.decode(latent[:, k, :]) + actor
            pred_sequence = torch.vstack([pred_sequence, pred])
        return pred_sequence[1:, :, :]

class PointNet2NJFAutoEncoder(nn.Module):
    def __init__(self, latent_channels, point_dim=8, device="cuda:0"):
        nn.Module.__init__(self)
        self.device = device
        per_point_features = False
        if point_dim>3:
            per_point_features = True
        self.encode = PointNet2Encoder(latent_channels, per_point_features=per_point_features)
        #self.decode = SpiralNet2Decoder(in_channels, out_channels, latent_channels, spiral_indices,
        #                                down_transform, up_transform)
        self.decode = NJFDecoder(latent_channel=latent_channels, point_dim=point_dim, device=device)

        #self.decode = MLPDecoder(latent=latent_channels)
        self.latent_channels = latent_channels
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_encoder.feature_extractor._freeze_parameters()

        self.audio_embedding = nn.Linear(768, self.latent_channels)
        self.lstm = nn.LSTM(input_size=2*self.latent_channels, hidden_size=int(self.latent_channels / 2),
                            num_layers=3, batch_first=True, bidirectional=True)

    def get_latent(self, x):
        z, points_features = self.encode(x)
        return z

    def forward(self, audio, actor, vertices):
        hidden_states = self.audio_encoder(audio, frame_num=len(vertices)).last_hidden_state
        pred_sequence = actor
        audio_emb = self.audio_embedding(hidden_states)
        actor_emb, per_point_actor_emb = self.encode(actor)
        actor_emb = actor_emb.expand(audio_emb.shape)
        latent, _ = self.lstm(torch.cat([audio_emb, actor_emb], dim=2))
        for k in range(latent.shape[1]):
            d = self.decode(latent[:, k, :], per_point_actor_emb)
            pred = d + actor
            pred_sequence = torch.vstack([pred_sequence, pred])

        return pred_sequence[1:, :, :]

    def predict(self, audio, actor):
        hidden_states = self.audio_encoder(audio).last_hidden_state
        pred_sequence = actor
        audio_emb = self.audio_embedding(hidden_states)
        actor_emb, per_point_actor_emb = self.encode(actor)
        actor_emb = actor_emb.expand(audio_emb.shape)
        latent, _ = self.lstm(torch.cat([audio_emb, actor_emb], dim=2))
        for k in range(latent.shape[1]):
            pred = self.decode(latent[:, k, :], per_point_actor_emb) + actor
            pred_sequence = torch.vstack([pred_sequence, pred])
        return pred_sequence[1:, :, :]

    # def forward(self, audio, actor, vertices):
    #     hidden_states = self.audio_encoder(audio, frame_num=len(vertices)).last_hidden_state
    #     pred_sequence = actor
    #     audio_emb = self.audio_embedding(hidden_states)
    #     actor_vertices_emb = self.encode(actor)
    #     latent, _ = self.lstm(audio_emb)
    #     for k in range(latent.shape[1]):
    #         pred_points = torch.zeros([1, 1, 3]).to('cuda:0')
    #         batches = torch.split(actor_vertices_emb, 512, dim=1)
    #         for batch in batches:
    #             decoder_input = torch.cat([batch, latent[:, k, :].expand(batch.shape)], dim=-1)
    #             pred_points = torch.hstack([pred_points, self.global_decoder(decoder_input).unsqueeze(0)])
    #         pred = pred_points[:, 1:, :] + actor
    #         pred_sequence = torch.vstack([pred_sequence, pred])
    #     return pred_sequence[1:, :, :]
    #
    # def predict(self, audio, actor):
    #     hidden_states = self.audio_encoder(audio).last_hidden_state
    #     pred_sequence = actor
    #     audio_emb = self.audio_embedding(hidden_states)
    #     actor_vertices_emb = self.encode(actor)
    #     latent, _ = self.lstm(audio_emb)
    #     for k in range(latent.shape[1]):
    #         pred_points = torch.zeros([1, 1, 3]).to('cuda:0')
    #         batches = torch.split(actor_vertices_emb, 512, dim=1)
    #         for batch in batches:
    #             decoder_input = torch.cat([batch, latent[:, k, :].expand(batch.shape)], dim=-1)
    #             pred_points = torch.hstack([pred_points, self.global_decoder(decoder_input).unsqueeze(0)])
    #         pred = pred_points[:, 1:, :] + actor
    #         pred_sequence = torch.vstack([pred_sequence, pred])
    #     return pred_sequence[1:, :, :]