import torch
from torch import nn
from torch.nn import functional as F
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
from .spiralnet import SpiralConv


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
        # x = F.log_softmax(x, -1)

        return x


class PointNet2(nn.Module):
    def __init__(self, normal_channel=False):
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
        self.fc3 = nn.Linear(256, 128)

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

        return x, l1_xyz, l2_xyz, l3_xyz, l1_points, l2_points, l3_points

    def use_sa(self, l_xyz, l_points):
        B, _, _ = l_xyz.shape
        l3_xyz, l3_points = self.sa3(l_xyz, l_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x, l3_xyz, l3_points


class Encoder(nn.Module):
    def __init__(self, latent=512, filters_fc=[32, 128, 256], desc=512, filters_conv=[512, 256, 128],
                 normal_channel=False):
        nn.Module.__init__(self)
        self.ptnet2 = PointNet2(normal_channel)

    def forward(self, x):
        x, l1_xyz, l2_xyz, l3_xyz, l1_points, l2_points, l3_points = self.ptnet2(x)
        return x, l1_xyz, l2_xyz, l3_xyz, l1_points, l2_points, l3_points

    def use_sa(self, l_xyz, l_points):
        x, l3_xyz, l3_points = self.ptnet2.use_sa(l_xyz, l_points)

        return x, l3_xyz, l3_points


class Decoder(nn.Module):
    def __init__(self, latent=512, filters=[1024, 2048], num_points=5023):
        nn.Module.__init__(self)
        self.num_points = num_points
        size = latent
        layers = []
        for f_size in filters:
            layers.append(nn.Linear(size, f_size))
            layers.append(nn.LeakyReLU())
            size = f_size
        layers.append(nn.Linear(size, num_points * 3))
        self.fc1 = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc1(x).view(x.shape[0], self.num_points, -1)
        return x


class Decoder_spirals(nn.Module):
    def __init__(self, filters_dec, latent_size, sizes, spiral_sizes, spirals, U, device=None,
                 activation='elu'):
        nn.Module.__init__(self)
        # super(PointNetSpiralAutoencoder, self).__init__()
        self.latent_size = latent_size
        self.sizes = sizes
        self.spirals = spirals
        self.filters_dec = filters_dec
        self.spiral_sizes = spiral_sizes
        self.U = U
        self.device = device
        self.activation = activation

        self.fc_latent_dec = nn.Linear(latent_size, (sizes[-1] + 1) * filters_dec[0][0])
        self.dconv = []
        input_size = filters_dec[0][0]
        for i in range(len(spiral_sizes) - 1):
            if i != len(spiral_sizes) - 2:
                self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[0][i + 1],
                                             activation=self.activation, device=device).to(device))
                input_size = filters_dec[0][i + 1]

                if filters_dec[1][i + 1]:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[1][i + 1],
                                                 activation=self.activation, device=device).to(device))
                    input_size = filters_dec[1][i + 1]
            else:
                if filters_dec[1][i + 1]:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[0][i + 1],
                                                 activation=self.activation, device=device).to(device))
                    input_size = filters_dec[0][i + 1]
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[1][i + 1],
                                                 activation='identity', device=device).to(device))
                    input_size = filters_dec[1][i + 1]
                else:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[0][i + 1],
                                                 activation='identity', device=device).to(device))
                    input_size = filters_dec[0][i + 1]

        self.dconv = nn.ModuleList(self.dconv)

    def decode(self, z):
        bsize = z.size(0)
        S = self.spirals
        U = self.U
        x = self.fc_latent_dec(z)
        x = x.view(bsize, self.sizes[-1] + 1, -1)
        j = 0
        for i in range(len(self.spiral_sizes) - 1):
            x = torch.matmul(U[-1 - i], x)
            x = self.dconv[j](x, S[-2 - i].repeat(bsize, 1, 1))
            j += 1
            if self.filters_dec[1][i + 1]:
                x = self.dconv[j](x, S[-2 - i].repeat(bsize, 1, 1))
                j += 1
        return x

    def forward(self, x):
        bsize = x.size(0)
        x_hat = self.decode(x)
        return x_hat


class PointNet2AutoEncoder(nn.Module):
    def __init__(self, latent_size=512, filter_enc=[[32, 128, 256], 512, [512, 256, 128]], filter_dec=[1024, 2048],
                 num_points=5023, device="cuda:0", normal_channel=False):
        nn.Module.__init__(self)
        self.device = device
        self.encoder = Encoder(latent_size, filter_enc[0], filter_enc[1], filter_enc[2], normal_channel)
        self.decoder = Decoder(latent_size, filter_dec, num_points)

    def forward(self, x):
        z = self.encoder(x)[0]

        return self.decoder(z)

    def get_latent(self, x):
        z = self.encoder(x)[0]
        l1_xyz, l1_points = self.encoder(x)[1], self.encoder(x)[4]
        l2_xyz, l2_points = self.encoder(x)[2], self.encoder(x)[5]
        l3_xyz, l3_points = self.encoder(x)[3], self.encoder(x)[6]

        return (z, l1_xyz, l2_xyz, l3_xyz, l1_points, l2_points, l3_points)

    def use_sa(self, l_xyz, l_points):
        z = self.encoder.use_sa(l_xyz, l_points)[0]

        return (self.decoder(z))


class PointNet2SpiralsAutoEncoder(nn.Module):
    def __init__(self, latent_size, filter_enc, filter_dec, num_points, sizes,
                 spiral_sizes,
                 spirals,
                 U, device="cuda:0"):
        nn.Module.__init__(self)
        self.device = device
        self.encoder = Encoder(latent_size, filter_enc[0], filter_enc[1], filter_enc[2])
        self.decoder = Decoder_spirals(filter_dec, latent_size, sizes, spiral_sizes, spirals, U, device=device)

    def forward(self, x):
        z = self.encoder(x)[0]
        return self.decoder(z)

    def get_latent(self, x):
        z = self.encoder(x)[0]
        l1_xyz, l1_points = self.encoder(x)[1], self.encoder(x)[4]
        l2_xyz, l2_points = self.encoder(x)[2], self.encoder(x)[5]
        l3_xyz, l3_points = self.encoder(x)[3], self.encoder(x)[6]

        return (z, l1_xyz, l2_xyz, l3_xyz, l1_points, l2_points, l3_points)

    def use_sa(self, l_xyz, l_points):
        z = self.encoder.use_sa(l_xyz, l_points)[0]

        return (self.decoder(z))


if __name__ == '__main__':
    model = PointNet2AutoEncoder()
    init_random_weights = False

    if init_random_weights == True:
        x = torch.randn((2, 5023, 3))
    else:
        x = torch.zeros((2, 5023, 3))
    print(model)
    print(model(x).shape)