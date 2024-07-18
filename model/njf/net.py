from torch import nn
import torch
from model.njf.SourceMesh import SourceMesh
USE_CUPY = False
FREQUENCY = 100 # frequency of logguing - every FREQUENCY iteration step
UNIT_TEST_POISSON_SOLVE = False


class njf_decoder(nn.Module):

    def __init__(self, latent_features_shape, args, point_dim=6, verbose=False):
        print("********** Some Network info...")
        print(f"********** code dim: {latent_features_shape}")
        print(f"********** centroid dim: {point_dim}")
        super().__init__()
        self.args = args
        self.latent_size = latent_features_shape

        layer_normalization = self.get_layer_normalization_type()
        if layer_normalization == "IDENTITY":
            # print("Using IDENTITY (no normalization) in per_face_decoder!")
            self.per_face_decoder = nn.Sequential(nn.Linear(self.latent_size, 128),
                                                  nn.Identity(),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.Identity(),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.Identity(),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.Identity(),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 9))
        elif layer_normalization == "BATCHNORM":
            # print("Using BATCHNORM in per_face_decoder!")
            self.per_face_decoder = nn.Sequential(nn.Linear(self.latent_size, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 9))
        elif layer_normalization == "GROUPNORM_CONV":
            # print("Using GROUPNORM2 in per_face_decoder!")
            self.per_face_decoder = nn.Sequential(nn.Conv1d(self.latent_size, 128, 1),
                                                  nn.GroupNorm(num_groups=4, num_channels=128),
                                                  nn.ReLU(),
                                                  nn.Conv1d(128, 128, 1),
                                                  nn.GroupNorm(num_groups=4, num_channels=128),
                                                  nn.ReLU(),
                                                  nn.Conv1d(128, 128, 1),
                                                  nn.GroupNorm(num_groups=4, num_channels=128),
                                                  nn.ReLU(),
                                                  nn.Conv1d(128, 128, 1),
                                                  nn.GroupNorm(num_groups=4, num_channels=128),
                                                  nn.ReLU(),
                                                  nn.Conv1d(128, 9, 1))
        elif layer_normalization == "GROUPNORM":
            # print("Using GROUPNORM in per_face_decoder!")
            self.per_face_decoder = nn.Sequential(nn.Linear(self.latent_size, 128),
                                                  nn.GroupNorm(num_groups=4, num_channels=128),
                                                  # , eps=0.0001 I have considered increasing this value in case we have channels from pointnet with the same values.
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.GroupNorm(num_groups=4, num_channels=128),
                                                  # , eps=0.0001 I have considered increasing this value in case we have channels from pointnet with the same values.
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.GroupNorm(num_groups=4, num_channels=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.GroupNorm(num_groups=4, num_channels=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 9))
        elif layer_normalization == "LAYERNORM":
            # print("Using LAYERNORM in per_face_decoder!")
            self.per_face_decoder = nn.Sequential(nn.Linear(self.latent_size, 128),
                                                  nn.LayerNorm(normalized_shape=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.LayerNorm(normalized_shape=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.LayerNorm(normalized_shape=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.LayerNorm(normalized_shape=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 9))
        else:
            raise Exception("unknown normalization method")

        self.__IDENTITY_INIT = True
        if self.__IDENTITY_INIT:
            self.per_face_decoder._modules["12"].bias.data.zero_()
            self.per_face_decoder._modules["12"].weight.data.zero_()

        self.__global_trans = False
        self.point_dim = point_dim
        self.verbose = verbose
        self.mse = nn.MSELoss()
        self.log_validate = True
        self.val_step_iter = 0

    ##################
    # inference code below
    ##################
    def forward(self, x):

        pred = self.per_face_decoder(x.type(self.per_face_decoder[0].bias.type()))
        return pred

    def predict_jacobians(self, latent_features):

        return self.predict_jacobians_from_codes(latent_features)

    def predict_jacobians_from_codes(self, latent_features):

        stacked = latent_features ### B x F x nb_features

        stacked = stacked.view(latent_features.shape[0]*latent_features.shape[1], self.latent_size)
        res = self.forward(stacked)
        res = res.view(latent_features.shape[0],latent_features.shape[1],-1)
        pred_J = res
        ret = pred_J.reshape(pred_J.shape[0], pred_J.shape[1], 3, 3)
        if self.__IDENTITY_INIT:
            for i in range(0, 3):
                ret[:, :, i, i] += 1
        return ret.to("cpu")

    def extract_code(self, source, target):

        return self.encoder.encode_deformation(source, target)

    #######################################
    # Pytorch Lightning Boilerplate code (training, logging, etc.)
    #######################################

    def get_gt_map(self, source, GT_V):
        GT_V = GT_V[:, :, :3]
        GT_J = source.jacobians_from_vertices(GT_V.to("cpu"))
        return GT_V, GT_J

    def predict_map(self, latent_features, source_verts, source_faces,  batch=False,
                    target_vertices=None):
        pred_J = self.predict_jacobians(latent_features)

        if not batch:

            source = SourceMesh(source_v=source_verts, source_f=source_faces, use_wks=False, random_centering=False, cpuonly=False)
            source.load(source_v=source_verts, source_f=source_faces)
            pred_V = source.vertices_from_jacobians(pred_J)

        else:
            L = []
            GT_Jac = []
            for i in range(latent_features.shape[0]):
                source = SourceMesh(source_v=source_verts, source_f=source_faces, use_wks=False, random_centering=False,
                                    cpuonly=False)
                source.load(source_v=source_verts, source_f=source_faces)
                pred_V = source.vertices_from_jacobians(pred_J)
                GT_V, GT_J = self.get_gt_map(source, target_vertices[i].unsqueeze(0))
                L.append(pred_V)
                GT_Jac.append(GT_J)
            GT_J = torch.stack(GT_Jac, dim=1).squeeze(0)
            pred_V = torch.stack(L, dim=1).squeeze(0)
            if target_vertices is not None:
                return pred_V, pred_J, GT_J

        if target_vertices is not None:
            GT_V, GT_J = self.get_gt_map(source, target_vertices)
            return pred_V, pred_J, GT_J
        else:
            return pred_V, pred_J

    def get_layer_normalization_type(self):
        if hasattr(self.args, 'layer_normalization'):
            layer_normalization = self.args.layer_normalization
        else:
            assert hasattr(self.args, 'batchnorm_decoder')
            layer_normalization = self.args.batchnorm_decoder
        return layer_normalization

