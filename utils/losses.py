import torch
import torch.nn as nn
import numpy as np
from pytorch3d.loss import(
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

from pytorch3d.structures import Meshes

import lddmm_utils

class Masked_Loss(nn.Module):
    def __init__(self, args):
        super(Masked_Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.weights = np.load('./template/template/Normalized_d_weights.npy', allow_pickle=True)
        self.weights = torch.from_numpy(self.weights[:-1]).float().to(args.device)

    def forward(self, predictions, target):

        rec_loss = torch.mean(self.mse(predictions, target))

        landmarks_loss = (self.mse(predictions, target).mean(axis=2) * self.weights).mean()

        prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_shift = target[:, 1:, :] - target[:, :-1, :]

        vel_loss = torch.mean((self.mse(prediction_shift, target_shift)))

        return rec_loss + 10 * landmarks_loss + 10 * vel_loss

class Chamfer_Loss(nn.Module):
    def __init__(self, args):
        super(Chamfer_Loss, self).__init__()
        self.device = args.device
        #self.faces = faces.to(self.device)
        self.w_edge_loss = 0.5
        self.w_laplacian_loss = 0.05
        self.w_normal_loss = 0.01
    def forward(self, predictions, targets, pred_faces):

        loss_chamfer, _ = chamfer_distance(predictions, targets)

        M = Meshes(verts=predictions,       ## if common faces (same topology), use faces.repeat(predictions.shape[0], 1, 1)
                   faces=pred_faces)

        if self.w_edge_loss > 0:
            self.w_edge_loss = self.w_edge_loss * mesh_edge_loss(M)
        if self.w_laplacian_loss > 0:
            self.w_laplacian_loss = self.w_laplacian_loss * mesh_laplacian_smoothing(M, method="cot")  # mesh laplacian smoothing
        if self.w_normal_loss > 0:
            self.w_normal_loss = self.w_normal_loss * mesh_normal_consistency(M)

        prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_shift = targets[:, 1:, :] - targets[:, :-1, :]

        vel_loss = chamfer_distance(prediction_shift, target_shift)[0]

        return loss_chamfer + self.w_laplacian_loss + self.w_normal_loss + self.w_edge_loss + 10*vel_loss


class Varifold_loss(nn.Module):
    def __init__(self, args):
        super(Varifold_loss, self).__init__()
        self.device = args.device
        self.torchdtype = torch.float
        self.sig = [0.08, 0.02]
        #self.sig_n = torch.tensor([0.5], dtype=self.torchdtype, device=self.device)
        for i, sigma in enumerate(self.sig):
            self.sig[i] = torch.tensor([sigma], dtype=self.torchdtype, device=self.device)
    def forward(self, predictions, targets, pred_faces, target_faces):
        L = []
        for i in range(predictions.shape[0]):
            Li = torch.Tensor([0.]).to(self.device)
            V1, F1 = predictions[i], pred_faces[i]
            V2, F2 = targets[i], target_faces[i]

            for sigma in self.sig:
                Li += (sigma / self.sig[0]) ** 2 * lddmm_utils.lossVarifoldSurf(F1, V2, F2,
                                                                lddmm_utils.GaussSquaredKernel_varifold_unoriented(
                                                                sigma=sigma))(V1)

            L.append(Li)

        return torch.stack(L).mean()