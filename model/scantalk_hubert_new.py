import torch
import torch.nn as nn
import torch.nn.functional as F
from hubert.modeling_hubert import HubertModel

import sys

sys.path.append('./')

import model.diffusion_net as diffusion_net
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DiffusionNetAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, device):
        super(DiffusionNetAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.device = device

        self.audio_encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.audio_encoder.feature_extractor._freeze_parameters()

        # encoder
        self.encoder = diffusion_net.layers.DiffusionNet(C_in=self.in_channels,
                                                         C_out=self.latent_channels,
                                                         C_width=self.latent_channels,
                                                         N_block=4,
                                                         outputs_at='vertices',
                                                         normalization="None",
                                                         dropout=False)
        # decoder
        self.decoder = diffusion_net.layers.DiffusionNet(C_in=self.latent_channels * 2,
                                                         C_out=self.out_channels,
                                                         C_width=self.latent_channels,
                                                         N_block=4,
                                                         outputs_at='vertices',
                                                         normalization="None",
                                                         dropout=False)

        nn.init.constant_(self.decoder.last_lin.weight, 0)
        nn.init.constant_(self.decoder.last_lin.bias, 0)

        self.audio_embedding = nn.Linear(768, latent_channels)
        self.lstm = nn.LSTM(input_size=latent_channels, hidden_size=int(latent_channels / 2), num_layers=3,
                            batch_first=True, bidirectional=True)

    def forward(self, audio, actor, vertices, mass, L, evals, evecs, gradX, gradY, faces, dataset):
        hidden_states = audio
        hidden_states = self.audio_encoder(hidden_states, dataset, frame_num=len(vertices)).last_hidden_state

        audio_emb = self.audio_embedding(hidden_states)
        actor_vertices_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY,
                                          faces=faces)
        latent, _ = self.lstm(audio_emb)

        combination = torch.cat(
            [actor_vertices_emb.expand((1, latent.shape[1], actor_vertices_emb.shape[1], actor_vertices_emb.shape[2])),
             latent.unsqueeze(2).expand(1, latent.shape[1], actor_vertices_emb.shape[1], latent.shape[2])], dim=-1)

        combination = combination.squeeze(0)
        mass = mass.expand(latent.shape[1], mass.shape[1])
        L = L.to_dense().expand(latent.shape[1], L.shape[1], L.shape[2])
        evals = evals.expand(latent.shape[1], evals.shape[1])
        evecs = evecs.expand(latent.shape[1], evecs.shape[1], evecs.shape[2])
        gradX = gradX.to_dense().expand(latent.shape[1], gradX.shape[1], gradX.shape[2])
        gradY = gradY.to_dense().expand(latent.shape[1], gradY.shape[1], gradY.shape[2])
        faces = faces.expand(latent.shape[1], faces.shape[1], faces.shape[2])

        pred_points = self.decoder(combination, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY,
                                   faces=faces)
        pred = pred_points + actor
        return pred

    def predict(self, audio, actor, mass, L, evals, evecs, gradX, gradY, faces, dataset='vocaset'):
        hidden_states = audio
        hidden_states = self.audio_encoder(hidden_states, dataset).last_hidden_state

        pred_sequence = actor
        audio_emb = self.audio_embedding(hidden_states)
        actor_vertices_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY,
                                          faces=faces)
        latent, _ = self.lstm(audio_emb)

        for k in range(latent.shape[1]):
            feats = torch.cat([actor_vertices_emb, latent[:, k, :].expand(
                    (actor_vertices_emb.shape[0], actor_vertices_emb.shape[1], latent.shape[-1]))], dim=-1)
            pred_points = self.decoder(feats, mass=mass,L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
            pred = pred_points + actor
            pred_sequence = torch.vstack([pred_sequence, pred])
        return pred_sequence[1:, :, :]

    def get_latent_features(self, audio, actor, mass, L, evals, evecs, gradX, gradY, faces, dataset='vocaset'):
        hidden_states = audio
        hidden_states = self.audio_encoder(hidden_states, dataset).last_hidden_state

        audio_emb = self.audio_embedding(hidden_states)
        actor_vertices_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY,
                                          faces=faces)
        latent, _ = self.lstm(audio_emb)

        for k in range(latent.shape[1]):
            feats = torch.cat([actor_vertices_emb, latent[:, k, :].expand(
                    (actor_vertices_emb.shape[0], actor_vertices_emb.shape[1], latent.shape[-1]))], dim=-1)
            feats_mean = torch.sum(feats * mass.unsqueeze(-1), dim=-2) / torch.sum(mass, dim=-1, keepdim=True)
            if k==0:
                global_features_temp = feats_mean
            else:
                global_features_temp = torch.cat([global_features_temp, feats_mean])

        return global_features_temp
