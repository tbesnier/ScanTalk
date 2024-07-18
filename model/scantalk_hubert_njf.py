import torch
import torch.nn as nn
import torch.nn.functional as F
from hubert.modeling_hubert import HubertModel

import sys

sys.path.append('./')

import model.diffusion_net as diffusion_net
#from model.njf.net import njf_decoder

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DiffusionNetAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, device, args):
        super(DiffusionNetAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.device = device
        self.args = args

        self.audio_encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.audio_encoder.feature_extractor._freeze_parameters()

        # encoder
        self.encoder = diffusion_net.layers.DiffusionNet(C_in=self.in_channels,
                                                         C_out=self.latent_channels,
                                                         C_width=self.latent_channels,
                                                         N_block=4,
                                                         outputs_at='faces',
                                                         normalization="None",
                                                         dropout=False)
        # decoder
        self.decoder = njf_decoder(latent_features_shape=self.latent_channels*2, args=args)

        #nn.init.constant_(self.decoder.last_lin.weight, 0)
        #nn.init.constant_(self.decoder.last_lin.bias, 0)

        self.audio_embedding = nn.Linear(768, latent_channels)
        self.lstm = nn.LSTM(input_size=latent_channels, hidden_size=int(latent_channels / 2), num_layers=3,
                            batch_first=True, bidirectional=True)

    def forward(self, audio, actor, vertices, mass, L, evals, evecs, gradX, gradY, faces, dataset):
        hidden_states = audio
        hidden_states = self.audio_encoder(hidden_states, dataset, frame_num=len(vertices)).last_hidden_state

        pred_sequence = actor
        audio_emb = self.audio_embedding(hidden_states)
        actor_vertices_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY,
                                          faces=faces)
        latent, _ = self.lstm(audio_emb)

        for k in range(latent.shape[1]):
            cat_latent = torch.cat([actor_vertices_emb, latent[:, k, :].expand(
                (actor_vertices_emb.shape[0], actor_vertices_emb.shape[1], latent.shape[-1]))], dim=-1)

            delta, pred_jac = self.decoder.predict_map(cat_latent, source_verts=actor,
                                                               source_faces=faces,
                                                               batch=False, target_vertices=None)
            delta, pred_jac = delta.to(self.device), pred_jac.to(self.device)
            pred = delta + actor
            pred_sequence = torch.vstack([pred_sequence, pred])
        return pred_sequence[1:, :, :]


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
