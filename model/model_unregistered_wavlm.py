import torch
import torch.nn as nn
from wav2vec import Wav2Vec2Model
#from hubert.modeling_hubert import HubertModel
from MDS_viz import WavLMModel
import sys

sys.path.append('./')

import model.diffusion_net as diffusion_net


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DiffusionNetAutoencoder(nn.Module):
    def __init__(self, in_channels, latent_channels, dataset):
        super(DiffusionNetAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.dataset = dataset

        self.audio_encoder = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")  ###Wav2Vec2
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_dim = self.audio_encoder.encoder.config.hidden_size
        # encoder
        self.encoder = diffusion_net.layers.DiffusionNet(C_in=self.in_channels,
                                                         C_out=self.latent_channels,
                                                         C_width=self.latent_channels,
                                                         N_block=4,
                                                         outputs_at='vertices',  # 'global_mean',
                                                         dropout=False)
        # decoder
        self.decoder = diffusion_net.layers.DiffusionNet(C_in=self.latent_channels*2,
                                                         C_out=self.in_channels,
                                                         C_width=self.latent_channels,
                                                         N_block=4,
                                                         outputs_at='vertices',  # 'global_mean',
                                                         dropout=False)

        print("encoder parameters: ", count_parameters(self.encoder))
        print("decoder parameters: ", count_parameters(self.decoder))

        nn.init.constant_(self.decoder.last_lin.weight, 0)
        nn.init.constant_(self.decoder.last_lin.bias, 0)

        self.audio_embedding = nn.Linear(768, latent_channels)
        self.lstm = nn.LSTM(input_size=latent_channels, hidden_size=int(latent_channels / 2), num_layers=1,
                            batch_first=True, bidirectional=True)

    def forward(self, audio, actor, vertices, mass, L, evals, evecs, gradX, gradY, faces):
        hidden_states = self.audio_encoder(audio, frame_num=len(vertices), dataset=self.dataset).last_hidden_state
        pred_sequence = actor
        audio_emb = self.audio_embedding(hidden_states)
        actor_vertices_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY,
                                          faces=faces)
        latent, _ = self.lstm(audio_emb)
        for k in range(latent.shape[1]):
            pred_points = self.decoder(
                torch.cat([actor_vertices_emb, latent[:, k, :].expand(actor_vertices_emb.shape)], dim=-1), mass=mass,
                L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
            pred = pred_points + actor
            pred_sequence = torch.vstack([pred_sequence, pred])
        return pred_sequence[1:, :, :]

    def predict(self, audio, actor, mass, L, evals, evecs, gradX, gradY, faces):
        hidden_states = self.audio_encoder(audio, dataset=self.dataset).last_hidden_state
        pred_sequence = actor
        audio_emb = self.audio_embedding(hidden_states)
        actor_vertices_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY,
                                          faces=faces)
        latent, _ = self.lstm(audio_emb)
        for k in range(latent.shape[1]):
            pred_points = self.decoder(
                torch.cat([actor_vertices_emb, latent[:, k, :].expand(actor_vertices_emb.shape)], dim=-1), mass=mass,
                L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
            pred = pred_points + actor
            pred_sequence = torch.vstack([pred_sequence, pred])
        return pred_sequence[1:, :, :]