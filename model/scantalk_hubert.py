import torch
import torch.nn as nn
import torch.nn.functional as F
from hubert.modeling_hubert import HubertModel

import sys

sys.path.append('./')

import model.diffusion_net as diffusion_net

def adjust_input_representation(audio_embedding_matrix, vertex_matrix, ifps, ofps):
    """
    Brings audio embeddings and visual frames to the same frame rate.

    Args:
        audio_embedding_matrix: The audio embeddings extracted by the audio encoder
        vertex_matrix: The animation sequence represented as a series of vertex positions (or blendshape controls)
        ifps: The input frame rate (it is 50 for the HuBERT encoder)
        ofps: The output frame rate
    """
    if ifps > ofps:
        factor = -1 * (-ifps // ofps)
        audio_embedding_seq_len = vertex_matrix.shape[1] * factor
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(audio_embedding_matrix, size=audio_embedding_seq_len, align_corners=True, mode='linear')
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
    else:
        factor = 1
        audio_embedding_seq_len = vertex_matrix.shape[1] * factor
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(audio_embedding_matrix, size=audio_embedding_seq_len, align_corners=True, mode='linear')
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)

    frame_num = vertex_matrix.shape[1]
    audio_embedding_matrix = torch.reshape(audio_embedding_matrix, (1, audio_embedding_matrix.shape[1] // factor, audio_embedding_matrix.shape[2] * factor))
    return audio_embedding_matrix, vertex_matrix, frame_num

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DiffusionNetAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, device):
        super(DiffusionNetAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.device = device

        # self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
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

        self.i_fps = 60  # audio fps (input to the network)
        self.o_fps = 30  # 4D Scan fps (output or target)


    def forward(self, audio, actor, vertices, mass, L, evals, evecs, gradX, gradY, faces):
        hidden_states = audio
        hidden_states = self.audio_encoder(hidden_states).last_hidden_state
        x = vertices.unsqueeze(0)
        x = x.reshape((1, vertices.shape[0], vertices.shape[1] * 3))
        hidden_states, vertice, frame_num = adjust_input_representation(hidden_states, x, self.i_fps, self.o_fps)
        hidden_states = hidden_states[:, :frame_num]

        pred_sequence = actor
        audio_emb = self.audio_embedding(hidden_states)
        actor_vertices_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY,
                                          faces=faces)
        latent, _ = self.lstm(audio_emb)

        for k in range(latent.shape[1]):
            cat_latent = torch.cat([actor_vertices_emb, latent[:, k, :].expand(
                (actor_vertices_emb.shape[0], actor_vertices_emb.shape[1], latent.shape[-1]))], dim=-1)
            pred_points = self.decoder(cat_latent, mass=mass,
                                       L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
            pred = pred_points + actor
            pred_sequence = torch.vstack([pred_sequence, pred])
        return pred_sequence[1:, :, :]

    def predict(self, audio, actor, mass, L, evals, evecs, gradX, gradY, faces):
        hidden_states = audio
        hidden_states = self.audio_encoder(hidden_states).last_hidden_state

        # if hidden_states.shape[1] % 2 != 0:
        #      hidden_states = hidden_states[:, :hidden_states.shape[1] - 1]
        # hidden_states = torch.reshape(hidden_states, (1, hidden_states.shape[1] // 2, hidden_states.shape[2] * 2))

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

    def get_latent_features(self, audio, actor, mass, L, evals, evecs, gradX, gradY, faces):
        hidden_states = audio
        hidden_states = self.audio_encoder(hidden_states).last_hidden_state

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
