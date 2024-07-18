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
    # if ifps % ofps == 0:
    #     factor = -1 * (-ifps // ofps)
    #     print(audio_embedding_matrix.shape)
    #     if audio_embedding_matrix.shape[1] % 2 != 0:
    #         audio_embedding_matrix = audio_embedding_matrix[:, :audio_embedding_matrix.shape[1] - 1]
    #
    #     if audio_embedding_matrix.shape[1] > vertex_matrix.shape[1] * 2:
    #         audio_embedding_matrix = audio_embedding_matrix[:, :vertex_matrix.shape[1] * 2]
    #
    #     elif audio_embedding_matrix.shape[1] < vertex_matrix.shape[1] * 2:
    #         vertex_matrix = vertex_matrix[:, :audio_embedding_matrix.shape[1] // 2]
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
        self.last_layer = nn.Linear(128, 3)
        self.decoder = nn.Sequential(nn.Linear(self.latent_channels*2, 128),
                      nn.BatchNorm1d(128),  #GroupNorm(4, 128)
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
                      self.last_layer)
        nn.init.constant_(self.last_layer.weight, 0)
        nn.init.constant_(self.last_layer.bias, 0)

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
        combination_ = combination.view(combination.shape[0]*combination.shape[1], combination.shape[-1])

        pred_points = self.decoder(combination_)
        pred_points = pred_points.view(combination.shape[0], combination.shape[1], 3)
        pred = pred_points + actor
        return pred

    def predict(self, audio, actor, mass, L, evals, evecs, gradX, gradY, faces):
        hidden_states = audio
        hidden_states = self.audio_encoder(hidden_states, dataset="vocaset").last_hidden_state

        audio_emb = self.audio_embedding(hidden_states)
        actor_vertices_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY,
                                          faces=faces)
        latent, _ = self.lstm(audio_emb)

        combination = torch.cat(
            [actor_vertices_emb.expand((1, latent.shape[1], actor_vertices_emb.shape[1], actor_vertices_emb.shape[2])),
             latent.unsqueeze(2).expand(1, latent.shape[1], actor_vertices_emb.shape[1], latent.shape[2])], dim=-1)

        combination = combination.squeeze(0)
        combination_ = combination.view(combination.shape[0] * combination.shape[1], combination.shape[-1])

        pred_points = self.decoder(combination_)
        pred_points = pred_points.view(combination.shape[0], combination.shape[1], 3)
        pred = pred_points + actor
        return pred

    def get_latent_features(self, audio, actor, mass, L, evals, evecs, gradX, gradY, faces):
        hidden_states = audio
        hidden_states = self.audio_encoder(hidden_states).last_hidden_state

        # if hidden_states.shape[1] % 2 != 0:
        #     hidden_states = hidden_states[:, :hidden_states.shape[1] - 1]
        # hidden_states = torch.reshape(hidden_states, (1, hidden_states.shape[1] // 2, hidden_states.shape[2] * 2))

        audio_emb = self.audio_embedding(hidden_states)
        actor_vertices_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY,
                                          faces=faces)
        latent, _ = self.lstm(audio_emb)

        return actor_vertices_emb, latent
