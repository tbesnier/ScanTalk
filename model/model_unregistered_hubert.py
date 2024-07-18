import torch
import torch.nn as nn
import torch.nn.functional as F
#from wav2vec import Wav2Vec2Model
from hubert.modeling_hubert import HubertModel
import sys

sys.path.append('./')

import model.diffusion_net as diffusion_net


def inputRepresentationAdjustment(audio_embedding_matrix, vertex_matrix, ifps, ofps):

    if ifps % ofps == 0:
        factor = -1 * (-ifps // ofps)
        if audio_embedding_matrix.shape[1] % 2 != 0:
            audio_embedding_matrix = audio_embedding_matrix[:, :audio_embedding_matrix.shape[1] - 1]

        if audio_embedding_matrix.shape[1] > vertex_matrix.shape[1] * 2:
            audio_embedding_matrix = audio_embedding_matrix[:, :vertex_matrix.shape[1] * 2]

        elif audio_embedding_matrix.shape[1] < vertex_matrix.shape[1] * 2:
            vertex_matrix = vertex_matrix[:, :audio_embedding_matrix.shape[1] // 2]
    else:
        factor = -1 * (-ifps // ofps)
        audio_embedding_seq_len = vertex_matrix.shape[1] * factor
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(audio_embedding_matrix, size=audio_embedding_seq_len, align_corners=True, mode='linear')
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)

    frame_num = vertex_matrix.shape[1]
    audio_embedding_matrix = torch.reshape(audio_embedding_matrix, (1, audio_embedding_matrix.shape[1] // factor, audio_embedding_matrix.shape[2] * factor))

    return audio_embedding_matrix, vertex_matrix, frame_num

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
    def __init__(self, in_channels, latent_channels, dataset, audio_latent):
        super(DiffusionNetAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.audio_latent = audio_latent
        self.dataset = dataset

        #self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")  ###Wav2Vec2
        #self.audio_encoder = HubertModel.from_pretrained("facebook/hubert-xlarge-ls960-ft")
        self.audio_encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.audio_encoder.feature_extractor._freeze_parameters()

        # frozen_layers = [0, 1]
        #
        # for name, param in self.audio_encoder.named_parameters():
        #     if name.startswith("feature_projection"):
        #         param.requires_grad = False
        #     if name.startswith("encoder.layers"):
        #         layer = int(name.split(".")[2])
        #         if layer in frozen_layers:
        #             param.requires_grad = False

        # encoder
        self.encoder = diffusion_net.layers.DiffusionNet(C_in=self.in_channels,
                                                         C_out=self.latent_channels,
                                                         C_width=self.audio_latent,
                                                         N_block=4,
                                                         outputs_at='vertices',  # 'global_mean',
                                                         dropout=False)
        # decoder
        self.decoder = diffusion_net.layers.DiffusionNet(C_in=self.latent_channels + audio_latent,
                                                         C_out=self.in_channels,
                                                         C_width=self.audio_latent,
                                                         N_block=4,
                                                         outputs_at='vertices',  # 'global_mean',
                                                         dropout=False)

        print("encoder parameters: ", count_parameters(self.encoder))
        print("decoder parameters: ", count_parameters(self.decoder))

        nn.init.constant_(self.decoder.last_lin.weight, 0)
        nn.init.constant_(self.decoder.last_lin.bias, 0)

        self.audio_embedding = nn.Linear(1536, audio_latent)
        self.lstm = nn.LSTM(input_size=audio_latent, hidden_size=int(audio_latent / 2), num_layers=1,
                            batch_first=True, bidirectional=True)

        self.i_fps = 50  # audio fps (input to the network)
        self.o_fps = 30  # 4D Scan fps (output or target)


    def forward(self, audio, actor, vertices, mass, L, evals, evecs, gradX, gradY, faces):
        hidden_states = audio
        hidden_states = self.audio_encoder(hidden_states).last_hidden_state
        x = vertices.unsqueeze(0)
        x = x.reshape((1, vertices.shape[0], vertices.shape[1]*3))
        hidden_states, vertice, frame_num = adjust_input_representation(hidden_states, x, self.i_fps, self.o_fps)
        hidden_states = hidden_states[:, :frame_num]

        pred_sequence = actor
        audio_emb = self.audio_embedding(hidden_states)
        actor_vertices_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY,
                                          faces=faces)
        latent, _ = self.lstm(audio_emb)

        for k in range(latent.shape[1]):
            cat_latent = torch.cat([actor_vertices_emb, latent[:, k, :].expand((actor_vertices_emb.shape[0], actor_vertices_emb.shape[1], latent.shape[-1]))], dim=-1)
            pred_points = self.decoder(cat_latent, mass=mass,
                L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
            pred = pred_points + actor
            pred_sequence = torch.vstack([pred_sequence, pred])
        return pred_sequence[1:, :, :]

    def predict(self, audio, actor, mass, L, evals, evecs, gradX, gradY, faces):
        hidden_states = audio
        hidden_states = self.audio_encoder(hidden_states).last_hidden_state

        if hidden_states.shape[1] % 2 != 0:
            hidden_states = hidden_states[:, :hidden_states.shape[1] - 1]
        hidden_states = torch.reshape(hidden_states, (1, hidden_states.shape[1] // 2, hidden_states.shape[2] * 2))

        pred_sequence = actor
        audio_emb = self.audio_embedding(hidden_states)
        actor_vertices_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY,
                                          faces=faces)
        latent, _ = self.lstm(audio_emb)

        for k in range(latent.shape[1]):
            pred_points = self.decoder(
                torch.cat([actor_vertices_emb, latent[:, k, :].expand((actor_vertices_emb.shape[0], actor_vertices_emb.shape[1], latent.shape[-1]))], dim=-1), mass=mass,
                L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
            pred = pred_points + actor
            pred_sequence = torch.vstack([pred_sequence, pred])
        return pred_sequence[1:, :, :]


    def get_latent_features(self, audio, actor, mass, L, evals, evecs, gradX, gradY, faces):
        hidden_states = audio
        hidden_states = self.audio_encoder(hidden_states).last_hidden_state

        if hidden_states.shape[1] % 2 != 0:
            hidden_states = hidden_states[:, :hidden_states.shape[1] - 1]
        hidden_states = torch.reshape(hidden_states, (1, hidden_states.shape[1] // 2, hidden_states.shape[2] * 2))

        audio_emb = self.audio_embedding(hidden_states)
        actor_vertices_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY,
                                          faces=faces)
        latent, _ = self.lstm(audio_emb)

        return actor_vertices_emb, latent
