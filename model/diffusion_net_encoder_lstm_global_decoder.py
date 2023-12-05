import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d, LeakyReLU, Dropout
from wav2vec import Wav2Vec2Model
import sys
sys.path.append('/home/federico/Scrivania/ST/ScanTalk/model/diffusion-net/src')
import diffusion_net
import pdb


class DiffusionNetAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, spiral_indices, down_transform, up_transform):
        super(DiffusionNetAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = self.down_transform[-1].size(0)

        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_encoder.feature_extractor._freeze_parameters()


        # encoder
        self.encoder = diffusion_net.layers.DiffusionNet(C_in=in_channels,
                                          C_out=latent_channels,
                                          C_width=32, 
                                          N_block=2, 
                                          last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
                                          outputs_at= 'vertices',#'global_mean', 
                                          dropout=False)
        # decoder
        self.global_decoder = nn.Sequential(nn.Linear(latent_channels*2, 128),
                                                nn.ReLU(),
                                                nn.Linear(128, 64),
                                                nn.ReLU(),
                                                nn.Linear(64, 32),
                                                nn.ReLU(),
                                                nn.Linear(32, 32),
                                                nn.ReLU())
        last_layer = nn.Linear(32, 3)
        nn.init.constant_(last_layer.weight, 0)
        nn.init.constant_(last_layer.bias, 0)
        
        self.global_decoder.append(last_layer)
        
        self.audio_embedding = nn.Linear(768, latent_channels)
        self.lstm = nn.LSTM(input_size=latent_channels, hidden_size=int(latent_channels/2), num_layers=3, batch_first=True, bidirectional=True)


    def forward(self, audio, actor, vertices, mass, L, evals, evecs, gradX, gradY):
        hidden_states = self.audio_encoder(audio, frame_num=len(vertices)).last_hidden_state
        pred_sequence = actor
        audio_emb = self.audio_embedding(hidden_states)
        actor_vertices_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)
        latent, _ = self.lstm(audio_emb)
        for k in range(latent.shape[1]):
            pred_points = torch.zeros([1, 1, 3]).to('cuda')
            batches = torch.split(actor_vertices_emb, 512, dim=1)
            for batch in batches:
                decoder_input = torch.cat([batch, latent[:, k, :].expand(batch.shape)], dim=-1)
                pred_points = torch.hstack([pred_points, self.global_decoder(decoder_input)])
            pred = pred_points[:, 1:, :] + actor
            pred_sequence = torch.vstack([pred_sequence, pred])
        return pred_sequence[1:, :, :]

    def predict(self, audio, actor, mass, L, evals, evecs, gradX, gradY):
        hidden_states = self.audio_encoder(audio).last_hidden_state
        pred_sequence = actor
        audio_emb = self.audio_embedding(hidden_states)
        actor_vertices_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)
        latent, _ = self.lstm(audio_emb)
        for k in range(latent.shape[1]):
            pred_points = torch.zeros([1, 1, 3]).to('cuda')
            batches = torch.split(actor_vertices_emb, 512, dim=1)
            for batch in batches:
                decoder_input = torch.cat([batch, latent[:, k, :].expand(batch.shape)], dim=-1)
                pred_points = torch.hstack([pred_points, self.global_decoder(decoder_input)])
            pred = pred_points[:, 1:, :] + actor
            pred_sequence = torch.vstack([pred_sequence, pred])
        return pred_sequence[1:, :, :]

