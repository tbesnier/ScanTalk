import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d, LeakyReLU, Dropout
#from wav2vec import Wav2Vec2Model
from hubert.modeling_hubert import HubertModel
import sys
sys.path.append('/home/federico/Scrivania/ST/ScanTalk/model/diffusion-net_/src')
import diffusion_net
import pdb
import math

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Alignment Bias
def enc_dec_mask(device, T, S):
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, i] = 0
    return (mask==1).to(device=device)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 600):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def casual_mask(n_head, max_seq_len):
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0)
    mask = mask.repeat(n_head, 1,1)
    return mask

class DiffusionNetAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, device):
        super(DiffusionNetAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.device = device

        #self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.audio_encoder.feature_extractor._freeze_parameters()


        # encoder
        self.encoder = diffusion_net.layers.DiffusionNet(C_in=self.in_channels,
                                          C_out=self.latent_channels,
                                          C_width=self.latent_channels, 
                                          N_block=4, 
                                          outputs_at='vertices', 
                                          dropout=False)
        # decoder
        self.decoder = diffusion_net.layers.DiffusionNet(C_in=self.latent_channels*2,
                                          C_out=self.out_channels,
                                          C_width=self.latent_channels, 
                                          N_block=4, 
                                          outputs_at='vertices', 
                                          dropout=False)
        
        print("encoder parameters: ", count_parameters(self.encoder))
        print("decoder parameters: ", count_parameters(self.decoder))


        nn.init.constant_(self.decoder.last_lin.weight, 0)
        nn.init.constant_(self.decoder.last_lin.bias, 0)
        
        self.audio_embedding = nn.Linear(768, latent_channels)

        self.PE = PositionalEncoding(self.latent_channels)

        self.biased_mask = casual_mask(n_head=4, max_seq_len=600)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.latent_channels, nhead=4, dim_feedforward=2*self.latent_channels, batch_first=True)        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)


    def forward(self, audio, actor, vertices, mass, L, evals, evecs, gradX, gradY, faces, dataset):
        hidden_states = self.audio_encoder(audio, dataset, frame_num=len(vertices)).last_hidden_state
        pred_sequence = actor
        audio_emb = self.audio_embedding(hidden_states)
        actor_vertices_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        actor_vertices_emb_mean = torch.mean(actor_vertices_emb, dim=1)
        for k in range(audio_emb.shape[1]):
            if k == 0:
                vertice_emb = actor_vertices_emb_mean.unsqueeze(1) # (1,1,feature_dim)
                vertice_input = self.PE(vertice_emb)
            else:
                vertice_input = self.PE(vertice_emb)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
                
            memory_mask = enc_dec_mask(self.device, vertice_input.shape[1], audio_emb.shape[1])
            vertice_out_emb = self.transformer_decoder(vertice_input, audio_emb, tgt_mask=tgt_mask, memory_mask=memory_mask)
                
            pred = self.decoder(torch.cat([actor_vertices_emb, vertice_out_emb[:, -1, :].expand(actor_vertices_emb.shape)], dim=-1), mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
            pred = pred + actor
            vertice_emb = torch.cat((vertice_emb, vertice_out_emb[:, -1, :].unsqueeze(1)), 1)
            pred_sequence = torch.vstack([pred_sequence, pred])
            
        return pred_sequence[1:, :, :]
    
    def predict(self, audio, actor, mass, L, evals, evecs, gradX, gradY, faces, dataset, hks=None):
        hidden_states = self.audio_encoder(audio, dataset).last_hidden_state
        pred_sequence = actor
        audio_emb = self.audio_embedding(hidden_states)
        actor_vertices_emb = self.encoder(actor, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        actor_vertices_emb_mean = torch.mean(actor_vertices_emb, dim=1)
        for k in range(audio_emb.shape[1]):
            if k == 0:
                vertice_emb = actor_vertices_emb_mean.unsqueeze(1) # (1,1,feature_dim)
                vertice_input = self.PE(vertice_emb)
            else:
                vertice_input = self.PE(vertice_emb)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
                
            memory_mask = enc_dec_mask(self.device, vertice_input.shape[1], audio_emb.shape[1])
            vertice_out_emb = self.transformer_decoder(vertice_input, audio_emb, tgt_mask=tgt_mask, memory_mask=memory_mask)
                
            pred = self.decoder(torch.cat([actor_vertices_emb, vertice_out_emb[:, -1, :].expand(actor_vertices_emb.shape)], dim=-1), mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
            pred = pred + actor
            vertice_emb = torch.cat((vertice_emb, vertice_out_emb[:, -1, :].unsqueeze(1)), 1)
            pred_sequence = torch.vstack([pred_sequence, pred])
            
        return pred_sequence[1:, :, :]

