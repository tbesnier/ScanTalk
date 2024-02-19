import os
import shape_data
import pickle
import trimesh
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import euclidean_distances
import argparse
from tqdm import tqdm
import librosa
import random
from transformers import Wav2Vec2Processor
from wav2vec import Wav2Vec2Model
from psbody.mesh import Mesh
from utils import utils, mesh_sampling
from multidataset_data_loader_diffusion_net import get_dataloaders
import scipy
import sys
import lddmm_utils
from model.diffusion_net_multidataset_encoder_decoder import DiffusionNetAutoencoder
sys.path.append('/home/federico/Scrivania/ST/ScanTalk/model/diffusion-net_/src')
import diffusion_net

class UnMasked_Loss(nn.Module):
    def __init__(self):
        super(UnMasked_Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, predictions, target):
        
        rec_loss = torch.mean(self.mse(predictions, target))
        
        prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_shift = target[:, 1:, :] - target[:, :-1, :]

        vel_loss = torch.mean((self.mse(prediction_shift, target_shift)))

        return rec_loss + 10 * vel_loss
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args):

    device = args.device
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    

    reference_voca = trimesh.load(args.reference_mesh_file_voca, process=False)
    template_tri_voca = reference_voca.faces
    template_vertices_voca = torch.tensor(reference_voca.vertices).to(args.device).float()

    temp_biwi = trimesh.load('/home/federico/Scrivania/ST/Data/Biwi_6/templates/M6.obj', process=False)
    template_vertices_biwi = torch.tensor(temp_biwi.vertices).to(args.device).float()
    template_tri_biwi = temp_biwi.faces
    
    temp_multiface = trimesh.load('/home/federico/Scrivania/ST/Data/Multiface/templates/20190828.ply', process=False)
    template_vertices_multiface = torch.tensor(temp_multiface.vertices).to(args.device).float()
    template_tri_multiface = temp_multiface.faces


    d2d = DiffusionNetAutoencoder(args.in_channels, args.latent_channels).to(args.device)

    print("model parameters: ", count_parameters(d2d))

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    dataset = get_dataloaders(args)
            

    lip_mask_voca = scipy.io.loadmat('/home/federico/Scrivania/ST/ScanTalk/FLAME_lips_idx.mat')
    lip_mask_voca = lip_mask_voca['lips_idx'] - 1 
    lip_mask_voca = np.reshape(np.array(lip_mask_voca, dtype=np.int64), (lip_mask_voca.shape[0]))
        
    
    lip_mask_biwi = np.load('/home/federico/Scrivania/ST/Data/Biwi_6/mouth_indices.npy')
    
    lip_mask_multiface = np.load('/home/federico/Scrivania/ST/Data/Multiface/mouth_indices.npy')
        

    criterion = nn.MSELoss()
    criterion_val = nn.MSELoss()
    
    optim = torch.optim.Adam(d2d.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        d2d.train()
        tloss = 0
        
        pbar_talk = tqdm(enumerate(dataset["train"]), total=len(dataset["train"]))
        for b, sample in pbar_talk:
            audio = sample[0].to(device)
            vertices = sample[1].to(device).squeeze(0)
            template = sample[2].to(device)
            mass = sample[3].to(device)
            L = sample[4].to(device)
            evals = sample[5].to(device)
            evecs = sample[6].to(device)
            gradX = sample[7].to(device)
            gradY = sample[8].to(device)
            faces = sample[10].to(device)
            dataset_type = sample[11][0]
            vertices_pred = d2d.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY, faces, dataset_type)
            optim.zero_grad()

            loss = criterion(vertices, vertices_pred) 
            torch.nn.utils.clip_grad_norm_(d2d.parameters(), 10.0)
            loss.backward()
            optim.step()
            tloss += loss.item()
            pbar_talk.set_description(
                "(Epoch {}) TRAIN LOSS:{:.10f}".format((epoch + 1), tloss/(b+1)))
        
        if epoch % 10 == 0:
            d2d.eval()
            with torch.no_grad():
                t_test_loss = 0
                pbar_talk = tqdm(enumerate(dataset["valid"]), total=len(dataset["valid"]))
                for b, sample in pbar_talk:
                    audio = sample[0].to(device)
                    vertices = sample[1].to(device).squeeze(0)
                    template = sample[2].to(device)
                    mass = sample[3].to(device)
                    L = sample[4].to(device)
                    evals = sample[5].to(device)
                    evecs = sample[6].to(device)
                    gradX = sample[7].to(device)
                    gradY = sample[8].to(device)
                    faces = sample[10].to(device)
                    dataset_type = sample[11][0]
                    #print(sample[9])
                    vertices_pred = d2d.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY, faces, dataset_type)
                    loss = criterion_val(vertices, vertices_pred)
                    t_test_loss += loss.item()
                    pbar_talk.set_description(
                                    "(Epoch {}) VAL LOSS:{:.10f}".format((epoch + 1), (t_test_loss)/(b+1)))
                
                #Sample from external audio and face Voca
                #Process the audio
                speech_array, sampling_rate = librosa.load(args.sample_audio, sr=16000)
                audio_feature = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)
                audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
                audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)
                #audio_feature = audio_encoder(audio_feature, frame_num=len(vertices)).last_hidden_state.to(device=args.device)
                
                #Compute Operators for DiffuserNet
                frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(template_vertices_voca.to('cpu'), faces=torch.tensor(template_tri_voca), k_eig=128)
                mass = torch.FloatTensor(np.array(mass)).float().to(device).unsqueeze(0)
                evals = torch.FloatTensor(np.array(evals)).to(device).unsqueeze(0)
                evecs = torch.FloatTensor(np.array(evecs)).to(device).unsqueeze(0)
                L = L.float().to(device).unsqueeze(0)
                gradX = gradX.float().to(device).unsqueeze(0)
                gradY = gradY.float().to(device).unsqueeze(0)
                faces = torch.tensor(template_tri_voca).to(device).float().unsqueeze(0)
                
                gen_seq = d2d.predict(audio_feature, template_vertices_voca.to(device).unsqueeze(0), mass, L, evals, evecs, gradX, gradY, faces, 'vocaset')
                
                gen_seq = gen_seq.cpu().detach().numpy()
                
                os.makedirs('/home/federico/Scrivania/ST/Data/saves/Meshes_DiffuserNet_Encoder_Decoder_Faces_MSE_All_Multidataset_VOCA/' + str(epoch), exist_ok=True)
                for m in range(len(gen_seq)):
                    mesh = trimesh.Trimesh(gen_seq[m], template_tri_voca)
                    mesh.export('/home/federico/Scrivania/ST/Data/saves/Meshes_DiffuserNet_Encoder_Decoder_Faces_MSE_All_Multidataset_VOCA/' + str(epoch) + '/frame_' + str(m).zfill(3) + '.ply')
                
                #Sample from external audio and face BIWI
                #Process the audio
                speech_array, sampling_rate = librosa.load(args.sample_audio, sr=16000)
                audio_feature = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)
                audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
                audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)
                #audio_feature = audio_encoder(audio_feature, frame_num=len(vertices)).last_hidden_state.to(device=args.device)
                
                #Compute Operators for DiffuserNet
                frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(template_vertices_biwi.to('cpu'), faces=torch.tensor(template_tri_biwi), k_eig=128)
                mass = torch.FloatTensor(np.array(mass)).float().to(device).unsqueeze(0)
                evals = torch.FloatTensor(np.array(evals)).to(device).unsqueeze(0)
                evecs = torch.FloatTensor(np.array(evecs)).to(device).unsqueeze(0)
                L = L.float().to(device).unsqueeze(0)
                gradX = gradX.float().to(device).unsqueeze(0)
                gradY = gradY.float().to(device).unsqueeze(0)
                faces = torch.tensor(template_tri_biwi).to(device).float().unsqueeze(0)
                
                gen_seq = d2d.predict(audio_feature, template_vertices_biwi.to(device).unsqueeze(0), mass, L, evals, evecs, gradX, gradY, faces, 'BIWI')
                
                gen_seq = gen_seq.cpu().detach().numpy()
                
                os.makedirs('/home/federico/Scrivania/ST/Data/saves/Meshes_DiffuserNet_Encoder_Decoder_Faces_MSE_All_Multidataset_BIWI/' + str(epoch), exist_ok=True)
                for m in range(len(gen_seq)):
                    mesh = trimesh.Trimesh(gen_seq[m], template_tri_biwi)
                    mesh.export('/home/federico/Scrivania/ST/Data/saves/Meshes_DiffuserNet_Encoder_Decoder_Faces_MSE_All_Multidataset_BIWI/' + str(epoch) + '/frame_' + str(m).zfill(3) + '.ply')
                
                #Sample from external audio and face Multiface
                #Process the audio
                speech_array, sampling_rate = librosa.load(args.sample_audio, sr=16000)
                audio_feature = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)
                audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
                audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)
                #audio_feature = audio_encoder(audio_feature, frame_num=len(vertices)).last_hidden_state.to(device=args.device)
                
                #Compute Operators for DiffuserNet
                frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(template_vertices_multiface.to('cpu'), faces=torch.tensor(template_tri_multiface), k_eig=128)
                mass = torch.FloatTensor(np.array(mass)).float().to(device).unsqueeze(0)
                evals = torch.FloatTensor(np.array(evals)).to(device).unsqueeze(0)
                evecs = torch.FloatTensor(np.array(evecs)).to(device).unsqueeze(0)
                L = L.float().to(device).unsqueeze(0)
                gradX = gradX.float().to(device).unsqueeze(0)
                gradY = gradY.float().to(device).unsqueeze(0)
                faces = torch.tensor(template_tri_multiface).to(device).float().unsqueeze(0)
                
                gen_seq = d2d.predict(audio_feature, template_vertices_multiface.to(device).unsqueeze(0), mass, L, evals, evecs, gradX, gradY, faces, 'multiface')
                
                gen_seq = gen_seq.cpu().detach().numpy()
                
                os.makedirs('/home/federico/Scrivania/ST/Data/saves/Meshes_DiffuserNet_Encoder_Decoder_Faces_MSE_All_Multidataset_Multiface/' + str(epoch), exist_ok=True)
                for m in range(len(gen_seq)):
                    mesh = trimesh.Trimesh(gen_seq[m], template_tri_biwi)
                    mesh.export('/home/federico/Scrivania/ST/Data/saves/Meshes_DiffuserNet_Encoder_Decoder_Faces_MSE_All_Multidataset_Multiface/' + str(epoch) + '/frame_' + str(m).zfill(3) + '.ply')
                error_lve_voca = 0
                count_voca = 0
                error_lve_biwi = 0
                count_biwi = 0
                error_lve_multiface = 0
                count_multiface = 0
                pbar_talk = tqdm(enumerate(dataset["test"]), total=len(dataset["test"]))
                for b, sample in pbar_talk:
                    audio = sample[0].to(device)
                    vertices = sample[1].to(device).squeeze(0)
                    template = sample[2].to(device)
                    mass = sample[3].to(device)
                    L = sample[4].to(device)
                    evals = sample[5].to(device)
                    evecs = sample[6].to(device)
                    gradX = sample[7].to(device)
                    gradY = sample[8].to(device)
                    faces = sample[10].to(device)
                    dataset_type = sample[11][0]
                    vertices_pred = d2d.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY, faces, dataset_type)
                    vertices = vertices.detach().cpu().numpy()
                    vertices_pred = vertices_pred.detach().cpu().numpy()
                    if dataset_type == 'vocaset':
                        for k in range(vertices_pred.shape[0]):
                            error_lve_voca += ((vertices_pred[k] - vertices[k]) ** 2)[lip_mask_voca].max()
                            count_voca += 1

                    if dataset_type == 'BIWI':
                        for k in range(vertices_pred.shape[0]):
                            error_lve_biwi += ((vertices_pred[k] - vertices[k]) ** 2)[lip_mask_biwi].max()
                            count_biwi += 1
                            
                    if dataset_type == 'multiface':
                        for k in range(vertices_pred.shape[0]):
                            error_lve_multiface += ((vertices_pred[k] - vertices[k]) ** 2)[lip_mask_multiface].max()
                            count_multiface += 1
                            
                print("LVE_VOCA:{:.8f}".format((error_lve_voca)/(count_voca)))
                print("LVE_BIWI:{:.8f}".format((error_lve_biwi)/(count_biwi)))
                print("LVE_Multiface:{:.8f}".format((error_lve_multiface)/(count_multiface)))
                
        torch.save({'epoch': epoch,
            'autoencoder_state_dict': d2d.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            }, os.path.join(args.result_dir, 'd2d_ScanTalk_DiffuserNet_Encoder_Decoder_Faces_MSE_Multidataset_VOCA_and_BIWI_and_Multiface.pth.tar'))
    
        
def main():
    parser = argparse.ArgumentParser(description='Diffusion Net Multidataset: Dense to Dense Encoder-Decoder')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument("--reference_mesh_file_voca", type=str, default='/home/federico/Scrivania/ST/ScanTalk/template/flame_model/FLAME_sample.ply', help='path of the template')
    parser.add_argument("--epochs", type=int, default=200, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--result_dir", type=str, default='/home/federico/Scrivania/ST/Data/results')
    parser.add_argument("--sample_audio", type=str, default='/home/federico/Scrivania/TH/photo.wav')
    parser.add_argument("--template_file_voca", type=str, default="/home/federico/Scrivania/TH/S2L/vocaset/templates.pkl", help='faces to animate')
    parser.add_argument("--template_file_biwi", type=str, default="/home/federico/Scrivania/ST/Data/Biwi_6/templates", help='faces to animate')
    parser.add_argument("--template_file_multiface", type=str, default="/home/federico/Scrivania/ST/Data/Multiface/templates", help='faces to animate')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
                                                              " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
                                                              " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
                                                              " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA F1 F2 F3 F4 F5 F6 M1 M2 M3 M4"
                                                              " 20171024 20180226 20180227 20180406 20180418 20180426 20180510 20180927 20190529")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
                                                            " FaceTalk_170908_03277_TA F7 M5 20190828 20180105")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
                                                             " FaceTalk_170731_00024_TA F8 M6 20181017 20190521")
    parser.add_argument("--wav_path_voca", type=str, default="/home/federico/Scrivania/TH/S2L/vocaset/wav", help='path of the audio signals')
    parser.add_argument("--vertices_path_voca", type=str, default="/home/federico/Scrivania/TH/S2L/vocaset/vertices_npy", help='path of the ground truth')
    parser.add_argument("--wav_path_biwi", type=str, default="/home/federico/Scrivania/ST/Data/Biwi_6/wav", help='path of the audio signals')
    parser.add_argument("--vertices_path_biwi", type=str, default="/home/federico/Scrivania/ST/Data/Biwi_6/vertices", help='path of the ground truth')
    parser.add_argument("--wav_path_multiface", type=str, default="/home/federico/Scrivania/ST/Data/Multiface/wav", help='path of the audio signals')
    parser.add_argument("--vertices_path_multiface", type=str, default="/home/federico/Scrivania/ST/Data/Multiface/vertices", help='path of the ground truth')

    parser.add_argument("--info", type=str, default="", help='experiment info')
    parser.add_argument("--dataset", type=str, default="multiface", help='dataset to use')
    
    ##Diffusion Net hyperparameters
    parser.add_argument('--latent_channels', type=int, default=32)
    parser.add_argument('--in_channels', type=int, default=3)


    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
