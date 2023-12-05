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
from data_loader_diffusion_net import get_dataloaders
import scipy
import sys
import lddmm_utils
from model.diffusion_net_encoder_lstm import DiffusionNetAutoencoder
sys.path.append('/home/federico/Scrivania/ST/ScanTalk/model/diffusion-net/src')
import diffusion_net



faces = torch.tensor(np.array(trimesh.load('/home/federico/Scrivania/ST/ScanTalk/template/flame_model/FLAME_sample.ply').faces)).to(dtype=torch.int32)

class Varifold_loss(nn.Module):
    def __init__(self, args):
        super(Varifold_loss, self).__init__()
        self.device = args.device
        self.faces = faces.to(self.device)
        self.torchdtype = torch.float
        self.sig = [0.08, 0.04, 0.02]
        self.sig_n = torch.tensor([0.5], dtype=self.torchdtype, device=self.device)
        for i, sigma in enumerate(self.sig):
            self.sig[i] = torch.tensor([sigma], dtype=self.torchdtype, device=self.device)
    def forward(self, predictions, targets):
        L = []
        for i in range(predictions.shape[0]):
            Li = torch.Tensor([0.]).to(self.device)
            V1, F1 = predictions[i], self.faces
            V2, F2 = targets[i], self.faces

            for sigma in self.sig:
                Li += (sigma / self.sig[0]) ** 2 * lddmm_utils.lossVarifoldSurf(F1, V2, F2,
                                                                lddmm_utils.GibbsKernel_varifold_oriented(
                                                                sigma=sigma,
                                                                sigma_n=self.sig_n))(V1)
            L.append(Li)

        return torch.stack(L).mean()

class Masked_Loss(nn.Module):
    def __init__(self, args):
        super(Masked_Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.weights = np.load('/home/federico/Scrivania/ST/ScanTalk/template/template/Normalized_d_weights.npy', allow_pickle=True)
        self.weights = torch.from_numpy(self.weights[:-1]).float().to(args.device)

    def forward(self, predictions, target):
        
        rec_loss = torch.mean(self.mse(predictions, target))
        
        landmarks_loss = (self.mse(predictions, target).mean(axis=2) * self.weights).mean()

        prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_shift = target[:, 1:, :] - target[:, :-1, :]

        vel_loss = torch.mean((self.mse(prediction_shift, target_shift)))

        return rec_loss + 10 * landmarks_loss + 10 * vel_loss


def train(args):

    device = args.device
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    meshpackage = 'trimesh'

    shapedata = shape_data.ShapeData(nVal=100,
                              reference_mesh_file=args.reference_mesh_file,
                              normalization=False,
                              meshpackage=meshpackage, load_flag=False)

    shapedata.n_vertex = 5023
    shapedata.n_features = 3
    
    reference = trimesh.load(args.reference_mesh_file, process=False)
    template_tri = reference.faces
    template_vertices = torch.tensor(reference.vertices).to(args.device).float()
    # generate/load transform matrices
    transform_fp = '/home/federico/Scrivania/ST/ScanTalk/template/template/transform.pkl'
    template_fp = args.reference_mesh_file
    if not os.path.exists(transform_fp):
        print('Generating transform matrices...')
        mesh = Mesh(filename=template_fp)
        ds_factors = [4, 4, 4, 4]
        _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(
            mesh, ds_factors)
        tmp = {
            'vertices': V,
            'face': F,
            'adj': A,
            'down_transform': D,
            'up_transform': U
        }
        print(tmp)

        with open(transform_fp, 'wb') as fp:
            pickle.dump(tmp, fp)
        print('Done!')
        print('Transform matrices are saved in \'{}\''.format(transform_fp))
    else:
        with open(transform_fp, 'rb') as f:
            tmp = pickle.load(f, encoding='latin1')

    spiral_indices_list = [
        utils.preprocess_spiral(tmp['face'][idx], args.seq_length[idx],
                                tmp['vertices'][idx],
                                args.dilation[idx]).to(device)
        for idx in range(len(tmp['face']) - 1)
    ]
    down_transform_list = [
        utils.to_sparse(down_transform).to(device)
        for down_transform in tmp['down_transform']
    ]
    up_transform_list = [
        utils.to_sparse(up_transform).to(device)
        for up_transform in tmp['up_transform']
    ]


    dataset = get_dataloaders(args)
    
    d2d = DiffusionNetAutoencoder(args.in_channels, args.out_channels, args.latent_channels,
           spiral_indices_list, down_transform_list,
           up_transform_list).to(device)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            
    starting_epoch = 0
    if args.load_model == True:
        checkpoint = torch.load(args.model_path, map_location=device)
        d2d.load_state_dict(checkpoint['autoencoder_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        print(starting_epoch)
        
    lip_mask = scipy.io.loadmat('/home/federico/Scrivania/ST/ScanTalk/FLAME_lips_idx.mat')
    
    lip_mask = lip_mask['lips_idx'] - 1 
    
    lip_mask = np.reshape(np.array(lip_mask, dtype=np.int64), (lip_mask.shape[0]))
    
    criterion = Masked_Loss(args) 
    criterion_val = nn.MSELoss()

    optim = torch.optim.Adam(d2d.parameters(), lr=args.lr)

    for epoch in range(starting_epoch, args.epochs):
        d2d.train()
        tloss = 0
        
        pbar_talk = tqdm(enumerate( dataset["train"]), total=len( dataset["train"]))
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
            vertices_pred = d2d.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY)
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
                pbar_talk = tqdm(enumerate(dataset["valid"]), total=len( dataset["valid"]))
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
                    vertices_pred = d2d.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY)
                    loss = criterion_val(vertices, vertices_pred)
                    t_test_loss += loss.item()
                    pbar_talk.set_description(
                                    "(Epoch {}) VAL LOSS:{:.10f}".format((epoch + 1), (t_test_loss)/(b+1)))
                    
                #Sample from external audio and face
                #Process the audio
                speech_array, sampling_rate = librosa.load(args.sample_audio, sr=16000)
                audio_feature = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)
                audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
                audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)
                #Compute Operators for DiffuserNet
                frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(template_vertices.to('cpu'), faces=None, k_eig=128)
                mass = torch.FloatTensor(np.array(mass)).float().to(device).unsqueeze(0)
                evals = torch.FloatTensor(np.array(evals)).to(device).unsqueeze(0)
                evecs = torch.FloatTensor(np.array(evecs)).to(device).unsqueeze(0)
                L = L.float().to(device).unsqueeze(0)
                gradX = gradX.float().to(device).unsqueeze(0)
                gradY = gradY.float().to(device).unsqueeze(0)
                
                gen_seq = d2d.predict(audio_feature, template_vertices.to(device).unsqueeze(0), mass, L, evals, evecs, gradX, gradY)
                
                gen_seq = gen_seq.cpu().detach().numpy()
                
                os.makedirs('/home/federico/Scrivania/ST/Data/saves/Meshes_DiffuserNet_Encoder/' + str(epoch), exist_ok=True)
                for m in range(len(gen_seq)):
                    mesh = trimesh.Trimesh(gen_seq[m], template_tri)
                    mesh.export('/home/federico/Scrivania/ST/Data/saves/Meshes_DiffuserNet_Encoder/' + str(epoch) + '/frame_' + str(m).zfill(3) + '.ply')
                
                #Sample from training set
                #Process the audio
                speech_array, sampling_rate = librosa.load(args.training_sample_audio, sr=16000)
                audio_feature = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)
                audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
                audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)
                with open(args.template_file, 'rb') as fin:
                    templates = pickle.load(fin, encoding='latin1')

                face = templates[args.training_sample_face]
                
                face = torch.tensor(face).to(args.device).float()
                
                #Compute Operators for DiffuserNet
                frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(face.to('cpu'), faces=None, k_eig=128)
                mass = torch.FloatTensor(np.array(mass)).float().to(device).unsqueeze(0)
                evals = torch.FloatTensor(np.array(evals)).to(device).unsqueeze(0)
                evecs = torch.FloatTensor(np.array(evecs)).to(device).unsqueeze(0)
                L = L.float().to(device).unsqueeze(0)
                gradX = gradX.float().to(device).unsqueeze(0)
                gradY = gradY.float().to(device).unsqueeze(0)
                
                gen_seq = d2d.predict(audio_feature, face.to(device).unsqueeze(0), mass, L, evals, evecs, gradX, gradY)
                
                gen_seq = gen_seq.cpu().detach().numpy()
                
                os.makedirs('/home/federico/Scrivania/ST/Data/saves/Meshes_Training_DiffuserNet_Encoder/' + str(epoch), exist_ok=True)
                for m in range(len(gen_seq)):
                    mesh = trimesh.Trimesh(gen_seq[m], template_tri)
                    mesh.export('/home/federico/Scrivania/ST/Data/saves/Meshes_Training_DiffuserNet_Encoder/' + str(epoch) + '/frame_' + str(m).zfill(3) + '.ply')
                    
                error_lve = 0
                count = 0
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
                    vertices_pred = d2d.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY)
                    vertices = vertices.detach().cpu().numpy()
                    vertices_pred = vertices_pred.detach().cpu().numpy()
                    for k in range(vertices_pred.shape[0]):
                        error_lve += ((vertices_pred[k] - vertices[k]) ** 2)[lip_mask].max()
                        count += 1
                    pbar_talk.set_description(" LVE:{:.8f}".format((error_lve)/(count)))
                
        torch.save({'epoch': epoch,
            'autoencoder_state_dict': d2d.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            }, os.path.join(args.result_dir, 'd2d_ScanTalk_DiffuserNet_Encoder.pth.tar'))
    
        
        
        
def main():
    parser = argparse.ArgumentParser(description='D2D: Dense to Dense Encoder-Decoder')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument("--reference_mesh_file", type=str, default='/home/federico/Scrivania/ScanTalk2.0/ScanTalk-thomas/template/flame_model/FLAME_sample.ply', help='path of the template')
    parser.add_argument("--epochs", type=int, default=300, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--result_dir", type=str, default='/home/federico/Scrivania/ST/Data/results')
    parser.add_argument("--sample_audio", type=str, default='/home/federico/Scrivania/TH/photo.wav')
    parser.add_argument("--training_sample_audio", type=str, default='/home/federico/Scrivania/TH/S2L/vocaset/wav/FaceTalk_170725_00137_TA_sentence02.wav')
    parser.add_argument("--template_file", type=str, default="/home/federico/Scrivania/TH/S2L/vocaset/templates.pkl", help='faces to animate')
    parser.add_argument("--training_sample_face", type=str, default="FaceTalk_170725_00137_TA", help='face to animate')
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--model_path", type=str, default='/home/federico/Scrivania/ST/Data/results/d2d_ScanTalk_bigger_lstm_masked_velocity_loss.pth.tar')
    parser.add_argument("--mask_path", type=str, default='/home/federico/Scrivania/ST/ScanTalk/mouth_region_registered_idx.npy')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
                                                              " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
                                                              " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
                                                              " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
                                                            " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
                                                             " FaceTalk_170731_00024_TA")
    parser.add_argument("--wav_path", type=str, default="/home/federico/Scrivania/TH/S2L/vocaset/wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="/home/federico/Scrivania/TH/S2L/vocaset/vertices_npy", help='path of the ground truth')

    ##Spiral++ hyperparameters
    parser.add_argument('--out_channels',
                        nargs='+',
                        default=[16, 32, 64, 128],#divided by 2
                        type=int)
    parser.add_argument('--latent_channels', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--seq_length', type=int, default=[12, 12, 12, 12], nargs='+')
    parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()