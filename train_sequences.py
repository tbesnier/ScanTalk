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
from d2d import SpiralAutoencoder
import librosa
import random
from transformers import Wav2Vec2Processor
from wav2vec import Wav2Vec2Model
from psbody.mesh import Mesh
from utils import utils, mesh_sampling
from data_loader import get_dataloaders

class Masked_Loss(nn.Module):
    def __init__(self, args):
        super(Masked_Loss, self).__init__()
        self.indices = np.load(args.mask_path)
        self.mask = torch.zeros((5023, 3))
        self.mask[self.indices] = 1
        self.mask = self.mask.to(args.device)
        self.mse = nn.MSELoss(reduction='none')
        self.number_of_indices = self.indices.shape[0]

        self.weights = np.load('/home/federico/Scrivania/ST/ScanTalk/template/template/Normalized_d_weights.npy', allow_pickle=True)
        self.weights = torch.from_numpy(self.weights[:-1]).float().to(args.device)

    def forward(self, predictions, target):
        #mb = predictions.shape[0]
        rec_loss = torch.mean(self.mse(predictions, target))

        #mouth_rec_loss = torch.sum(self.mse(predictions * self.mask, target * self.weights)) / (
        #           self.number_of_indices * mb)

        #prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
        #target_shift = target[:, 1:, :] - target[:, :-1, :]

        #vel_loss = torch.mean(
        #    (self.mse(prediction_shift, target_shift)))

        return rec_loss# + mouth_rec_loss# + 0.2*vel_loss

    def forward_weighted(self, predictions, target):

        L = (torch.matmul(self.weights, self.mse(predictions, target))).mean()

        return L

def train(args):

    device = args.device
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    meshpackage = 'trimesh'

    shapedata = shape_data.ShapeData(nVal=100,
                              reference_mesh_file=args.reference_mesh_file,
                              normalization=False,
                              meshpackage=meshpackage, load_flag=False)

    shapedata.n_vertex = 5023
    shapedata.n_features = 3
    
    reference = trimesh.load(args.reference_mesh_file, process=False)
    template_tri = reference.faces
    template_vertices = torch.tensor(reference.vertices).unsqueeze(0).to(args.device)

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
    
    d2d = SpiralAutoencoder(args.in_channels, args.out_channels, args.latent_channels,
           spiral_indices_list, down_transform_list,
           up_transform_list).to(device)

    starting_epoch = 0
    if args.load_model == True:
        checkpoint = torch.load(args.model_path, map_location=device)
        d2d.load_state_dict(checkpoint['autoencoder_state_dict'])
        starting_epoch = checkpoint['epoch']
        print(starting_epoch)
    
    criterion = nn.MSELoss()#Masked_Loss(args) #Masked_Loss(args)  # nn.MSELoss()

    optim = torch.optim.Adam(d2d.parameters(), lr=args.lr)

    for epoch in range(starting_epoch, args.epochs):
        d2d.train()
        tloss = 0
        
        pbar_talk = tqdm(enumerate( dataset["train"]), total=len( dataset["train"]))
        for b, sample in pbar_talk:
            audio = sample[0].to(device)
            vertices = sample[1].to(device).squeeze(0)
            template = sample[2].to(device)
            vertices_pred = d2d.forward(audio, template, vertices)
            optim.zero_grad()

            #loss = criterion.forward_weighted(vertices, vertices_pred) + criterion(vertices_pred - template, vertices - template)ù
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
                    vertices_pred = d2d.forward(audio, template, vertices)
                    #loss_ = criterion.forward_weighted(vertices, vertices_pred)
                    loss_ = criterion(vertices, vertices_pred)
                    t_test_loss += loss_
                    pbar_talk.set_description(
                                    "(Epoch {}) VAL LOSS:{:.10f}".format((epoch + 1), (t_test_loss)/(b+1)))
                
                #Sample from external audio and face
                speech_array, sampling_rate = librosa.load(args.sample_audio, sr=16000)
                audio_feature = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)
                audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
                audio_feature = torch.FloatTensor(audio_feature)
                hidden_states = audio_encoder(audio_feature).last_hidden_state.to(args.device)
                
                gen_seq = d2d.predict(hidden_states, template_vertices.float())
                
                gen_seq = gen_seq.cpu().detach().numpy()
                
                os.makedirs('/home/federico/Scrivania/ST/Data/saves/Meshes/' + str(epoch), exist_ok=True)
                for m in range(len(gen_seq)):
                    mesh = trimesh.Trimesh(gen_seq[m], template_tri)
                    mesh.export('/home/federico/Scrivania/ST/Data/saves/Meshes/' + str(epoch) + '/frame_' + str(m).zfill(3) + '.ply')
                
                #Sample from training set
                speech_array, sampling_rate = librosa.load(args.training_sample_audio, sr=16000)
                audio_feature = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)
                audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
                audio_feature = torch.FloatTensor(audio_feature)
                hidden_states = audio_encoder(audio_feature).last_hidden_state.to(args.device)
                with open(args.template_file, 'rb') as fin:
                    templates = pickle.load(fin, encoding='latin1')

                face = templates[args.training_sample_face]
                
                face = torch.tensor(face).to(args.device)
                
                gen_seq = d2d.predict(hidden_states, face.float().unsqueeze(0))
                
                gen_seq = gen_seq.cpu().detach().numpy()
                
                os.makedirs('/home/federico/Scrivania/ST/Data/saves/Meshes_Training/' + str(epoch), exist_ok=True)
                for m in range(len(gen_seq)):
                    mesh = trimesh.Trimesh(gen_seq[m], template_tri)
                    mesh.export('/home/federico/Scrivania/ST/Data/saves/Meshes_Training/' + str(epoch) + '/frame_' + str(m).zfill(3) + '.ply')
         
        torch.save({'epoch': epoch,
                    'autoencoder_state_dict': d2d.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    }, os.path.join(args.result_dir, 'd2d_ScanTalk_new_training_strat_disp.pth.tar'))
               
            

def main():
    parser = argparse.ArgumentParser(description='D2D: Dense to Dense Encoder-Decoder')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument("--reference_mesh_file", type=str, default='/home/federico/Scrivania/ScanTalk2.0/ScanTalk-thomas/template/flame_model/FLAME_sample.ply', help='path of the template')
    parser.add_argument("--epochs", type=int, default=200, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--result_dir", type=str, default='/home/federico/Scrivania/ST/Data/results')
    parser.add_argument("--sample_audio", type=str, default='/home/federico/Scrivania/TH/photo.wav')
    parser.add_argument("--training_sample_audio", type=str, default='/home/federico/Scrivania/TH/S2L/vocaset/wav/FaceTalk_170725_00137_TA_sentence02.wav')
    parser.add_argument("--template_file", type=str, default="/home/federico/Scrivania/TH/S2L/vocaset/templates.pkl", help='faces to animate')
    parser.add_argument("--training_sample_face", type=str, default="FaceTalk_170725_00137_TA", help='face to animate')
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--model_path", type=str, default='../Data/VOCA/res/Results_Actor/Models/d2d_ScanTalk_new_training_strat_disp.pth.tar')
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
                        default=[32, 64, 64, 128],#divided by 2
                        type=int)
    parser.add_argument('--latent_channels', type=int, default=32)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--seq_length', type=int, default=[12, 12, 12, 12], nargs='+')
    parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()