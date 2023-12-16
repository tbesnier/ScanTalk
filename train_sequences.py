import os
import pickle
import trimesh
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from model.model_semi_registered import PointNet2SpiralsAutoEncoder, PointNet2NJFAutoEncoder
from model.model_registered import SpiralAutoencoder
import librosa
from transformers import Wav2Vec2Processor
from psbody.mesh import Mesh
from utils import utils, mesh_sampling
import shape_data
from data_loader import get_dataloaders
import scipy
from pytorch3d.loss import(
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

from pytorch3d.structures import Meshes

import lddmm_utils

faces = torch.tensor(np.array(trimesh.load('./template/flame_model/FLAME_sample.ply').faces)).to(dtype=torch.int32)

class Masked_Loss(nn.Module):
    def __init__(self, args):
        super(Masked_Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.weights = np.load('./template/template/Normalized_d_weights.npy', allow_pickle=True)
        self.weights = torch.from_numpy(self.weights[:-1]).float().to(args.device)

    def forward(self, predictions, target):

        rec_loss = torch.mean(self.mse(predictions, target))

        landmarks_loss = (self.mse(predictions, target).mean(axis=2) * self.weights).mean()

        prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_shift = target[:, 1:, :] - target[:, :-1, :]

        vel_loss = torch.mean((self.mse(prediction_shift, target_shift)))

        return rec_loss + 10 * landmarks_loss# + 10 * vel_loss

class Chamfer_Loss(nn.Module):
    def __init__(self, args):
        super(Chamfer_Loss, self).__init__()
        self.device = args.device
        self.faces = faces.to(self.device)
        self.w_edge_loss = 0.5
        self.w_laplacian_loss = 0.05
        self.w_normal_loss = 0.01
    def forward(self, predictions, targets):

        loss_chamfer, _ = chamfer_distance(predictions, targets)

        M = Meshes(verts=predictions,
                   faces=self.faces.repeat(predictions.shape[0], 1, 1))

        if self.w_edge_loss > 0:
            self.w_edge_loss = self.w_edge_loss * mesh_edge_loss(M)
        if self.w_laplacian_loss > 0:
            self.w_laplacian_loss = self.w_laplacian_loss * mesh_laplacian_smoothing(M, method="cot")  # mesh laplacian smoothing
        if self.w_normal_loss > 0:
            self.w_normal_loss = self.w_normal_loss * mesh_normal_consistency(M)

        prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_shift = targets[:, 1:, :] - targets[:, :-1, :]

        vel_loss = chamfer_distance(prediction_shift, target_shift)[0]

        return loss_chamfer + self.w_laplacian_loss + self.w_normal_loss + self.w_edge_loss + 10*vel_loss


class Varifold_loss(nn.Module):
    def __init__(self, args):
        super(Varifold_loss, self).__init__()
        self.device = args.device
        self.faces = faces.to(self.device)
        self.torchdtype = torch.float
        self.sig = [0.08, 0.02]
        #self.sig_n = torch.tensor([0.5], dtype=self.torchdtype, device=self.device)
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
                                                                lddmm_utils.GaussSquaredKernel_varifold_unoriented(
                                                                sigma=sigma))(V1)

            L.append(Li)

        return torch.stack(L).mean()

def train(args):
    use_spirals=True
    device = args.device
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

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
    if use_spirals:
        transform_fp = './template/template/transform.pkl'
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

    d2d = PointNet2SpiralsAutoEncoder(args.latent_channels, args.in_channels, args.out_channels,
                                      spiral_indices_list, down_transform_list, up_transform_list).to(args.device)

    #d2d = SpiralAutoencoder(args.in_channels, args.out_channels, args.latent_channels,
    #       spiral_indices_list, down_transform_list,
    #       up_transform_list).to(args.device)

    #d2d = PointNet2NJFAutoEncoder(latent_channels=args.latent_channels, point_dim=8).to(args.device)

    starting_epoch = 0
    if args.load_model == True:
        checkpoint = torch.load(args.model_path, map_location=device)
        d2d.load_state_dict(checkpoint['autoencoder_state_dict'])
        starting_epoch = checkpoint['epoch']
        print(starting_epoch)

    lip_mask = scipy.io.loadmat('./FLAME_lips_idx.mat')

    lip_mask = lip_mask['lips_idx'] - 1

    lip_mask = np.reshape(np.array(lip_mask, dtype=np.int64), (lip_mask.shape[0]))

    criterion = Masked_Loss(args)  ##Chamfer_Loss(args)  ##Varifold_loss(args)  ##Masked_Loss(args)
    criterion_val = nn.MSELoss()

    optim = torch.optim.Adam(d2d.parameters(), lr=args.lr)

    for epoch in range(starting_epoch, args.epochs):
        d2d.train()
        tloss = 0

        pbar_talk = tqdm(enumerate(dataset["train"]), total=len(dataset["train"]))
        for b, sample in pbar_talk:
            audio = sample[0].to(device)
            vertices = sample[1].to(device).squeeze(0)
            template = sample[2].to(device)
            vertices_pred = d2d.forward(audio, template, vertices)
            optim.zero_grad()

            loss = criterion(vertices, vertices_pred)
            torch.nn.utils.clip_grad_norm_(d2d.parameters(), 10.0)
            loss.backward()
            optim.step()
            tloss += loss.item()
            pbar_talk.set_description(
                "(Epoch {}) TRAIN LOSS:{:.10f}".format((epoch + 1), tloss / (b + 1)))

        if epoch % 10 == 0:
            d2d.eval()
            with torch.no_grad():
                t_test_loss = 0
                pbar_talk = tqdm(enumerate(dataset["valid"]), total=len(dataset["valid"]))
                for b, sample in pbar_talk:
                    audio = sample[0].to(device)
                    vertices = sample[1].to(device).squeeze(0)
                    template = sample[2].to(device)
                    vertices_pred = d2d.forward(audio, template, vertices)
                    loss = criterion_val(vertices, vertices_pred)
                    t_test_loss += loss.item()
                    pbar_talk.set_description(
                        "(Epoch {}) VAL LOSS:{:.10f}".format((epoch + 1), (t_test_loss) / (b + 1)))

                # Sample from external audio and face
                speech_array, sampling_rate = librosa.load(args.sample_audio, sr=16000)
                audio_feature = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)
                audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
                audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)
                gen_seq = d2d.predict(audio_feature, template_vertices.float())

                gen_seq = gen_seq.cpu().detach().numpy()

                os.makedirs(
                    '../Data/VOCA/res/Results_Actor/Meshes/' + str(epoch),
                    exist_ok=True)
                for m in range(len(gen_seq)):
                    mesh = trimesh.Trimesh(gen_seq[m], template_tri)
                    mesh.export('../Data/VOCA/res/Results_Actor/Meshes/' + str(
                        epoch) + '/frame_' + str(m).zfill(3) + '.ply')

                # Sample from training set
                speech_array, sampling_rate = librosa.load(args.training_sample_audio, sr=16000)
                audio_feature = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)
                audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
                audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)
                with open(args.template_file, 'rb') as fin:
                    templates = pickle.load(fin, encoding='latin1')

                face = templates[args.training_sample_face]

                face = torch.tensor(face).to(args.device)

                gen_seq = d2d.predict(audio_feature, face.float().unsqueeze(0))

                gen_seq = gen_seq.cpu().detach().numpy()

                os.makedirs(
                    '../Data/VOCA/res/Results_Actor/Meshes_Training/' + str(
                        epoch), exist_ok=True)
                for m in range(len(gen_seq)):
                    mesh = trimesh.Trimesh(gen_seq[m], template_tri)
                    mesh.export(
                        '../Data/VOCA/res/Results_Actor/Meshes_Training/' + str(
                            epoch) + '/frame_' + str(m).zfill(3) + '.ply')

        torch.save({'epoch': epoch,
                    'autoencoder_state_dict': d2d.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    }, os.path.join(args.result_dir, 'd2d_ScanTalk_bigger_lstm_masked_velocity_loss.pth.tar'))

    with torch.no_grad():
        d2d.eval()
        error_lve = 0
        count = 0
        pbar_talk = tqdm(enumerate(dataset["test"]), total=len(dataset["test"]))
        for b, sample in pbar_talk:
            audio = sample[0].to(device)
            vertices = sample[1].to(device).squeeze(0)
            template = sample[2].to(device)
            vertices_pred = d2d.forward(audio, template, vertices)
            for k in range(vertices_pred.shape[0]):
                error_lve += ((vertices_pred[k] - vertices[k]) ** 2)[lip_mask].max()
                count += 1
            pbar_talk.set_description(" LVE:{:.8f}".format((error_lve) / (b + 1) * count))


def main():
    parser = argparse.ArgumentParser(description='D2D: Dense to Dense Encoder-Decoder')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument("--reference_mesh_file", type=str,
                        default='./template/flame_model/FLAME_sample.ply',
                        help='path of the template')
    parser.add_argument("--epochs", type=int, default=300, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--result_dir", type=str, default='../Data/VOCA/res/Results_Actor/Models')
    parser.add_argument("--sample_audio", type=str, default='../Data/VOCA/res/TH/photo.wav')
    parser.add_argument("--training_sample_audio", type=str,
                        default='./FaceTalk_170725_00137_TA_sentence01.wav')
    parser.add_argument("--template_file", type=str, default="../datasets/VOCA_training/templates.pkl",
                        help='faces to animate')
    parser.add_argument("--training_sample_face", type=str, default="FaceTalk_170725_00137_TA", help='face to animate')
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--model_path", type=str,
                        default='../Data/VOCA/res/Results_Actor/Models/d2d_ScanTalk_new_training_strat_disp.pth.tar')
    parser.add_argument("--mask_path", type=str,
                        default='./mouth_region_registered_idx.npy')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
                                                              " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
                                                              " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
                                                              " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
                                                            " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
                                                             " FaceTalk_170731_00024_TA")
    parser.add_argument("--wav_path", type=str, default="../datasets/VOCA_training/wav",
                        help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="../Data/VOCA/preprocessed/vertices_npy",
                        help='path of the ground truth')
    parser.add_argument("--vertex_normals_path", type=str, default="../Data/VOCA/preprocessed/verts_normals_npy",
                        help='path of the ground truth')

    ##Spiral++ hyperparameters
    parser.add_argument('--out_channels',
                        nargs='+',
                        default=[32, 64, 64, 128],  # divided by 2
                        type=int)
    parser.add_argument('--latent_channels', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--seq_length', type=int, default=[9, 9, 9, 9], nargs='+')
    parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()