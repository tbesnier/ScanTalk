import os
import spiral_utils
import shape_data
import new_data_loader as new_data_loader
import pickle
import trimesh
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from d2d_pretrain import SpiralAutoencoder
import librosa
from psbody.mesh import Mesh
from utils import utils, mesh_sampling


class Masked_Loss(nn.Module):
    def __init__(self, args):
        super(Masked_Loss, self).__init__()
        self.indices = np.load(args.mask_path)
        self.weights = torch.zeros((5023, 3))
        self.weights[self.indices] = 1
        self.weights = self.weights.to(args.device)
        self.mse = nn.MSELoss(reduction='none')
        self.number_of_indices = self.indices.shape[0]

    def forward(self, predictions, target):
        rec_loss = torch.mean(self.mse(predictions, target))

        return rec_loss  # + 0.2*mouth_rec_loss# + 0.2*vel_loss


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
    template_vertices = torch.tensor(reference.vertices).unsqueeze(0).to(args.device)

    # generate/load transform matrices
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

    dataset_train = new_data_loader.TH_pretrain_spiral_Dataset(args.dataset_dir_frame,
                                                   args.dataset_dir_actor,
                                                   1,
                                                   100000)

    dataset_test = new_data_loader.TH_pretrain_spiral_Dataset(args.dataset_dir_frame,
                                                  args.dataset_dir_actor,
                                                  100000,
                                                  121000)

    dataloader_train = DataLoader(dataset_train, batch_size=32,
                                  shuffle=True, num_workers=4)

    dataloader_test = DataLoader(dataset_test, batch_size=32,
                                 shuffle=True, num_workers=4)

    d2d = SpiralAutoencoder(args.in_channels, args.out_channels, args.latent_channels,
                            spiral_indices_list, down_transform_list,
                            up_transform_list).to(device)

    starting_epoch = 0
    if args.load_model == True:
        checkpoint = torch.load(args.model_path, map_location=device)
        d2d.load_state_dict(checkpoint['autoencoder_state_dict'])
        starting_epoch = checkpoint['epoch']
        print(starting_epoch)

    criterion = nn.MSELoss()

    optim = torch.optim.Adam(d2d.parameters(), lr=args.lr)

    for epoch in range(starting_epoch, args.epochs):
        d2d.train()
        tloss = 0
        pbar_talk = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        for b, sample in pbar_talk:
            optim.zero_grad()
            frame = sample['frame'].to(device)
            actor = sample['actor'].to(device)
            frame_pred = d2d.forward(frame, actor)
            loss = criterion(frame, frame_pred)
            loss.backward()
            optim.step()
            tloss += loss
            pbar_talk.set_description(
                "(Epoch {}) TRAIN LOSS:{:.10f}".format((epoch + 1), tloss / (b + 1)))

        if epoch % 10 == 0:
            d2d.eval()
            with torch.no_grad():
                t_test_loss = 0
                pbar_talk = tqdm(enumerate(dataloader_test), total=len(dataloader_test))
                for b, sample in pbar_talk:
                    frame = sample['frame'].to(device)
                    actor = sample['actor'].to(device)
                    frame_pred = d2d.forward(frame, actor)
                    loss = criterion(frame, frame_pred)
                    t_test_loss += loss
                    pbar_talk.set_description(
                        "(Epoch {}) TEST LOSS:{:.10f}".format((epoch + 1), (t_test_loss) / (b + 1)))

                    mesh = trimesh.Trimesh(frame_pred[0].detach().cpu().numpy(), template_tri)
                    mesh.export(
                        '../Data/VOCA/res/Results_pretrain/Meshes/' + str(epoch) + '.ply')

                with open(args.template_file, 'rb') as fin:
                    templates = pickle.load(fin, encoding='latin1')

                face = templates[args.training_sample_face]
                face = torch.tensor(face).to(args.device)

                mesh = trimesh.Trimesh(face.detach().cpu().numpy(), template_tri)
                mesh.export(
                    '../Data/VOCA/res/Results_pretrain/Meshes_Training/' + str(epoch) + '.ply')

        torch.save({'epoch': epoch,
                    'autoencoder_state_dict': d2d.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    }, os.path.join(args.result_dir, 'd2d_ScanTalk_new_training_strat_disp.pth.tar'))


def main():
    parser = argparse.ArgumentParser(description='D2D: Dense to Dense Encoder-Decoder')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument("--reference_mesh_file", type=str, default='./template/flame_model/FLAME_sample.ply',
                        help='path of the template')
    parser.add_argument("--epochs", type=int, default=300, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset_dir_audios", type=str, default='../Data/VOCA/Consecutive_Dataset/Audio_Snippet')
    parser.add_argument("--dataset_dir_frame", type=str, default='../Data/VOCA/Consecutive_Dataset/Frame')
    parser.add_argument("--dataset_dir_actor", type=str, default='../Data/VOCA/Consecutive_Dataset/Actor')
    parser.add_argument("--dataset_dir_next_frame", type=str, default='../Data/VOCA/Consecutive_Dataset/Next_Frame')
    parser.add_argument("--result_dir", type=str, default='../Data/VOCA/res/Results_pretrain/Models')
    parser.add_argument("--template_file", type=str, default="../datasets/VOCA_training/templates.pkl",
                        help='faces to animate')
    parser.add_argument("--training_sample_face", type=str, default="FaceTalk_170725_00137_TA", help='face to animate')
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--model_path", type=str,
                        default='../Data/VOCA/res/Results_pretrain/Models/d2d_ScanTalk_new_training_strat_disp.pth.tar')
    parser.add_argument("--mask_path", type=str, default='./mouth_region_registered_idx.npy')

    ##Spiral++ hyperparameters
    parser.add_argument('--out_channels',
                        nargs='+',
                        default=[64, 64, 64, 128],
                        type=int)
    parser.add_argument('--latent_channels', type=int, default=32)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--seq_length', type=int, default=[9, 9, 9, 9], nargs='+')
    parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()