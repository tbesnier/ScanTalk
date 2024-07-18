import os, sys
import pickle
import trimesh
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import librosa
from transformers import Wav2Vec2Processor
from data_loader_diffusion_net_multi import get_dataloaders
import lddmm_utils
#from model.model_unregistered import DiffusionNetAutoencoder
from model.model_unregistered_hubert import DiffusionNetAutoencoder
#from model.model_unregistered_wavlm import DiffusionNetAutoencoder
from pytorch3d.loss import(
    chamfer_distance,
)

#from pytorch3d.structures import Meshes

sys.path.append('./model/diffusion-net/src')
import model.diffusion_net as diffusion_net

#faces = torch.tensor(np.array(trimesh.load('./template/flame_model/FLAME_sample.ply').faces)).to(dtype=torch.int32)
faces = torch.tensor(np.array(trimesh.load('../datasets/multiface/data/templates/20180227.ply').faces)).to(dtype=torch.int32)

class Varifold_loss(nn.Module):
    def __init__(self, args):
        super(Varifold_loss, self).__init__()
        self.device = args.device
        self.faces = faces.to(self.device)
        self.torchdtype = torch.float
        self.sig = [0.02, 0.01, 0.008]
        self.sig_n = torch.tensor([0.5], dtype=self.torchdtype, device=self.device)
        for i, sigma in enumerate(self.sig):
            self.sig[i] = torch.tensor([sigma], dtype=self.torchdtype, device=self.device)
    def forward(self, predictions, targets):
        L = torch.Tensor([0.]).to(self.device)
        for i in range(predictions.shape[0]):
            Li = torch.Tensor([0.]).to(self.device)
            V1, F1 = predictions[i], self.faces
            V2, F2 = targets[i], self.faces

            for sigma in self.sig:
                Li += (sigma / self.sig[0]) ** 2 *lddmm_utils.lossVarifoldSurf(F1, V2, F2, lddmm_utils.GibbsKernel_varifold_oriented(
                                                                sigma=sigma, sigma_n=self.sig_n))(V1)

            L += Li

        #predictions.reshape(1, predictions.shape[0], predictions.shape[1] * predictions.shape[2])
        #targets.reshape(1, targets.shape[0], targets.shape[1] * targets.shape[2])

        #loss_chamfer, _ = chamfer_distance(predictions, targets)

        return L/predictions.shape[0]# + 0.1*loss_chamfer


class Masked_Loss(nn.Module):
    def __init__(self, args):
        super(Masked_Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        #self.weights = np.load('./template/template/Normalized_d_weights.npy',
        #                       allow_pickle=True)
        #self.weights = torch.from_numpy(self.weights[:-1]).float().to(args.device)

    def forward(self, predictions, target):

        rec_loss = torch.mean(self.mse(predictions, target))

        #landmarks_loss = (self.mse(predictions, target).mean(axis=2) * self.weights).mean()

        prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_shift = target[:, 1:, :] - target[:, :-1, :]

        vel_loss = torch.mean((self.mse(prediction_shift, target_shift)))

        return rec_loss + vel_loss


def train(args):
    device = args.device
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    reference_voca = trimesh.load(args.reference_mesh_file_voca, process=False)
    template_tri_voca = reference_voca.faces
    template_vertices_voca = torch.tensor(reference_voca.vertices).to(args.device).float()

    reference_multiface = trimesh.load("../datasets/multiface/data/templates/20180227.ply")
    template_tri_multiface = reference_multiface.faces
    template_vertices_multiface = torch.tensor(reference_multiface.vertices).to(args.device).float()

    dataset = get_dataloaders(args)

    d2d = DiffusionNetAutoencoder(args.in_channels, args.latent_channels, args.dataset, args.audio_latent).to(args.device)

    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")

    starting_epoch = 0
    if args.load_model == True:
        checkpoint = torch.load(args.model_path, map_location=device)
        d2d.load_state_dict(checkpoint['autoencoder_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        print(starting_epoch)

    criterion = nn.MSELoss()
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
            mass = sample[3].to(device)
            L = sample[4].to(device)
            evals = sample[5].to(device)
            evecs = sample[6].to(device)
            gradX = sample[7].to(device)
            gradY = sample[8].to(device)
            faces = sample[10].to(device)
            vertices_pred = d2d.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY, faces)

            optim.zero_grad()

            loss = criterion(vertices_pred, vertices)
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
                    mass = sample[3].to(device)
                    L = sample[4].to(device)
                    evals = sample[5].to(device)
                    evecs = sample[6].to(device)
                    gradX = sample[7].to(device)
                    gradY = sample[8].to(device)
                    faces = sample[10].to(device)
                    vertices_pred = d2d.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY, faces)
                    t_test_loss += criterion_val(vertices_pred, vertices).item()
                    pbar_talk.set_description(
                        "(Epoch {}) VAL LOSS:{:.10f}".format((epoch + 1), (t_test_loss) / (b + 1)))

                # Sample from external audio and face
                # Process the audio
                speech_array, sampling_rate = librosa.load(args.sample_audio, sr=16000)
                audio_feature = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)
                audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
                audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)
                # audio_feature = audio_encoder(audio_feature, frame_num=len(vertices)).last_hidden_state.to(device=args.device)

                # Compute Operators for DiffuserNet
                frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
                    torch.tensor(template_vertices_multiface).to('cpu'), faces=torch.tensor(template_tri_multiface), k_eig=args.k_eig)
                mass = torch.FloatTensor(np.array(mass)).float().to(device).unsqueeze(0)
                evals = torch.FloatTensor(np.array(evals)).to(device).unsqueeze(0)
                evecs = torch.FloatTensor(np.array(evecs)).to(device).unsqueeze(0)
                L = L.float().to(device).unsqueeze(0)
                gradX = gradX.float().to(device).unsqueeze(0)
                gradY = gradY.float().to(device).unsqueeze(0)
                faces = torch.tensor(template_tri_multiface).to(device).float().unsqueeze(0)

                gen_seq = d2d.predict(audio_feature, template_vertices_multiface.to(device).unsqueeze(0), mass, L, evals, evecs,
                                      gradX, gradY, faces)

                gen_seq = gen_seq.cpu().detach().numpy()

                os.makedirs('../Data/multiface/res/Results_Actor/Meshes/' + str(epoch),
                            exist_ok=True)
                for m in range(len(gen_seq)):
                    mesh = trimesh.Trimesh(gen_seq[m], template_tri_multiface)
                    mesh.export('../Data/multiface/res/Results_Actor/Meshes/' + str(
                        epoch) + '/frame_' + str(m).zfill(3) + '.ply')

        torch.save({'epoch': epoch,
                    'autoencoder_state_dict': d2d.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    }, os.path.join(args.result_dir, 'ScanTalk_DiffusionNet_Encoder_Decoder_Hubert_MULTIFACE.pth.tar'))

    d2d.eval()
    with torch.no_grad():
        t_test_loss = 0
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
            vertices_pred = d2d.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY, faces)
            t_test_loss += criterion_val(vertices_pred, vertices).item()
            pbar_talk.set_description(
                "(Epoch {}) TEST LOSS:{:.10f}".format((epoch + 1), (t_test_loss) / (b + 1)))

        # Sample from external audio and face
        # Process the audio
        speech_array, sampling_rate = librosa.load(args.sample_audio, sr=16000)
        audio_feature = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)
        audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
        audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)
        # audio_feature = audio_encoder(audio_feature, frame_num=len(vertices)).last_hidden_state.to(device=args.device)

        # Compute Operators for DiffuserNet
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
            template_vertices_voca.to('cpu'), faces=torch.tensor(template_tri_multiface), k_eig=args.k_eig)
        mass = torch.FloatTensor(np.array(mass)).float().to(device).unsqueeze(0)
        evals = torch.FloatTensor(np.array(evals)).to(device).unsqueeze(0)
        evecs = torch.FloatTensor(np.array(evecs)).to(device).unsqueeze(0)
        L = L.float().to(device).unsqueeze(0)
        gradX = gradX.float().to(device).unsqueeze(0)
        gradY = gradY.float().to(device).unsqueeze(0)
        faces = torch.tensor(template_tri_multiface).to(device).float().unsqueeze(0)

        gen_seq = d2d.predict(audio_feature, template_vertices_multiface.to(device).unsqueeze(0), mass, L, evals, evecs,
                              gradX, gradY, faces)

        gen_seq = gen_seq.cpu().detach().numpy()

        os.makedirs('../Data/VOCA_multiface/res/Results_Actor/Meshes_test/' + str(epoch),
                    exist_ok=True)
        for m in range(len(gen_seq)):
            mesh = trimesh.Trimesh(gen_seq[m], template_tri_multiface)
            mesh.export('../Data/VOCA_multiface/res/Results_Actor/Meshes_test/' + str(
                epoch) + '/frame_' + str(m).zfill(3) + '.ply')


def main():
    parser = argparse.ArgumentParser(description='D2D: Dense to Dense Encoder-Decoder')

    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument("--reference_mesh_file_voca", type=str,
                        default='./template/flame_model/FLAME_sample.ply',
                        help='path of the template')
    parser.add_argument("--epochs", type=int, default=300, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--result_dir", type=str, default='../Data/multiface/res/Results_Actor/Models')
    parser.add_argument("--sample_audio", type=str, default='../Data/VOCA/res/TH/photo.wav')

    # parser.add_argument("--template_file_voca", type=str, default="../datasets/VOCA_training/templates.pkl",
    #                     help='faces to animate')
    parser.add_argument("--template_file_multiface", type=str, default="../datasets/multiface/data/templates",
                        help='faces to animate')

    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--model_path", type=str,
                        default='../Data/multiface/res/Results_Actor/Models/ScanTalk_DiffusionNet_Encoder_Decoder_Hubert_MULTIFACE.pth.tar')

    parser.add_argument("--train_subjects", type=str, default="20171024 20180226 20180227 20180406 20180418 20180426 20180510 20180927 20190529 20190828")
    # parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
    #                                                           " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
    #                                                           " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
    #                                                           " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA"
    #                                                           " 20171024 20180226 20180227 20180406 20180418 20180426 20180510 20180927")
    # parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
    #                                                         " FaceTalk_170908_03277_TA 20190828 20190521")
    parser.add_argument("--val_subjects", type=str, default="20190828 20180105")
    # parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
    #                                                          " FaceTalk_170731_00024_TA 20181017")
    parser.add_argument("--test_subjects", type=str, default="20181017 20190521")
    # parser.add_argument("--wav_path_voca", type=str, default="../datasets/VOCA_training/wav",
    #                     help='path of the audio signals')
    # parser.add_argument("--vertices_path_voca", type=str, default="../datasets/VOCA_training/vertices_npy",
    #                     help='path of the ground truth')

    parser.add_argument("--wav_path_multiface", type=str, default="../datasets/multiface/data/wav",
                        help='path of the audio signals')
    parser.add_argument("--vertices_path_multiface", type=str, default="../datasets/multiface/data/vertices_npy",
                        help='path of the ground truth')

    parser.add_argument('--latent_channels', type=int, default=32)
    parser.add_argument('--audio_latent', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='multiface')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--k_eig', type=int, default=128)


    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()

