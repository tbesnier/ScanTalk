import os, sys
import shape_data
import pickle
import trimesh
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import librosa
from transformers import Wav2Vec2Processor, AutoProcessor
from data_loader_diffusion_net import get_dataloaders
import lddmm_utils
#from model.model_unregistered import DiffusionNetAutoencoder
from model.scantalk_hubert import DiffusionNetAutoencoder
#from model.model_unregistered_wavlm import DiffusionNetAutoencoder
#from pytorch3d.structures import Meshes

sys.path.append('./model/diffusion-net/src')
import model.diffusion_net as diffusion_net

#faces = torch.tensor(np.array(trimesh.load('./template/flame_model/FLAME_sample.ply').faces)).to(dtype=torch.int32)
faces = torch.tensor(np.array(trimesh.load('../datasets/BIWI/data/templates/F1.obj').faces)).to(dtype=torch.int32)

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

        #prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
        #target_shift = target[:, 1:, :] - target[:, :-1, :]

        #vel_loss = torch.mean((self.mse(prediction_shift, target_shift)))

        return rec_loss# + vel_loss


def train(args):
    device = args.device
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if args.dataset == 'vocaset':
        reference = trimesh.load(args.reference_mesh_file, process=False)
        template_tri = reference.faces
        template_vertices = torch.tensor(reference.vertices).to(args.device).float()

    reference = trimesh.load(args.reference_mesh_file, process=False)
    template_tri = reference.faces
    template_vertices = torch.tensor(reference.vertices).to(args.device).float()

    dataset = get_dataloaders(args)

    d2d = DiffusionNetAutoencoder(args.in_channels, args.latent_channels, args.dataset, args.audio_latent).to(args.device)

    #processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")
    #processor = AutoProcessor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")

    starting_epoch = 0
    if args.load_model == True:
        checkpoint = torch.load(args.model_path, map_location=device)
        d2d.load_state_dict(checkpoint['autoencoder_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        print(starting_epoch)

    #lip_mask = scipy.io.loadmat('./FLAME_lips_idx.mat')
    #lip_mask = lip_mask['lips_idx'] - 1
    #lip_mask = np.reshape(np.array(lip_mask, dtype=np.int64), (lip_mask.shape[0]))

    criterion = Masked_Loss(args)
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
            #torch.nn.utils.clip_grad_norm_(d2d.parameters(), 10.0)
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
                    template_vertices.to('cpu'), faces=torch.tensor(template_tri), k_eig=args.k_eig)
                mass = torch.FloatTensor(np.array(mass)).float().to(device).unsqueeze(0)
                evals = torch.FloatTensor(np.array(evals)).to(device).unsqueeze(0)
                evecs = torch.FloatTensor(np.array(evecs)).to(device).unsqueeze(0)
                L = L.float().to(device).unsqueeze(0)
                gradX = gradX.float().to(device).unsqueeze(0)
                gradY = gradY.float().to(device).unsqueeze(0)
                faces = torch.tensor(template_tri).to(device).float().unsqueeze(0)

                gen_seq = d2d.predict(audio_feature, template_vertices.to(device).unsqueeze(0), mass, L, evals, evecs,
                                      gradX, gradY, faces)

                gen_seq = gen_seq.cpu().detach().numpy()

                os.makedirs('../Data/VOCA/res/Results_Actor/Meshes/' + str(epoch),
                            exist_ok=True)
                for m in range(len(gen_seq)):
                    mesh = trimesh.Trimesh(gen_seq[m], template_tri)
                    mesh.export('../Data/VOCA/res/Results_Actor/Meshes/' + str(
                        epoch) + '/frame_' + str(m).zfill(3) + '.ply')

                # Sample from training set
                # Process the audio
                speech_array, sampling_rate = librosa.load(args.training_sample_audio, sr=16000)
                audio_feature = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)
                audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
                audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)
                # audio_feature = audio_encoder(audio_feature, frame_num=len(vertices)).last_hidden_state.to(device=args.device)
                if args.dataset == 'vocaset':
                    with open(args.template_file, 'rb') as fin:
                        templates = pickle.load(fin, encoding='latin1')
                        face = templates[args.training_sample_face]   ### for vocaset
                if args.dataset=='BIWI':
                    face = trimesh.load(args.training_sample_face, process=False).vertices   ### For BIWI

                face = torch.tensor(face).to(args.device).float()

                # Compute Operators for DiffuserNet
                frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(face.to('cpu'),
                                                                                                       faces=torch.tensor(
                                                                                                           template_tri),
                                                                                                       k_eig=args.k_eig)
                mass = torch.FloatTensor(np.array(mass)).float().to(device).unsqueeze(0)
                evals = torch.FloatTensor(np.array(evals)).to(device).unsqueeze(0)
                evecs = torch.FloatTensor(np.array(evecs)).to(device).unsqueeze(0)
                L = L.float().to(device).unsqueeze(0)
                gradX = gradX.float().to(device).unsqueeze(0)
                gradY = gradY.float().to(device).unsqueeze(0)
                faces = torch.tensor(template_tri).to(device).float().unsqueeze(0)

                gen_seq = d2d.predict(audio_feature, face.to(device).unsqueeze(0), mass, L, evals, evecs, gradX, gradY,
                                      faces)

                gen_seq = gen_seq.cpu().detach().numpy()

                os.makedirs(
                    '../Data/VOCA/res/Results_Actor/Meshes_Training/' + str(epoch),
                    exist_ok=True)
                for m in range(len(gen_seq)):
                    mesh = trimesh.Trimesh(gen_seq[m], template_tri)
                    mesh.export('../Data/VOCA/res/Results_Actor/Meshes_Training/' + str(
                        epoch) + '/frame_' + str(m).zfill(3) + '.ply')

                # pbar_talk = tqdm(enumerate(dataset["test"]), total=len(dataset["test"]))
                # for b, sample in pbar_talk:
                #     audio = sample[0].to(device)
                #     vertices = sample[1].to(device).squeeze(0)
                #     template = sample[2].to(device)
                #     mass = sample[3].to(device)
                #     L = sample[4].to(device)
                #     evals = sample[5].to(device)
                #     evecs = sample[6].to(device)
                #     gradX = sample[7].to(device)
                #     gradY = sample[8].to(device)
                #     vertices_pred = d2d.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY, faces)
                    #vertices = vertices.detach().cpu().numpy()
                    #vertices_pred = vertices_pred.detach().cpu().numpy()

                    # for k in range(vertices_pred.shape[0]):
                    #     error_lve += ((vertices_pred[k] - vertices[k]) ** 2)[lip_mask].max()
                    #     error_recons += ((vertices_pred[k] - vertices[k]) ** 2).max()
                    #     count += 1
                    # pbar_talk.set_description(" LVE:{:.8f}".format((error_lve) / (count)))

        torch.save({'epoch': epoch,
                    'autoencoder_state_dict': d2d.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    }, os.path.join(args.result_dir, 'ScanTalk_DiffusionNet_Encoder_Decoder_Hubert_NO_LSTM.pth.tar'))


def main():
    parser = argparse.ArgumentParser(description='D2D: Dense to Dense Encoder-Decoder')
    dataset='vocaset'
    if dataset=='vocaset':
        parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument("--reference_mesh_file", type=str,
                            default='./template/flame_model/FLAME_sample.ply',
                            help='path of the template')
        parser.add_argument("--epochs", type=int, default=200, help='number of epochs')
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--result_dir", type=str, default='../Data/VOCA/res/Results_Actor/Models')

        parser.add_argument("--sample_audio", type=str, default='../Data/VOCA/res/TH/photo.wav')
        parser.add_argument("--training_sample_audio", type=str,
                            default='../datasets/VOCA/data/wav/F1_01.wav')
        parser.add_argument("--template_file", type=str, default="../datasets/VOCA_training/templates.pkl",
                            help='faces to animate')
        parser.add_argument("--training_sample_face", type=str, default="FaceTalk_170725_00137_TA", help='face to animate')
        parser.add_argument("--load_model", type=bool, default=False)
        parser.add_argument("--model_path", type=str,
                            default='../Data/VOCA/res/Results_Actor/Models/ScanTalk_DiffusionNet_Encoder_Decoder_Hubert.pth.tar')
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
        parser.add_argument("--vertices_path", type=str, default="../datasets/VOCA_training/vertices_npy",
                            help='path of the ground truth')

        parser.add_argument('--latent_channels', type=int, default=4)
        parser.add_argument('--audio_latent', type=int, default=16)
        parser.add_argument('--dataset', type=str, default='vocaset')
        parser.add_argument('--in_channels', type=int, default=3)


    if dataset=='biwi':
        parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument("--reference_mesh_file", type=str,
                            default='../datasets/BIWI_original/data/templates/F1.obj',
                            help='path of the template')
        parser.add_argument("--epochs", type=int, default=300, help='number of epochs')
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--result_dir", type=str, default='../Data/BIWI_original/res/Results_Actor/Models')
        parser.add_argument("--sample_audio", type=str, default='../Data/VOCA/res/TH/photo.wav')
        parser.add_argument("--training_sample_audio", type=str,
                            default='../datasets/BIWI_original/data/wav/F1_01.wav')
        parser.add_argument("--template_file", type=str, default="../datasets/BIWI_original/data/templates",
                            help='faces to animate')
        parser.add_argument("--training_sample_face", type=str, default="../datasets/BIWI_original/data/templates/F1.obj",
                            help='face to animate')
        parser.add_argument("--load_model", type=bool, default=False)
        parser.add_argument("--model_path", type=str,
                            default='../Data/BIWI/res/Results_Actor/Models/ScanTalk_DiffusionNet_Encoder_Decoder_Hubert.pth.tar')
        #parser.add_argument("--mask_path", type=str,
        #                    default='../datasets/BIWI/data/mouth_indices.npy')
        parser.add_argument("--train_subjects", type=str, default="F1 F2 F3 F4 F5 F6 M1 M2 M3 M4")
        parser.add_argument("--val_subjects", type=str, default="F7 M5")
        parser.add_argument("--test_subjects", type=str, default="F8 M6")
        parser.add_argument("--wav_path", type=str, default="../datasets/BIWI_original/data/wav",
                            help='path of the audio signals')
        parser.add_argument("--vertices_path", type=str, default="../datasets/BIWI_original/data/vertices_npy",
                            help='path of the ground truth')

        parser.add_argument('--latent_channels', type=int, default=4)
        parser.add_argument('--audio_latent', type=int, default=8)
        parser.add_argument('--dataset', type=str, default='BIWI')
        parser.add_argument('--in_channels', type=int, default=3)
        parser.add_argument('--k_eig', type=int, default=32)

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()