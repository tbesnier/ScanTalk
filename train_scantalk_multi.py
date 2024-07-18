import os
import trimesh
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import librosa
from transformers import Wav2Vec2Processor
from data_loader_diffusion_net_multi import get_dataloaders
import sys
from model.scantalk_hubert_new import DiffusionNetAutoencoder
from pytorch3d.loss import(
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing
)
from pytorch3d.structures import Meshes

sys.path.append('./model/diffusion-net/src')
import model.diffusion_net as diffusion_net


class Masked_Loss(nn.Module):
    def __init__(self, voca_mask, biwi_mask, multiface_mask):
        super(Masked_Loss, self).__init__()
        self.voca_mask = voca_mask
        self.biwi_mask = biwi_mask
        self.multiface_mask = multiface_mask
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, target, predictions, dataset_type):

        rec_loss = torch.mean(self.mse(predictions, target))

        if dataset_type == 'vocaset':
            mouth_loss = torch.mean(self.mse(predictions[:, self.voca_mask, :], target[:, self.voca_mask, :]))

        if dataset_type == 'BIWI':
            mouth_loss = torch.mean(self.mse(predictions[:, self.biwi_mask, :], target[:, self.biwi_mask, :]))

        if dataset_type == 'multiface':
            mouth_loss = torch.mean(self.mse(predictions[:, self.multiface_mask, :], target[:, self.multiface_mask, :]))

        prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_shift = target[:, 1:, :] - target[:, :-1, :]

        vel_loss = torch.mean((self.mse(prediction_shift, target_shift)))

        return rec_loss + 5 * mouth_loss + 10 * vel_loss

class Chamfer_Loss(nn.Module):
    def __init__(self, args):
        super(Chamfer_Loss, self).__init__()
        self.device = args.device
        self.torchdtype = torch.float

    def forward(self, predictions, targets):

        loss_chamfer, _ = chamfer_distance(predictions, targets)
        return loss_chamfer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args):
    device = args.device
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    reference_voca = trimesh.load(args.reference_mesh_file, process=False)
    template_tri_voca = reference_voca.faces
    template_vertices_voca = torch.tensor(reference_voca.vertices).to(args.device).float()

    model = DiffusionNetAutoencoder(args.in_channels, args.in_channels, args.latent_channels, args.device).to(args.device)

    print("model parameters: ", count_parameters(model))

    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")

    dataset = get_dataloaders(args)

    starting_epoch = 0
    if args.load_model == True:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['autoencoder_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        print(starting_epoch)

    criterion = Chamfer_Loss(args) #nn.MSELoss()
    criterion_val = nn.MSELoss()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(starting_epoch, args.epochs):
        model.train()
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
            vertices_pred = model.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY, faces, dataset_type)
            optim.zero_grad()
            loss = criterion(vertices, vertices_pred)
            loss.backward()
            optim.step()
            tloss += loss.item()
            pbar_talk.set_description(
                "(Epoch {}) TRAIN LOSS:{:.10f}".format((epoch + 1), tloss / (b + 1)))

        if epoch % 10 == 0:
            model.eval()
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
                    vertices_pred = model.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY, faces, dataset_type)
                    loss = criterion_val(vertices, vertices_pred)
                    t_test_loss += loss.item()
                    pbar_talk.set_description(
                        "(Epoch {}) VAL LOSS:{:.10f}".format((epoch + 1), (t_test_loss) / (b + 1)))

                # Sample from external audio and face Voca
                speech_array, sampling_rate = librosa.load(args.sample_audio, sr=16000)
                audio_feature = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)
                audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
                audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)

                # Compute Operators for DiffuserNet
                frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
                    torch.tensor(reference_voca.vertices).to('cpu'), faces=torch.tensor(template_tri_voca), k_eig=args.k_eig)
                mass = torch.FloatTensor(np.array(mass)).float().to(device).unsqueeze(0)
                evals = torch.FloatTensor(np.array(evals)).to(device).unsqueeze(0)
                evecs = torch.FloatTensor(np.array(evecs)).to(device).unsqueeze(0)
                L = L.float().to(device).unsqueeze(0)
                gradX = gradX.float().to(device).unsqueeze(0)
                gradY = gradY.float().to(device).unsqueeze(0)
                faces = torch.tensor(template_tri_voca).to(device).float().unsqueeze(0)

                gen_seq = model.predict(audio_feature, template_vertices_voca.to(device).unsqueeze(0), mass, L, evals,
                                        evecs, gradX, gradY, faces)

                gen_seq = gen_seq.cpu().detach().numpy()

                os.makedirs(
                    '../Data/VOCA/res/Results_Actor/Meshes' + str(
                        epoch), exist_ok=True)
                for m in range(len(gen_seq)):
                    mesh = trimesh.Trimesh(gen_seq[m], template_tri_voca)
                    mesh.export(
                        '../Data/VOCA/res/Results_Actor/Meshes' + str(
                            epoch) + '/frame_' + str(m).zfill(3) + '.ply')


        torch.save({'epoch': epoch,
                    'autoencoder_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    }, args.model_path)


def test(args):
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = DiffusionNetAutoencoder(args.in_channels, args.in_channels, args.latent_channels, args.device).to(args.device)

    dataset = get_dataloaders(args)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['autoencoder_state_dict'])

    pbar_talk = tqdm(enumerate(dataset["test"]), total=len(dataset["test"]))
    for b, sample in pbar_talk:
        audio = sample[0].to(args.device)
        vertices = sample[1].to(args.device).squeeze(0)
        template = sample[2].to(args.device)
        mass = sample[3].to(args.device)
        L = sample[4].to(args.device)
        evals = sample[5].to(args.device)
        evecs = sample[6].to(args.device)
        gradX = sample[7].to(args.device)
        gradY = sample[8].to(args.device)
        faces = sample[10].to(args.device)
        dataset_type = sample[11][0]
        vertices_pred = model.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY, faces, dataset_type)
        vertices = vertices.detach().cpu().numpy()
        vertices_pred = vertices_pred.detach().cpu().numpy()
        os.makedirs(f"../Data/scantalk_extension/{dataset_type}/TARGETS_npy", exist_ok=True)
        os.makedirs(f"../Data/scantalk_extension/{dataset_type}/PREDS_npy", exist_ok=True)
        np.save(f"../Data/scantalk_extension/{dataset_type}/TARGETS_npy/{sample[9][0][:-4]}.npy", vertices)
        np.save(f"../Data/scantalk_extension/{dataset_type}/PREDS_npy/{sample[9][0][:-4]}.npy", vertices_pred)


def infer(args):
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    reference_voca = trimesh.load(args.reference_mesh_file, process=False)
    template_tri_voca = reference_voca.faces
    template_vertices_voca = torch.tensor(reference_voca.vertices).to(args.device).float()

    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")
    model = DiffusionNetAutoencoder(args.in_channels, args.in_channels, args.latent_channels, args.device).to(args.device)

    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['autoencoder_state_dict'])

    # Sample from external audio and face Voca
    speech_array, sampling_rate = librosa.load(args.sample_audio, sr=16000)
    audio_feature = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)
    audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)

    # Compute Operators for DiffuserNet
    frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
        torch.tensor(reference_voca.vertices).to('cpu'), faces=torch.tensor(template_tri_voca), k_eig=args.k_eig)
    mass = torch.FloatTensor(np.array(mass)).float().to(args.device).unsqueeze(0)
    evals = torch.FloatTensor(np.array(evals)).to(args.device).unsqueeze(0)
    evecs = torch.FloatTensor(np.array(evecs)).to(args.device).unsqueeze(0)
    L = L.float().to(args.device).unsqueeze(0)
    gradX = gradX.float().to(args.device).unsqueeze(0)
    gradY = gradY.float().to(args.device).unsqueeze(0)
    faces = torch.tensor(template_tri_voca).to(args.device).float().unsqueeze(0)

    gen_seq = model.predict(audio_feature, template_vertices_voca.to(args.device).unsqueeze(0), mass, L, evals,
                            evecs, gradX, gradY, faces)

    gen_seq = gen_seq.cpu().detach().numpy()

    os.makedirs(
        '../Data/scantalk_extension/Meshes_infer', exist_ok=True)
    for m in range(len(gen_seq)):
        mesh = trimesh.Trimesh(gen_seq[m], template_tri_voca)
        mesh.export(
            '../Data/scantalk_extension/Meshes_infer' + '/frame_' + str(m).zfill(3) + '.ply')


def main():
    parser = argparse.ArgumentParser(description='Diffusion Net Multidataset: Dense to Dense Encoder-Decoder')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument("--reference_mesh_file", type=str,
                        default='./template/flame_model/FLAME_sample_remeshed.ply',
                        help='path of the template')
    parser.add_argument("--epochs", type=int, default=200, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--result_dir", type=str, default='../Data/VOCA/res/Results_Actor/Models/results')
    parser.add_argument("--sample_audio", type=str, default='../Data/VOCA/res/TH/photo.wav')


    parser.add_argument("--template_file_voca", type=str,
                        default="../datasets/VOCA_training/templates.pkl", help='faces to animate')
    parser.add_argument("--template_file_biwi", type=str, default="/media/tbesnier/T5 EVO/datasets/Face/BIWI/data/templates",
                        help='faces to animate')
    parser.add_argument("--template_file_multiface", type=str,
                        default="/media/tbesnier/T5 EVO/datasets/Face/multiface/Aligned_with_VOCA/templates", help='faces to animate')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
                                                              " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
                                                              " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
                                                              " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA F1 F2 F3 F4 F5 F6 M1 M2 M3 M4"
                                                              " 20171024 20180226 20180227 20180406 20180418 20180426 20180510 20180927 20190529")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
                                                            " M5 20180105")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
                                                             " FaceTalk_170731_00024_TA F8 M6 20181017 20190521")
    parser.add_argument("--wav_path_voca", type=str, default="../datasets/VOCA_training/wav",
                        help='path of the audio signals')
    parser.add_argument("--vertices_path_voca", type=str,
                        default="../datasets/VOCA_training/vertices_npy",
                        help='path of the ground truth')
    parser.add_argument("--wav_path_biwi", type=str, default="/media/tbesnier/T5 EVO/datasets/Face/BIWI/data/wav",
                        help='path of the audio signals')
    parser.add_argument("--vertices_path_biwi", type=str, default="/media/tbesnier/T5 EVO/datasets/Face/BIWI/data/vertices_npy",
                        help='path of the ground truth')
    parser.add_argument("--wav_path_multiface", type=str, default="/media/tbesnier/T5 EVO/datasets/Face/multiface/Aligned_with_VOCA/wav",
                        help='path of the audio signals')
    parser.add_argument("--vertices_path_multiface", type=str,
                        default="/media/tbesnier/T5 EVO/datasets/Face/multiface/Aligned_with_VOCA/vertices", help='path of the ground truth')


    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--model_path", type=str,
                        default='../Data/VOCA/res/Results_Actor/Models/results/ScanTalk_MULTI_new_CHAMFER.pth.tar')

    parser.add_argument("--info", type=str, default="", help='experiment info')

    ##Diffusion Net hyperparameters
    parser.add_argument('--latent_channels', type=int, default=32)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--k_eig', type=int, default=32)

    args = parser.parse_args()

    #train(args)
    #test(args)
    infer(args)


if __name__ == "__main__":
    main()
