import os, sys
import time

import trimesh
import numpy as np
import torch
import argparse
from tqdm import tqdm
from data_loader_diffusion_net_unsup import get_dataloaders
#from model.model_unregistered import DiffusionNetAutoencoder
from model.model_unregistered_hubert import DiffusionNetAutoencoder
#from model.model_unregistered_hubert_transformer import DiffusionNetAutoencoder
#from model.model_unregistered_wavlm import DiffusionNetAutoencoder
#from model.scantalk_hubert import DiffusionNetAutoencoder

sys.path.append('./model/diffusion-net/src')
#faces = torch.tensor(np.array(trimesh.load('../datasets/BIWI/data/templates/F1.obj').faces)).to(dtype=torch.int32)
#faces = torch.tensor(np.array(trimesh.load('../datasets/VOCA_training/templates/FaceTalk_170725_00137_TA.ply').faces)).to(dtype=torch.int32)
faces = torch.tensor(np.array(trimesh.load('../datasets/multiface/data/templates/20180227.ply').faces)).to(dtype=torch.int32)

def test(args):
    device = args.device
    dataset = get_dataloaders(args)

    d2d = DiffusionNetAutoencoder(args.in_channels, args.latent_channels, args.dataset, args.audio_latent).to(args.device)
    #d2d = DiffusionNetAutoencoder(args.in_channels, args.latent_channels, args.dataset, args.audio_latent).to(args.device)
    checkpoint = torch.load(args.model_path, map_location=device)
    d2d.load_state_dict(checkpoint['autoencoder_state_dict'])
    starting_epoch = checkpoint['epoch'] + 1
    print(starting_epoch)

    d2d.eval()
    inference_time = []
    with torch.no_grad():
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
            start = time.time()
            vertices_pred = d2d.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY, faces)
            delta_time = time.time() - start
            inference_time.append(delta_time/vertices.shape[0])
            vertices = vertices.detach().cpu().numpy()
            vertices_pred = vertices_pred.detach().cpu().numpy()
            print(sample[-2][0][:-4])
            os.makedirs("../results_multiface/results_scantalk_multiface_npy_new", exist_ok=True)
            np.save("../results_multiface/results_scantalk_multiface_npy_new/" + sample[-2][0][:-4], vertices_pred)

            #os.makedirs("../results_multiface/TARGETS_biwi_scantalk_hubert_npy", exist_ok=True)
            #np.save("../results_multiface/TARGETS_mu_scantalk_hubert_npy/" + sample[-2][0][:-4], vertices)

    print(f"Mean inference time: {np.array(inference_time).mean()}")
def main():
    dataset = 'multiface'
    if dataset=='vocaset':

        parser = argparse.ArgumentParser(description='D2D: Dense to Dense Encoder-Decoder')
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--result_dir", type=str, default='../Data/VOCA/res/Results_Actor/Models')
        parser.add_argument("--template_file", type=str, default="../datasets/VOCA_training/templates.pkl",
                            help='faces to animate')
        parser.add_argument("--load_model", type=bool, default=True)
        parser.add_argument("--model_path", type=str,
                            default='../Data/d2d_ScanTalk_DiffuserNet_Encoder_Decoder_Faces_MSE_Multidataset_VOCA_and_BIWI_and_Multiface_Hubert.pth.tar')
        parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
                                                                 " FaceTalk_170731_00024_TA")
        parser.add_argument("--wav_path", type=str, default="../datasets/VOCA_training/wav_test",
                            help='path of the audio signals')
        parser.add_argument("--vertices_path", type=str, default="../datasets/VOCA_training/vertices_npy_test",
                            help='path of the ground truth')

        parser.add_argument('--latent_channels', type=int, default=32)
        parser.add_argument('--audio_latent', type=int, default=16)
        parser.add_argument('--dataset', type=str, default='vocaset')
        parser.add_argument('--in_channels', type=int, default=3)

    if dataset=='multiface':

        parser = argparse.ArgumentParser(description='D2D: Dense to Dense Encoder-Decoder')
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--result_dir", type=str, default='../Data/VOCA_multiface/res/Results_Actor/Models')
        parser.add_argument("--template_file", type=str, default="../datasets/multiface/data/templates",
                            help='faces to animate')
        parser.add_argument("--load_model", type=bool, default=True)
        parser.add_argument("--model_path", type=str,
                            default='../Data/multiface/res/Results_Actor/Models/ScanTalk_DiffusionNet_Encoder_Decoder_Hubert_MULTIFACE.pth.tar')
        parser.add_argument("--test_subjects", type=str, default="20181017 20190521")
        parser.add_argument("--wav_path", type=str, default="../datasets/multiface/data/wav",
                            help='path of the audio signals')
        parser.add_argument("--vertices_path", type=str, default="../datasets/multiface/data/vertices_npy",
                            help='path of the ground truth')

        parser.add_argument('--latent_channels', type=int, default=32)
        parser.add_argument('--audio_latent', type=int, default=32)
        parser.add_argument('--dataset', type=str, default='multiface')
        parser.add_argument('--in_channels', type=int, default=3)

    if dataset=='BIWI':
        parser = argparse.ArgumentParser(description='D2D: Dense to Dense Encoder-Decoder')
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--result_dir", type=str, default='../Data/BIWI/res/Results_Actor/Models')
        parser.add_argument("--template_file", type=str, default="../datasets/BIWI/data/templates",
                            help='faces to animate')
        parser.add_argument("--load_model", type=bool, default=True)
        parser.add_argument("--model_path", type=str,
                            default='../Data/BIWI/res/Results_Actor/Models/ScanTalk_DiffusionNet_Encoder_Decoder_Hubert_BIG.pth.tar')
        parser.add_argument("--test_subjects", type=str, default="F8 M6")
        parser.add_argument("--wav_path", type=str, default="../datasets/BIWI/data/wav_test",
                            help='path of the audio signals')
        parser.add_argument("--vertices_path", type=str, default="../datasets/BIWI/data/vertices_npy_test",
                            help='path of the ground truth')

        parser.add_argument('--latent_channels', type=int, default=4)
        parser.add_argument('--audio_latent', type=int, default=16)
        parser.add_argument('--dataset', type=str, default='BIWI')
        parser.add_argument('--in_channels', type=int, default=3)

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    main()