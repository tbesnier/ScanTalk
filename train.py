import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from data_loader import get_dataloaders
from model.scantalk import DiffusionNetAutoencoder


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args):
    device = args.device
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    d2d = DiffusionNetAutoencoder(args).to(args.device)

    print("model parameters: ", count_parameters(d2d))

    dataset = get_dataloaders(args)

    criterion = nn.MSELoss()
    criterion_val = nn.MSELoss()

    optim = torch.optim.Adam(d2d.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
            vertices_pred = d2d.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY, faces)
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
                    mass = sample[3].to(device)
                    L = sample[4].to(device)
                    evals = sample[5].to(device)
                    evecs = sample[6].to(device)
                    gradX = sample[7].to(device)
                    gradY = sample[8].to(device)
                    faces = sample[10].to(device)
                    dataset_type = sample[11][0]
                    vertices_pred = d2d.forward(audio, template, vertices, mass, L, evals, evecs, gradX, gradY, faces)
                    loss = criterion_val(vertices, vertices_pred)
                    t_test_loss += loss.item()
                    pbar_talk.set_description(
                        "(Epoch {}) VAL LOSS:{:.10f}".format((epoch + 1), (t_test_loss) / (b + 1)))

        torch.save({'epoch': epoch,
                    'autoencoder_state_dict': d2d.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    }, os.path.join(args.result_dir,
                                    './pretrained_model/model.pth.tar'))


def main():
    parser = argparse.ArgumentParser(description='Diffusion Net Multidataset: Dense to Dense Encoder-Decoder')

    # Learning specs
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument("--epochs", type=int, default=200, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--result_dir", type=str, default='./results')

    # Data specs
    parser.add_argument("--template_file_voca", type=str,
                        default="./data/vocaset/templates.pkl", help='faces to animate')
    parser.add_argument("--template_file_biwi", type=str, default="./data/BIWI_6/templates",
                        help='faces to animate')
    parser.add_argument("--template_file_multiface", type=str,
                        default="./data/multiface/templates", help='faces to animate')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
                                                              " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
                                                              " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
                                                              " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA F1 F2 F3 F4 F5 F6 M1 M2 M3 M4"
                                                              " 20171024 20180226 20180227 20180406 20180418 20180426 20180510 20180927 20190529")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
                                                            " FaceTalk_170908_03277_TA F7 M5 20190828 20180105")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
                                                             " FaceTalk_170731_00024_TA F8 M6 20181017 20190521")
    parser.add_argument("--wav_path_voca", type=str, default="./data/vocaset/wav",
                        help='path of the audio signals')
    parser.add_argument("--vertices_path_voca", type=str,
                        default="./data/vocaset/vertices_npy", help='path of the ground truth')
    parser.add_argument("--wav_path_biwi", type=str, default="./data/BIWI_6/wav",
                        help='path of the audio signals')
    parser.add_argument("--vertices_path_biwi", type=str, default="./data/BIWI_6/vertices",
                        help='path of the ground truth')
    parser.add_argument("--wav_path_multiface", type=str, default="./data/multiface/wav",
                        help='path of the audio signals')
    parser.add_argument("--vertices_path_multiface", type=str,
                        default="./data/multiface/vertices", help='path of the ground truth')

    # Diffusion Net hyperparameters
    parser.add_argument('--latent_channels', type=int, default=32)
    parser.add_argument('--in_channels', type=int, default=3)

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
