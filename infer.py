import numpy as np
import argparse
import os, glob, shutil
import spiral_utils
import shape_data
import librosa
import torch
import trimesh as tri
import pickle

from transformers import Wav2Vec2Processor
from wav2vec import Wav2Vec2Model

from model.model_semi_registered import PointNet2SpiralsAutoEncoder, PointNet2NJFAutoEncoder
from psbody.mesh import Mesh
from utils import utils, mesh_sampling

def infer(args):

    os.makedirs(args.result_dir, exist_ok=True)
    for f in glob.glob(args.result_dir + "/*"):
        os.remove(f)

    audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    shapedata = shape_data.ShapeData(nVal=100,
                                     reference_mesh_file=args.reference_mesh_file,
                                     normalization=False,
                                     meshpackage='trimesh', load_flag=False)

    shapedata.n_vertex = 5023
    shapedata.n_features = 3

    reference = tri.load(args.reference_mesh_file, process=False)
    template_tri = reference.faces
    template_vertices = torch.FloatTensor(reference.vertices).unsqueeze(0)
    template_vertices = template_vertices.to(device=args.device)

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
                                args.dilation[idx]).to(args.device)
        for idx in range(len(tmp['face']) - 1)
    ]
    down_transform_list = [
        utils.to_sparse(down_transform).to(args.device)
        for down_transform in tmp['down_transform']
    ]
    up_transform_list = [
        utils.to_sparse(up_transform).to(args.device)
        for up_transform in tmp['up_transform']
    ]

    #model = SpiralAutoencoder(args.in_channels, args.out_channels, args.latent_channels,
     #                 spiral_indices_list, down_transform_list,
     #                 up_transform_list).to(args.device)

    #model = PointNet2SpiralsAutoEncoder(args.latent_channels, args.in_channels, args.out_channels,
    #                            spiral_indices_list, down_transform_list, up_transform_list).to(args.device)

    model = PointNet2NJFAutoEncoder(latent_channels=args.latent_channels, point_dim=3, device=args.device).to(args.device)

    checkpoint_dict = torch.load(os.path.join(args.model_path), map_location=args.device)
    model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])

    model.eval()
    with torch.no_grad():
        # Sample from external audio and face
        speech_array, sampling_rate = librosa.load(args.sample_audio, sr=16000)
        audio_feature = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)
        audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
        audio_feature = torch.FloatTensor(audio_feature).to(args.device)
        #hidden_states = audio_encoder(audio_feature).last_hidden_state.to(args.device)

        #gen_seq = model.predict_new(hidden_states, template_vertices)
        gen_seq = model.predict(audio_feature, template_vertices.float())
        gen_seq = gen_seq.cpu().detach().numpy()

        for m in range(len(gen_seq)):
            mesh = tri.Trimesh(gen_seq[m], template_tri)
            mesh.export('../Data/VOCA/res/Results_Actor/Meshes_infer/' + '/frame_' + str(m).zfill(3) + '.ply')
            #if m<len(gen_seq) - 1:
            #    displacements = gen_seq[m+1] - gen_seq[m]
            #    np.save(file=args.result_dir + '/frame_' + str(m).zfill(3) + '.npy', arr=displacements)
            #else:
            #    np.save(file=args.result_dir + '/frame_' + str(m).zfill(3) + '.npy', arr=np.zeros((displacements.shape)))


def main():
    parser = argparse.ArgumentParser(description='Infer ScanTalk on a mesh file with an audio file and visualize what happens during inference')
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--result_dir", type=str, default='../Data/VOCA/res/Results_Actor/Meshes_infer/')
    parser.add_argument("--reference_mesh_file", type=str, default='./template/flame_model/FLAME_sample.ply',
                        help='path of the template')
    parser.add_argument("--sample_audio", type=str, default='../Data/VOCA/res/TH/short_audio.wav')
    parser.add_argument("--template_file", type=str, default="../datasets/VOCA_training/templates.pkl",
                        help='faces to animate')
    parser.add_argument("--model_path", type=str, default='../Data/VOCA/res/Results_Actor/Models/d2d_ScanTalk_bigger_lstm_masked_velocity_loss.pth.tar')

    ##Spiral++ hyperparameters
    parser.add_argument('--out_channels',
                        nargs='+',
                        default=[32, 64, 64, 128],
                        type=int)
    parser.add_argument('--latent_channels', type=int, default=32)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--seq_length', type=int, default=[9, 9, 9, 9], nargs='+')
    parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')

    args = parser.parse_args()

    infer(args)

if __name__ == "__main__":
    main()