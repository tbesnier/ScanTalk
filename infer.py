import numpy as np
import argparse
import os, glob, sys
import librosa
import torch
import trimesh as tri
from aitviewer.renderables.meshes import Meshes
import trimesh
from aitviewer.viewer import Viewer
from transformers import Wav2Vec2Processor
from model.scantalk_hubert import DiffusionNetAutoencoder

sys.path.append('./model/diffusion-net/src')
import model.diffusion_net as diffusion_net

def infer(args):
    os.makedirs(args.result_dir, exist_ok=True)
    for f in glob.glob(args.result_dir + "/*"):
        os.remove(f)

    reference = tri.load(args.reference_mesh_file, process=False)
    template_tri = reference.faces
    template_vertices = torch.FloatTensor(np.array(reference.vertices) + np.array([0, 0, 0]))
    template_vertices = template_vertices.to(device=args.device)

    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")

    model = DiffusionNetAutoencoder(args.in_channels, args.in_channels, args.latent_channels, args.device).to(args.device)

    checkpoint_dict = torch.load(os.path.join(args.model_path), map_location=args.device)
    model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])

    model.eval()
    with torch.no_grad():
        # Sample from external audio and face
        speech_array, sampling_rate = librosa.load(args.sample_audio, sr=16000)
        audio_feature = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)
        audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
        audio_feature = torch.FloatTensor(audio_feature).to(args.device)

        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
            template_vertices.to('cpu'), faces=torch.tensor(template_tri), k_eig=128)
        #np.save("../Data/VOCA/res/eig.npy", evecs.detach().cpu().numpy())
        mass = torch.FloatTensor(np.array(mass)).float().to(args.device).unsqueeze(0)
        evals = torch.FloatTensor(np.array(evals)).to(args.device).unsqueeze(0)
        evecs = torch.FloatTensor(np.array(evecs)).to(args.device).unsqueeze(0)
        L = L.float().to(args.device).unsqueeze(0)
        gradX = gradX.float().to(args.device).unsqueeze(0)
        gradY = gradY.float().to(args.device).unsqueeze(0)
        faces = torch.tensor(template_tri).to(args.device).float().unsqueeze(0)

        gen_seq = model.predict(audio_feature, template_vertices.to(args.device).unsqueeze(0), mass, L, evals, evecs,
                              gradX, gradY, faces)#, dataset='vocaset')
        gen_seq = gen_seq.cpu().detach().numpy()

        latent = model.get_latent_features(audio_feature, template_vertices.to(args.device).unsqueeze(0), mass, L, evals, evecs,
                              gradX, gradY, faces)

        np.save("../Data/scantalk_extension/latent_trajectories/VOCA_remeshed.npy", latent.detach().cpu().numpy())

        for m in range(len(gen_seq)):
            mesh = tri.Trimesh(gen_seq[m], template_tri)
            mesh.export(args.result_dir + '/frame_' + str(m).zfill(3) + '.ply')


def render(args):
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        'burlywood', 'cadetblue',
        'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue',
        'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
        'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey',
        'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', '#000000'
    ]

    meshes_dir = '../Data/scantalk_extension/inference_exp' #'../Data/LISC/results_dual_MANO/MANO_anim'
    mesh_list = [os.path.join(meshes_dir, os.listdir(meshes_dir)[i]) for i in range(len(os.listdir(meshes_dir)))]
    mesh_list.sort()

    # Number of frames.
    N = len(mesh_list)

    vertices = np.array([trimesh.load(mesh_list[i]).vertices for i in range(N)])

    seq = Meshes(
        vertices,
        trimesh.load(mesh_list[0]).faces,
        name="Prediction",
        position=[0, 0, 0],
        flat_shading=False
    )

    viewer = Viewer()
    viewer.scene.add(seq)

    viewer.auto_set_camera_target = True
    seq.norm_coloring = True

    viewer.scene.origin.enabled = False
    viewer.playback_fps = 30

    viewer.run_animations = True
    viewer.run()


def main():
    parser = argparse.ArgumentParser(description='Infer ScanTalk on a mesh file with an audio file and visualize what happens during inference')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--dataset', type=str, default='vocaset')
    parser.add_argument("--result_dir", type=str, default='../Data/scantalk_extension/inference_exp')#f'../../papers/ScanTalk/res_ICT/meshes/meshes_{str(i).zfill(2)}')
    parser.add_argument("--reference_mesh_file", type=str, default="../datasets/VOCA_training/templates/FaceTalk_170809_00138_TA_remeshed_UpDown.ply") #"../datasets/VOCA_training/templates/FaceTalk_170809_00138_TA_remeshed_UpDown.ply", #"../datasets/VOCA_training/templates/FaceTalk_170809_00138_TA.ply", #"../datasets/COMA_scans/COMA_FaceTalk_170809_00138_TA.ply", #"../datasets/other/arnold_aligned.ply",#f'../datasets/ICT/head_and_neck/{str(i).zfill(2)}.ply', #'../datasets/VOCA_training/templates/FaceTalk_170725_00137_TA.ply' ,#'../datasets/ICT/head_and_neck/01.ply',  #'../datasets/ICT/head_and_neck/00.ply' ,#'../datasets/VOCA_training/templates/FaceTalk_170809_00138_TA.ply' ,#'../datasets/multiface/data/templates/20171024.ply', #'../Data/VOCA/res/Results_Actor/Meshes_Training/10/frame_000.ply' ,#'../datasets/ICT/narrow_face_area/33.ply',#'./template/flame_model/FLAME_sample.ply',
    parser.add_argument("--sample_audio", type=str, default='../Data/VOCA/res/TH/photo.wav') # '../datasets/VOCA_training/wav_test/FaceTalk_170809_00138_TA_sentence35.wav')#'../Data/VOCA/res/TH/photo.wav')  #FaceTalk_170809_00138_TA_sentence40.wav
    parser.add_argument("--model_path", type=str,
                        default='../Data/d2d_ScanTalk_DiffuserNet_Encoder_Decoder_Faces_MSE_Multidataset_VOCA_and_BIWI_and_Multiface_Hubert.pth.tar')#'../Data/d2d_ScanTalk_DiffuserNet_Encoder_Decoder_Faces_MSE_Multidataset_VOCA_and_BIWI_and_Multiface_Hubert.pth.tar')  #ScanTalk_DiffuserNet_Encoder_Decoder_Faces_MSE_Multidataset_VOCA_and_BIWI_and_Multiface.pth.tar')  #'ScanTalk_DiffusionNet_Encoder_Decoder300epochsMSE_VOCA_wav2vec.pth.tar')
    parser.add_argument('--latent_channels', type=int, default=32)
    parser.add_argument('--in_channels', type=int, default=3)

    args = parser.parse_args()

    infer(args)
    render(args)


if __name__ == "__main__":
    main()


