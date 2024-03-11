import sys
import librosa
import torch
import trimesh as tri
from transformers import Wav2Vec2Processor

import generate_video
from model.scantalk import DiffusionNetAutoencoder
from generate_video import *

#sys.path.append('./model/diffusion-net/src')
import model.diffusion_net as diffusion_net


def infer(args):
    os.makedirs(args.result_dir, exist_ok=True)
    for f in glob.glob(args.result_dir + "/*"):
        os.remove(f)

    reference = tri.load(args.reference_mesh_file, process=False)
    template_tri = reference.faces
    template_vertices = torch.FloatTensor(np.array(reference.vertices))
    template_vertices = template_vertices.to(device=args.device)

    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")

    model = DiffusionNetAutoencoder(args).to(args.device)

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
        mass = torch.FloatTensor(np.array(mass)).float().to(args.device).unsqueeze(0)
        evals = torch.FloatTensor(np.array(evals)).to(args.device).unsqueeze(0)
        evecs = torch.FloatTensor(np.array(evecs)).to(args.device).unsqueeze(0)
        L = L.float().to(args.device).unsqueeze(0)
        gradX = gradX.float().to(args.device).unsqueeze(0)
        gradY = gradY.float().to(args.device).unsqueeze(0)
        faces = torch.tensor(template_tri).to(args.device).float().unsqueeze(0)

        gen_seq = model.predict(audio_feature, template_vertices.to(args.device).unsqueeze(0), mass, L, evals, evecs,
                              gradX, gradY, faces)

        gen_seq = gen_seq.cpu().detach().numpy()
        os.makedirs(args.result_dir, exist_ok=True)
        for m in range(len(gen_seq)):
            mesh = tri.Trimesh(gen_seq[m], template_tri)
            mesh.export(args.result_dir + '/frame_' + str(m).zfill(3) + '.ply')


def main():
    parser = argparse.ArgumentParser(description='Infer ScanTalk on a mesh file with an audio file and visualize what happens during inference')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--result_dir", type=str, default='../demo/meshes')
    parser.add_argument("--reference_mesh_file", type=str, default="./demo/demo_mesh.ply", help='path of the template') # tested on .ply or .obj file
    parser.add_argument("--sample_audio", type=str, default='./demo/demo_audio.wav') # tested on .wav file
    parser.add_argument("--model_path", type=str, default='../pretrained_model/ScanTalk.pth.tar')

    parser.add_argument('--latent_channels', type=int, default=32)
    parser.add_argument('--in_channels', type=int, default=3)

    args = parser.parse_args()

    print("Generate the mesh sequence...")
    infer(args)
    print("Done !")
    generate_video.main(audio_path="./demo/demo_audio.wav",
                        meshes_path="../demo/meshes",
                        save_path="./demo",
                        name="demo.mp4", fps=50)


if __name__ == "__main__":
    main()
