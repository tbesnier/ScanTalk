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
from sklearn.metrics.pairwise import euclidean_distances
import argparse
import time
from tqdm import tqdm
from d2d import SpiralAutoencoder
from transformers import Wav2Vec2Processor
import librosa
from wav2vec import Wav2Vec2Model


def train(args):
    filter_sizes_enc = [[3, 64, 64, 64, 128], [[], [], [], [], []]]
    filter_sizes_dec = [[128, 64, 32, 32, 16], [[], [], [], [], 3]]
    ds_factors = [4, 4, 4, 4]
    reference_points = [[3567, 4051, 4597]]
    latent_size = 32
    step_sizes = [2, 2, 1, 1, 1]
    dilation = [2, 2, 1, 1, 1]
    device = args.device
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
        
    audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
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
    


    with open('/home/federico/Scrivania/D2D/template/template/downsampling_matrices.pkl', 'rb') as fp:
        downsampling_matrices = pickle.load(fp)

    M_verts_faces = downsampling_matrices['M_verts_faces']
    M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process=False) for i in range(len(M_verts_faces))]

    A = downsampling_matrices['A']
    D = downsampling_matrices['D']
    U = downsampling_matrices['U']
    F = downsampling_matrices['F']

    for i in range(len(ds_factors)):
        dist = euclidean_distances(M[i + 1].vertices, M[0].vertices[reference_points[0]])
        reference_points.append(np.argmin(dist, axis=0).tolist())

    Adj, Trigs = spiral_utils.get_adj_trigs(A, F, shapedata.reference_mesh, meshpackage='trimesh')

    spirals_np, spiral_sizes, spirals = spiral_utils.generate_spirals(step_sizes,
                                                            M, Adj, Trigs,
                                                            reference_points = reference_points,
                                                            dilation = dilation, random = False,
                                                            meshpackage = 'trimesh',
                                                            counter_clockwise = True)

    spirals_np[0] = spirals_np[0][:, :-1, :]
    sizes = [x.vertices.shape[0] for x in M]


    tspirals = [torch.from_numpy(s).long().to(device) for s in spirals_np]

    bU = []
    bD = []
    for i in range(len(D)):
        d = np.zeros((1, D[i].shape[0] + 1, D[i].shape[1] + 1))
        u = np.zeros((1, U[i].shape[0] + 1, U[i].shape[1] + 1))
        d[0, :-1, :-1] = D[i].todense()
        u[0, :-1, :-1] = U[i].todense()
        d[0, -1, -1] = 1
        u[0, -1, -1] = 1
        bD.append(d)
        bU.append(u)


    tD = [torch.from_numpy(s).float().to(device) for s in bD]
    tU = [torch.from_numpy(s).float().to(device) for s in bU]

    tD[0] = tD[0][:, :, 1:]
    tU[0] = tU[0][:, 1:, :]
    
    dataset_train = new_data_loader.TH_Dataset(args.dataset_dir_audios, 
                                               args.dataset_dir_frame, 
                                               args.dataset_dir_next_frame, 
                                               args.dataset_dir_actor,
                                               0,
                                               100000)
    
    dataset_test = new_data_loader.TH_Dataset(args.dataset_dir_audios,
                                              args.dataset_dir_frame, 
                                              args.dataset_dir_next_frame, 
                                              args.dataset_dir_actor,
                                              100000,
                                              121000)


    dataloader_train = DataLoader(dataset_train, batch_size=128,
                                 shuffle=True, num_workers=16)
    
    dataloader_test = DataLoader(dataset_test, batch_size=128,
                                 shuffle=True, num_workers=16)
    

    d2d = SpiralAutoencoder(filters_enc=filter_sizes_enc,
                                      filters_dec=filter_sizes_dec,
                                      latent_size=latent_size,
                                      sizes=sizes,
                                      spiral_sizes=spiral_sizes,
                                      spirals=tspirals,
                                      D=tD, U=tU, device=device).to(device)
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
            audio = sample['audio'].to(device)
            frame = sample['frame'].to(device)
            next_frame = sample['next_frame'].to(device)
            actor = sample['actor'].to(device)
            next_frame_pred = d2d.forward(audio, frame, actor)
            loss = criterion(next_frame, next_frame_pred)
            loss.backward()
            optim.step()
            tloss += loss
            pbar_talk.set_description(
                "(Epoch {}) TRAIN LOSS:{:.10f}".format((epoch + 1), tloss/(b+1)))
        
        if epoch % 10 == 0:
            d2d.eval()
            with torch.no_grad():
                t_test_loss = 0
                pbar_talk = tqdm(enumerate(dataloader_test), total=len(dataloader_test))
                for b, sample in pbar_talk:
                    audio = sample['audio'].to(device)
                    frame = sample['frame'].to(device)
                    next_frame = sample['next_frame'].to(device)
                    actor = sample['actor'].to(device)
                    next_frame_pred = d2d.forward(audio, frame, actor)
                    loss = criterion(next_frame, next_frame_pred)
                    t_test_loss += loss
                    pbar_talk.set_description(
                                    "(Epoch {}) TEST LOSS:{:.10f}".format((epoch + 1), (t_test_loss)/(b+1)))
            
            speech_array, sampling_rate = librosa.load(args.sample_audio, sr=16000)
            audio_feature = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)
            audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
            audio_feature = torch.FloatTensor(audio_feature)
            hidden_states = audio_encoder(audio_feature).last_hidden_state.to(args.device)
            
            gen_seq = d2d.predict(hidden_states, template_vertices.float())
            
            gen_seq = gen_seq.cpu().detach().numpy()
            
            os.mkdir('/home/federico/Scrivania/ScanTalk/ScanTalk/Results_Actor/Meshes/' + str(epoch))
            for m in range(len(gen_seq)):
                mesh = trimesh.Trimesh(gen_seq[m], template_tri)
                mesh.export('/home/federico/Scrivania/ScanTalk/ScanTalk/Results_Actor/Meshes/' + str(epoch) + '/frame_' + str(m).zfill(3) + '.ply')
                
            torch.save({'epoch': epoch,
                        'autoencoder_state_dict': d2d.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        }, os.path.join(args.result_dir, 'd2d_ScanTalk_new_training_strat_disp_with_Actor.pth.tar'))
            
            

def main():
    parser = argparse.ArgumentParser(description='D2D: Dense to Dense Encoder-Decoder')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--reference_mesh_file", type=str, default='/home/federico/Scrivania/D2D/template/flame_model/FLAME_sample.ply', help='path of the template')
    parser.add_argument("--epochs", type=int, default=300, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset_dir_audios", type=str, default='/home/federico/Scrivania/ScanTalk/ScanTalk/Consecutive_Dataset/Audio_Snippet')
    parser.add_argument("--dataset_dir_frame", type=str, default='/home/federico/Scrivania/ScanTalk/ScanTalk/Consecutive_Dataset/Frame')
    parser.add_argument("--dataset_dir_actor", type=str, default='/home/federico/Scrivania/ScanTalk/ScanTalk/Consecutive_Dataset/Actor')
    parser.add_argument("--dataset_dir_next_frame", type=str, default='/home/federico/Scrivania/ScanTalk/ScanTalk/Consecutive_Dataset/Next_Frame')
    parser.add_argument("--result_dir", type=str, default='/home/federico/Scrivania/ScanTalk/ScanTalk/Results_Actor/Models')
    parser.add_argument("--sample_audio", type=str, default='/home/federico/Scrivania/TH/photo.wav')
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--model_path", type=str, default='/home/federico/Scrivania/ScanTalk/ScanTalk/Results_Actor/Models/d2d_ScanTalk_new_training_strat_disp.pth.tar')

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
