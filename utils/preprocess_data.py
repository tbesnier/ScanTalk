import numpy as np
import os, shutil, glob
import trimesh
from tqdm import tqdm
from pytorch3d.structures import Meshes
import torch
import pymeshlab


def preprocess_VOCA_pad(dir_data="/media/tbesnier/T5 EVO/datasets/Face/VOCA_SCANS", dir_out="/media/tbesnier/T5 EVO/datasets/Face/VOCA_SCANS", normals=False, device="cuda:0"):

    ### TO DO ###

    subjs = [f for f in os.listdir(dir_data) if os.path.isdir(os.path.join(dir_data, f))]
    subjs.sort()

    #for f in glob.glob(dir_out + '/*'):
    #    shutil.rmtree(f)
    os.makedirs(dir_out, exist_ok=True)
    os.makedirs(os.path.join(dir_out, "vertices_npy"), exist_ok=True)
    os.makedirs(os.path.join(dir_out, "faces_npy"), exist_ok=True)
    print(subjs)

    for subjdir in subjs:
        L = os.listdir(os.path.join(dir_data, subjdir))
        L.sort()
        for sentence in L:
            if not os.path.exists(os.path.join(dir_out, "vertices_npy", subjdir + "_" + sentence + ".npy")):
                sent, f = [], []
                frames = os.listdir(os.path.join(dir_data, subjdir, sentence))
                frames.sort()
                for mesh in tqdm(frames, f"Processing folder: {subjdir} {sentence}"):
                    #data_loaded = trimesh.load(os.path.join(dir_data, subjdir, sentence, mesh), process=False)
                    ms = pymeshlab.MeshSet()
                    ms.load_new_mesh(os.path.join(dir_data, subjdir, sentence, mesh))
                    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=8000)
                    print(np.array(ms.current_mesh().vertex_matrix()).shape)
                    sent.append(torch.FloatTensor(np.array(ms.current_mesh().vertex_matrix())).to(device))
                    f.append(torch.IntTensor(np.array(ms.current_mesh().face_matrix())).to(device))

                print(len(sent))
                meshes_sent = Meshes(verts=sent, faces=f)
                verts, faces = meshes_sent.verts_padded().cpu().detach().numpy(), meshes_sent.faces_padded().cpu().detach().numpy()
                print(verts.shape)
                print(faces.shape)
                np.save(os.path.join(dir_out, "vertices_npy", subjdir + "_" + sentence + ".npy"), verts)
                np.save(os.path.join(dir_out, "faces_npy", subjdir + "_" + sentence + ".npy"), faces)


if __name__ == "__main__":
    preprocess_VOCA_pad(dir_data="/media/tbesnier/T5 EVO/datasets/Face/VOCA_SCANS", dir_out="/media/tbesnier/T5 EVO/datasets/Face/VOCA_SCANS_DS")

