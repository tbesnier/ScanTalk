import numpy as np
import os, shutil, glob
import trimesh
from tqdm import tqdm
from pytorch3d.structures import Meshes
import torch

def preprocess_VOCA(dir_data="../datasets/VOCA", dir_out="../Data/VOCA/preprocessed", normals=True, device="cuda:0"):

    subjs = [f for f in os.listdir(dir_data) if os.path.isdir(os.path.join(dir_data, f))]

    for f in glob.glob(dir_out + '/*'):
        shutil.rmtree(f)
    os.makedirs(dir_out, exist_ok=True)
    os.makedirs(os.path.join(dir_out, "vertices_npy"))
    os.makedirs(os.path.join(dir_out, "faces_npy"))
    if normals:
        os.makedirs(os.path.join(dir_out, "verts_normals_npy"))
    print(subjs)
    for subjdir in subjs:
        for sentence in os.listdir(os.path.join(dir_data, subjdir)):
            sent, f, norm = [], [], []
            for mesh in tqdm(os.listdir(os.path.join(dir_data, subjdir, sentence)), f"Processing folder: {subjdir} {sentence}"):
                data_loaded = trimesh.load(os.path.join(dir_data, subjdir, sentence, mesh), process=False)
                sent.append(torch.FloatTensor(np.array(data_loaded.vertices)).to(device))
                f.append(torch.IntTensor(np.array(data_loaded.faces)).to(device))
                if normals:
                    norm.append(torch.FloatTensor(np.array(data_loaded.vertex_normals)).to(device))
            print(len(sent))

            if normals:
                meshes_sent = Meshes(verts=sent, faces=f, verts_normals=norm)
                verts, faces = meshes_sent.verts_padded().cpu().detach().numpy(), meshes_sent.faces_padded().cpu().detach().numpy()
                verts_normals = meshes_sent.verts_normals_padded().cpu().detach().numpy
                np.save(os.path.join(dir_out, "verts_normals_npy", subjdir + "_" + sentence + ".npy"), verts_normals)
            else:
                meshes_sent = Meshes(verts=sent, faces=f)
                verts, faces = meshes_sent.verts_padded().cpu().detach().numpy(), meshes_sent.faces_padded().cpu().detach().numpy()

            np.save(os.path.join(dir_out, "vertices_npy", subjdir + "_" + sentence + ".npy"), verts)
            np.save(os.path.join(dir_out, "faces_npy", subjdir + "_" + sentence + ".npy"), faces)


def preprocess_VOCA_pad(dir_data="../datasets/VOCA", dir_out="../Data/VOCA/preprocessed", normals=True, device="cuda:0"):

    subjs = [f for f in os.listdir(dir_data) if os.path.isdir(os.path.join(dir_data, f))]

    for f in glob.glob(dir_out + '/*'):
        shutil.rmtree(f)
    os.makedirs(dir_out, exist_ok=True)
    os.makedirs(os.path.join(dir_out, "vertices_npy"))
    os.makedirs(os.path.join(dir_out, "faces_npy"))
    if normals:
        os.makedirs(os.path.join(dir_out, "verts_normals_npy"))
    print(subjs)
    sent, f, norm = [], [], []
    for subjdir in subjs:
        for sentence in os.listdir(os.path.join(dir_data, subjdir)):
            for mesh in tqdm(os.listdir(os.path.join(dir_data, subjdir, sentence)), f"Processing folder: {subjdir} {sentence}"):
                data_loaded = trimesh.load(os.path.join(dir_data, subjdir, sentence, mesh), process=False)
                sent.append(torch.FloatTensor(np.array(data_loaded.vertices)).to(device))
                f.append(torch.IntTensor(np.array(data_loaded.faces)).to(device))
                if normals:
                    norm.append(torch.FloatTensor(np.array(data_loaded.vertex_normals)).to(device))
            print(len(sent))

            if normals:
                meshes_sent = Meshes(verts=sent, faces=f, verts_normals=norm)
                verts, faces = meshes_sent.verts_padded().cpu().detach().numpy(), meshes_sent.faces_padded().cpu().detach().numpy()
                verts_normals = meshes_sent.verts_normals_padded().cpu().detach().numpy
            else:
                meshes_sent = Meshes(verts=sent, faces=f)
                verts, faces = meshes_sent.verts_padded().cpu().detach().numpy(), meshes_sent.faces_padded().cpu().detach().numpy()



np.save(os.path.join(dir_out, "vertices_npy", subjdir + "_" + sentence + ".npy"), verts)
np.save(os.path.join(dir_out, "faces_npy", subjdir + "_" + sentence + ".npy"), faces)

if __name__ == "__main__":
    preprocess_VOCA()