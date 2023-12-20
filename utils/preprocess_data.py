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


def preprocess_VOCA_pad(dir_data="../datasets/VOCA_remeshed_test", dir_out="../Data/VOCA/preprocessed_padded", normals=False, device="cuda:0"):

    ### TO DO ###

    subjs = [f for f in os.listdir(dir_data) if os.path.isdir(os.path.join(dir_data, f))]
    subjs.sort()

    for f in glob.glob(dir_out + '/*'):
        shutil.rmtree(f)
    os.makedirs(dir_out, exist_ok=True)
    os.makedirs(os.path.join(dir_out, "vertices_npy"))
    os.makedirs(os.path.join(dir_out, "faces_npy"))
    if normals:
        os.makedirs(os.path.join(dir_out, "verts_normals_npy"))
    print(subjs)
    sent, f, norm = [], [], []
    T = []  ## list of sentences length
    cpt = 0
    for subjdir in subjs:
        L = os.listdir(os.path.join(dir_data, subjdir))
        L.sort()
        for sentence in L:
            if cpt==0:
                T.append([subjdir, sentence, len(os.listdir(os.path.join(dir_data, subjdir, sentence)))])
            else:
                T.append([subjdir, sentence, T[-1][-1] + len(os.listdir(os.path.join(dir_data, subjdir, sentence)))])
            for mesh in tqdm(os.listdir(os.path.join(dir_data, subjdir, sentence)), f"Processing folder: {subjdir} {sentence}"):
                data_loaded = trimesh.load(os.path.join(dir_data, subjdir, sentence, mesh), process=False)
                sent.append(torch.FloatTensor(np.array(data_loaded.vertices)).to(device))
                f.append(torch.IntTensor(np.array(data_loaded.faces)).to(device))
                if normals:
                    norm.append(torch.FloatTensor(np.array(data_loaded.vertex_normals)).to(device))
            cpt+=1
            print(len(sent))

    if normals:
        meshes_sent = Meshes(verts=sent, faces=f, verts_normals=norm)
        verts, faces = meshes_sent.verts_padded().cpu().detach().numpy(), meshes_sent.faces_padded().cpu().detach().numpy()
        verts_normals = meshes_sent.verts_normals_padded().cpu().detach().numpy
    else:
        meshes_sent = Meshes(verts=sent, faces=f)
        verts, faces = meshes_sent.verts_padded().cpu().detach().numpy(), meshes_sent.faces_padded().cpu().detach().numpy()
        print(verts.shape)
        print(faces.shape)

    for i in range(len(T)):
        if 0 < i <= len(T):

            np.save(os.path.join(dir_out, "vertices_npy", T[i][0] + "_" + T[i][1] + ".npy"), verts[T[i-1][-1]:T[i][-1]])
            np.save(os.path.join(dir_out, "faces_npy", T[i][0] + "_" + T[i][1] + ".npy"),
                    faces[T[i - 1][-1]:T[i][-1]])
        elif i==0:
            print(T[i][-1])
            np.save(os.path.join(dir_out, "vertices_npy", T[i][0] + "_" + T[i][1] + ".npy"),
                    verts[:T[i][-1]])
            np.save(os.path.join(dir_out, "faces_npy", T[i][0] + "_" + T[i][1] + ".npy"),
                    faces[:T[i][-1]])


if __name__ == "__main__":
    preprocess_VOCA_pad()