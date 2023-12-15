import trimesh
import numpy as np
import os
import sys
import torch
sys.path.append('./model/diffusion-net/src')
import model.diffusion_net as diffusion_net

base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, "data", "op_cache")
print(op_cache_dir)

vertices_path = '../datasets/VOCA_training/vertices_npy'
faces = trimesh.load('./template/flame_model/FLAME_sample.ply', process=False).faces

vertices_list = []
faces_list = []

for sequence in os.listdir(vertices_path):
    seq = np.load(os.path.join(vertices_path, sequence))
    for i in range(seq.shape[0]):
        vertices_list.append(torch.tensor(np.reshape(seq[i], (5023, 3))))
        faces_list.append(torch.tensor(faces))

frames_list, massvec_list, L_list, evals_list, evecs_list, gradX_list, gradY_list = diffusion_net.geometry.get_all_operators(vertices_list, faces_list=None, k_eig=128, op_cache_dir=op_cache_dir)

print('done')