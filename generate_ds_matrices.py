import pickle
from utils import utils, mesh_sampling
import shape_data
import os, shutil, glob
from psbody.mesh import Mesh
import trimesh
import numpy as np
from tqdm import tqdm

seq_length = [9, 9, 9, 9]
dilation = [1, 1, 1, 1]

def generate_ds_matrices(reference_mesh, seq_length=seq_length, dilation=dilation):

    device = "cuda:0"
    meshpackage = 'trimesh'

    shapedata = shape_data.ShapeData(nVal=100,
                                     reference_mesh_file=reference_mesh,
                                     normalization=False,
                                     meshpackage=meshpackage, load_flag=False)

    shapedata.n_vertex = 5023
    shapedata.n_features = 3

    # generate/load transform matrices
    transform_fp = './template/template/transform.pkl'
    template_fp = reference_mesh
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

        with open(transform_fp, 'wb') as fp:
            pickle.dump(tmp, fp)
        print('Done!')
        print('Transform matrices are saved in \'{}\''.format(transform_fp))
    else:
        with open(transform_fp, 'rb') as f:
            tmp = pickle.load(f, encoding='latin1')

    spiral_indices_list = [
        utils.preprocess_spiral(tmp['face'][idx], seq_length[idx],
                                tmp['vertices'][idx],
                                dilation[idx]).to(device)
        for idx in range(len(tmp['face']) - 1)
    ]
    down_transform_list = [
        utils.to_sparse(down_transform).to(device)
        for down_transform in tmp['down_transform']
    ]
    up_transform_list = [
        utils.to_sparse(up_transform).to(device)
        for up_transform in tmp['up_transform']
    ]

    dir_out = "../datasets/VOCA_ds"
    dir_data = "../datasets/VOCA"
    for f in glob.glob(dir_out + '/*'):
        shutil.rmtree(f)
    os.makedirs(dir_out, exist_ok=True)

    subjs = [f for f in os.listdir("../datasets/VOCA") if os.path.isdir(os.path.join("../datasets/VOCA", f))]
    for subjdir in subjs:
        os.makedirs(os.path.join(dir_out, subjdir), exist_ok=True)
        for sentence in os.listdir(os.path.join(dir_data, subjdir)):
            os.makedirs(os.path.join(dir_out, subjdir, sentence), exist_ok=True)
            sent, f, norm = [], [], []
            for i, mesh in tqdm(enumerate(os.listdir(os.path.join(dir_data, subjdir, sentence))),
                             f"Processing folder: {subjdir} {sentence}"):
                data_loaded = trimesh.load(os.path.join(dir_data, subjdir, sentence, mesh), process=False)
                vert = np.array(data_loaded.vertices)
                #print(os.path.join(dir_out, subjdir, sentence, sentence + '.' + str(i+1).zfill(6)) + '.ply')
                utils.export_mesh(tmp['down_transform'][0]*vert, tmp['face'][1],
                os.path.join(dir_out, subjdir, sentence, sentence + '.' + str(i+1).zfill(6)) + '.ply')


generate_ds_matrices(trimesh.load("./template/flame_model/FLAME_sample.ply"))

