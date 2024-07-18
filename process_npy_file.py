import os
from tqdm import tqdm
import numpy as np
import trimesh

template_BIWI = trimesh.load("/media/tbesnier/T5 EVO/datasets/Face/BIWI/data/templates/F1.obj")
template_BIWI_original = trimesh.load("../CodeTalker/BIWI/BIWI.ply")

template_VOCA = trimesh.load("./template/flame_model/FLAME_sample.ply")

template_multiface = trimesh.load("/media/tbesnier/T5 EVO/datasets/Face/multiface/Aligned_with_VOCA/templates/20171024.ply")

dataset="multiface"


def export_mesh(V, F, file_name):
    """
    Export mesh as .ply file from vertices coordinates and face connectivity
    """
    result = trimesh.exchange.ply.export_ply(trimesh.Trimesh(V, F), encoding='ascii')
    output_file = open(file_name, "wb+")
    output_file.write(result)
    output_file.close()


def export_mesh_obj(V, F, file_name):
    """
    Export mesh as .obj file from vertices coordinates and face connectivity
    """
    result = trimesh.exchange.obj.export_obj(trimesh.Trimesh(V, F))
    output_file = open(file_name, "w")
    output_file.write(result)
    output_file.close()


def align_scan(V_ref, V_toalign):
    nose_tip_index_reg = np.argmax(V_ref,axis=0)[2]
    nose_tip_index_scan = np.argmax(V_toalign,axis=0)[2]
    translation_vec = V_toalign[nose_tip_index_scan] - V_ref[nose_tip_index_reg]
    V = V_toalign - translation_vec
    return V


def normalize_mesh(V):
    center = np.mean(V, 0)
    V = V - center
    scale = np.max(np.absolute(V),0).max()
    return V/scale


process=False
if process:
    vertices_npy_path = "../Data/scantalk_extension/multiface/TARGETS_npy" #f"/home/tbesnier/phd/projects/Data/VOCA/PREDS_npy"  #"../results_multiface/results_scantalk_multiface_npy" #"../TARGETS_biwi_scantalk_hubert_test_npy"
    for i, elt in tqdm(enumerate(os.listdir(vertices_npy_path))):
        sent = np.load(os.path.join(vertices_npy_path, elt))
        print(sent.shape)
        if sent.shape[-1]==16413:
            sent = sent.reshape((sent.shape[0], 5471, 3))
        if sent.shape[-1] == 11685:
            sent = sent.reshape((sent.shape[0], 3895, 3))
        os.makedirs(f"/home/tbesnier/phd/projects/Data/scantalk_extension/multiface/TARGETS_meshes/" + elt[:-4], exist_ok=True)  #"/home/tbesnier/phd/projects/Data/VOCA/PREDS_meshes/"
        for k in range(sent.shape[0]):
            export_mesh(V=sent[k], F=template_multiface.faces, file_name=f"/home/tbesnier/phd/projects/Data/scantalk_extension/multiface/TARGETS_meshes/" + elt[:-4] + '/frame_' + str(k+1).zfill(3) + '.ply')
        print(elt, sent.shape)

reshape_npy=False
if reshape_npy:
    vertices_npy_path = "/media/tbesnier/T5 EVO/datasets/Face/VOCA_remeshed_npy"
    faces_npy_path = "/media/tbesnier/T5 EVO/datasets/Face/VOCA_remeshed_npy_faces"
    for i, elt in tqdm(enumerate(os.listdir(vertices_npy_path))):
        if "FaceTalk_170725_00137_TA" in elt or "FaceTalk_170904_00128_TA" in elt:
            sent = np.load(os.path.join(vertices_npy_path, elt))
            sent_new = sent.reshape((sent.shape[0], sent.shape[-2] * sent.shape[-1]))
            np.save(f"/media/tbesnier/T5 EVO/datasets/Face/VOCA_remeshed_npy_{1}/{elt}", sent_new)

            faces = np.load(os.path.join(faces_npy_path, elt))
            faces_new = faces.reshape((faces.shape[0], faces.shape[-2] * faces.shape[-1]))
            np.save(f"/media/tbesnier/T5 EVO/datasets/Face/VOCA_remeshed_npy_faces_{1}/{elt}", faces_new)


check_mesh = True
if check_mesh:
    from pytorch3d.structures import Meshes
    import torch
    verts = np.load("/media/tbesnier/T5 EVO/datasets/Face/VOCA_SCANS_DS/vertices_npy/FaceTalk_170809_00138_TA_sentence10.npy")
    faces = np.load("/media/tbesnier/T5 EVO/datasets/Face/VOCA_SCANS_DS/faces_npy/FaceTalk_170809_00138_TA_sentence10.npy")
    verts, faces = torch.tensor(verts).to("cuda:0"), torch.tensor(faces).to(dtype=torch.int32, device="cuda:0")
    meshes_sent = Meshes(verts=list(verts), faces=list(faces))

    i = 100
    # Mask
    mask = ~(meshes_sent.verts_list()[i] == 0).all(dim=1)
    # Use the mask to filter the rows
    filtered_tensor = meshes_sent.verts_list()[i][mask]

    mask_faces = ~(meshes_sent.faces_list()[i] == -1).all(dim=1)
    # Use the mask to filter the rows
    filtered_tensor_faces = meshes_sent.faces_list()[i][mask_faces]

    mesh = trimesh.Trimesh(vertices=filtered_tensor.detach().cpu().numpy(), faces=filtered_tensor_faces.detach().cpu().numpy())
    mesh.export("../check.ply")


