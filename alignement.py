import trimesh
from pytorch3d.structures import Meshes
import torch
import numpy as np
import os

ids = ['FaceTalk_170725_00137_TA', 'FaceTalk_170728_03272_TA', 'FaceTalk_170731_00024_TA', 'FaceTalk_170809_00138_TA',
       'FaceTalk_170811_03274_TA', 'FaceTalk_170811_03275_TA', 'FaceTalk_170904_00128_TA', 'FaceTalk_170904_03276_TA',
       'FaceTalk_170908_03277_TA', 'FaceTalk_170912_03278_TA', 'FaceTalk_170913_03279_TA', 'FaceTalk_170915_00223_TA']
sentences = [f'sentence{i:02}' for i in range(1,41)]


def align_scan(V_mesh, V_scan):
    nose_tip_index_reg = np.argmax(V_mesh,axis=0)[2]
    nose_tip_index_scan = np.argmax(V_scan,axis=0)[2]
    translation_vec = V_scan[nose_tip_index_scan] - V_mesh[nose_tip_index_reg]
    V_scan = V_scan - translation_vec
    return V_scan, translation_vec

dir_out = "/media/tbesnier/T5 EVO/datasets/Face/VOCA_SCANS_DS_ALIGNED"

for id in ids:

    ref_mesh = trimesh.load(f"/media/tbesnier/T5 EVO/scantalk/data/vocaset/templates/{id}.ply")

    for sent in sentences:

        verts = np.load(f"/media/tbesnier/T5 EVO/datasets/Face/VOCA_SCANS_DS/vertices_npy/{id}_{sent}.npy")
        faces = np.load(f"/media/tbesnier/T5 EVO/datasets/Face/VOCA_SCANS_DS/faces_npy/{id}_{sent}.npy")
        verts, faces = torch.tensor(verts).to("cuda:0"), torch.tensor(faces).to(dtype=torch.int32, device="cuda:0")
        verts = 0.001*verts
        meshes_sent = Meshes(verts=list(verts), faces=list(faces))
        L = []
        for i, frame in enumerate(meshes_sent.verts_list()):
            if i==0:
                # Mask
                mask = ~(meshes_sent.verts_list()[i] == 0).all(dim=1)
                # Use the mask to filter the rows
                filtered_tensor = meshes_sent.verts_list()[i][mask]

                mask_faces = ~(meshes_sent.faces_list()[i] == -1).all(dim=1)
                filtered_tensor_faces = meshes_sent.faces_list()[i][mask_faces]

                aligned_verts, trans_vec = align_scan(ref_mesh.vertices, filtered_tensor.detach().cpu().numpy())
                aligned_verts = torch.tensor(aligned_verts).to(device="cuda:0")
                L.append(aligned_verts)
                aligned_mesh = trimesh.Trimesh(aligned_verts.detach().cpu().numpy(),
                                               filtered_tensor_faces.detach().cpu().numpy())
                aligned_mesh.export(os.path.join(dir_out, "templates", id + ".ply"))
            else:
                # Mask
                mask = ~(meshes_sent.verts_list()[i] == 0).all(dim=1)
                # Use the mask to filter the rows
                filtered_tensor = meshes_sent.verts_list()[i][mask]

               # mask_faces = ~(meshes_sent.faces_list()[i] == -1).all(dim=1)
                #filtered_tensor_faces = meshes_sent.faces_list()[i][mask_faces]

                aligned_verts = filtered_tensor.detach().cpu().numpy() - trans_vec
                aligned_verts = torch.tensor(aligned_verts).to(device="cuda:0")
                L.append(aligned_verts)

        verts_padded = Meshes(verts=L, faces=list(faces)).verts_padded()
        verts, faces = verts_padded.cpu().detach().numpy(), meshes_sent.faces_padded().cpu().detach().numpy()
        verts, faces = verts.reshape((verts.shape[0], verts.shape[1], verts.shape[-1])), faces.reshape((faces.shape[0], faces.shape[1], faces.shape[-1]))

        np.save(os.path.join(dir_out, "vertices_npy", id + "_" + sent + ".npy"), verts)
        np.save(os.path.join(dir_out, "faces_npy", id + "_" + sent + ".npy"), faces)


