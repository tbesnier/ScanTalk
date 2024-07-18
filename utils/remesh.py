import pymeshlab
import os
import numpy as np
import trimesh
import open3d as o3d

ids = ['FaceTalk_170725_00137_TA', 'FaceTalk_170728_03272_TA', 'FaceTalk_170731_00024_TA', 'FaceTalk_170809_00138_TA',
       'FaceTalk_170811_03274_TA', 'FaceTalk_170811_03275_TA', 'FaceTalk_170904_00128_TA', 'FaceTalk_170904_03276_TA',
       'FaceTalk_170908_03277_TA', 'FaceTalk_170912_03278_TA', 'FaceTalk_170913_03279_TA', 'FaceTalk_170915_00223_TA']

sentences = [f'sentence{i:02}' for i in range(1,41)]
print(sentences)

base_path = "/media/tbesnier/T5 EVO/datasets/Face/VOCA_training/vertices_npy"
output_path = "/media/tbesnier/T5 EVO/datasets/Face/VOCA_remeshed"

faces = np.array(trimesh.load("/media/tbesnier/T5 EVO/datasets/Face/VOCA/FaceTalk_170725_00137_TA/sentence01/sentence01.000001.ply").faces)

os.makedirs(output_path, exist_ok=True)


def getMeshFromData(mesh, Rho=None, color=None):
    """
    Convert Data into mesh object
    """
    V = mesh[0]
    F = mesh[1]

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))

    if Rho is not None:
        Rho = np.squeeze(Rho)
        col = np.stack((Rho, Rho, Rho))
        mesh.vertex_colors = o3d.utility.Vector3dVector(col.T)

    if color is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(color)
    return mesh
def decimate_mesh(V, F, target):
    """
    Decimates mesh given by V,F to have number of faces approximately equal to target
    """
    mesh = getMeshFromData([V, F])
    mesh = mesh.simplify_quadric_decimation(target)
    VS = np.asarray(mesh.vertices, dtype=np.float64)  # get vertices of the mesh as a numpy array
    FS = np.asarray(mesh.triangles, np.int64)  # get faces of the mesh as a numpy array
    return VS, FS


def subdivide_mesh(V, F, Rho=None, order=1):
    """
    Performs midpoint subdivision. Order determines the number of iterations
    """
    mesh = getMeshFromData([V, F], Rho=Rho)
    mesh = mesh.subdivide_midpoint(number_of_iterations=order)
    VS = np.asarray(mesh.vertices, dtype=np.float64)  # get vertices of the mesh as a numpy array
    FS = np.asarray(mesh.triangles, np.int64)  # get faces of the mesh as a numpy array
    if Rho is not None:
        RhoS = np.asarray(mesh.vertex_colors, np.float64)[:, 0]
        return VS, FS, RhoS

    return VS, FS


# for id in ids:
#        for sentence in sentences:
#               i = 1
#               verts = np.load(f"{base_path}/{id}_{sentence}.npy")
#               length_sentence = verts.shape[0]#len(os.listdir(base_path + "/" + id + "/" + sentence))
#               while i <= length_sentence:
#                      if not os.path.exists(output_path + "/" + id + "/" + sentence + "/" + str(sentence) + f'.{i:06}' + ".ply"):# and os.path.exists(base_path + "/" + id + "/" + sentence + "/" + str(sentence) + f'.{i:06}' + ".ply"):
#                             V = verts[i-1].reshape((5023, 3))
#                             new_V, new_F = subdivide_mesh(V, faces, order=1)
#                             new_V, new_F = decimate_mesh(new_V, new_F, target=5000)
#
#                             # ms = pymeshlab.MeshSet()
#                             # ms.add_mesh()
#                             # ms.load_new_mesh(base_path + "/" + id + "/" + sentence + "/" + str(sentence) + f'.{i:06}' + ".ply")
#                             # #ms.load_filter_script('./sub_simp.mlx')
#                             # ms.apply_filter('meshing_surface_subdivision_butterfly')
#                             # ms.meshing_decimation_quadric_edge_collapse(targetfacenum=5000)
#
#                             if not os.path.exists(output_path + "/" + id):
#                                    os.mkdir(output_path + "/" + id)
#                             if not os.path.exists(output_path + "/" + id + "/" + sentence):
#                                    os.mkdir(output_path + "/" + id + "/" + sentence)
#
#                             mesh = trimesh.Trimesh(new_V, new_F)
#                             mesh.export(
#                                    output_path + "/" + id + "/" + sentence + "/" + str(sentence) + f'.{i:06}' + ".ply")
#                      if i==1 and not os.path.exists(output_path + "/templates/" + id + ".ply"):
#                          mesh.export(output_path + "/templates/" + id + ".ply")
#                      i += 1

process_plys=True
if process_plys:
       import torch
       from pytorch3d.structures import Meshes
       import numpy as np
       output_path = "/media/tbesnier/T5 EVO/datasets/Face/VOCA_remeshed_npy"
       output_path_faces= "/media/tbesnier/T5 EVO/datasets/Face/VOCA_remeshed_npy_faces"
       os.makedirs(output_path, exist_ok=True)
       os.makedirs(output_path_faces, exist_ok=True)
       for id in ids:
              for sentence in sentences:
                     dir = os.path.join("/media/tbesnier/T5 EVO/datasets/Face/VOCA_remeshed", id, sentence)
                     L, L_faces = [], []
                     for frame in os.listdir(dir):
                            mesh = trimesh.load(os.path.join(dir, frame))
                            vertices = torch.tensor(np.array(mesh.vertices)).to(device="cuda:0", dtype=torch.float)
                            faces = torch.tensor(np.array(mesh.faces)).to(device="cuda:0", dtype=torch.int32)
                            L.append(vertices)
                            L_faces.append(faces)
                     meshes = Meshes(verts=L, faces=L_faces)
                     verts = meshes.verts_padded().cpu().detach().numpy()
                     faces = meshes.faces_padded().cpu().detach().numpy()
                     np.save(os.path.join(output_path, f"{id}_{sentence}.npy"), verts)
                     np.save(os.path.join(output_path_faces, f"{id}_{sentence}.npy"), faces)


clean_dir = False
if clean_dir:
       dir_path = "/media/tbesnier/T5 EVO/datasets/Face/VOCA_SCANS/"
       subj = os.listdir(dir_path)
       for i, sent in enumerate(subj):
              sent_dir = os.path.join(dir_path, sent)
              for j, elt in enumerate(os.listdir(sent_dir)):
                     os.makedirs(f"/media/tbesnier/T5 EVO/datasets/Face/VOCA_SCANS_TEXTURE/{sent}/{elt}", exist_ok=True)
                     for k, file in enumerate(os.listdir(os.path.join(sent_dir, elt))):
                            file_name = os.path.join(sent_dir, elt, file)
                            if file_name.endswith(".png"):
                                   os.remove(file_name)
                            elif file_name.endswith(".mtl"):
                                   os.rename(file_name, f"/media/tbesnier/T5 EVO/datasets/Face/VOCA_SCANS_TEXTURE/{sent}/{elt}/{file_name[-21:]}")