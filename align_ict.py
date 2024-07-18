import numpy as np
import trimesh
import pymeshlab

def export_mesh(V, F, file_name):

    result = trimesh.exchange.ply.export_ply(trimesh.Trimesh(V,F), encoding='ascii')
    output_file = open(file_name, "wb+")
    output_file.write(result)
    output_file.close()

def normalize_mesh(V):
    center = np.mean(V, 0)
    V = V-center
    scale= np.max(np.absolute(V), 0).max()
    return V/scale


mesh_path = "../ICT-FaceKit/sample_data_out/random_identity00.obj"

mesh = trimesh.load(mesh_path)
ms = pymeshlab.MeshSet()
ms.load_new_mesh(mesh_path)
V, F = ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()

mask = np.array([1 if i <= 6705 else 0 for i in range(0, 26719)])
print(mask.shape)
print(F.shape)

m1 = pymeshlab.Mesh(
    vertex_matrix=V,
    face_matrix= F,
    v_scalar_array=mask
)
ms.add_mesh(m1, 'masked_mesh')
ms.compute_selection_by_scalar_per_vertex(minq=-0.1, maxq=0.1)
ms.meshing_remove_selected_vertices()
V_new, F_new = ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()


V, F = 0.009*V_new + np.array([0, -0.01, -0.05]), F_new
export_mesh(V=V, F=F, file_name="../test_narrow_face.ply")