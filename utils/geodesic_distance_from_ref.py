import numpy as np
import trimesh
import potpourri3d as pp3d
import polyscope as ps

colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
    'cornsilk', 'crimson', 'cyan', 'darkblue',
    'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
    'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
    'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
    'darkslateblue', 'darkslategray', 'darkslategrey',
    'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
    'dimgray', 'dimgrey', 'dodgerblue', '#000000'
]

mesh_test = trimesh.load("../datasets/VOCA_training/templates/FaceTalk_170809_00138_TA.ply")
V_test, F_test = np.array(mesh_test.vertices), np.array(mesh_test.faces)
nose_tip_index = np.argmax(V_test, axis=0)[2]

solver = pp3d.MeshHeatMethodDistanceSolver(V_test, F_test)

dist = solver.compute_distance(nose_tip_index)
heatmap = dist

def compute_mask_mouth_region(mesh):
    V, F = np.array(mesh.vertices), np.array(mesh.faces)
    nose_tip_index = np.argmax(V_test, axis=0)[2]
    radius = 1/2 * (V.max() - V.min())

    #idx_mouth_region = np.where(np.linalg.norm(V - V[nose_tip_index], axis=1)<radius)[0]
    idx_mouth_region = np.where(np.linalg.norm(V - V[nose_tip_index], axis=1) < radius)[0]
    idx_y_neg = np.where(V[:, 1]<0)[0]

    mask_mouth_region = np.array([1 if i in idx_mouth_region and i in idx_y_neg else 0 for i in range(V.shape[0])])

    return mask_mouth_region

mask = compute_mask_mouth_region(mesh_test).T * heatmap
print(mask.min(), mask.max())

def register_surface(name, mesh, x=0.0, y=0.0, z=0.0, idx_color=0, transparency=1.0, disp_vectors=None, disp_heatmap=None):
    vertices, faces = np.array(mesh.vertices), np.array(mesh.faces)
    vertices = vertices + np.stack((x*np.ones((vertices.shape[0],1)), np.zeros((vertices.shape[0],1)), np.zeros((vertices.shape[0],1))), axis=1)[:,:,0]
    vertices = vertices + np.stack((np.zeros((vertices.shape[0],1)), y*np.ones((vertices.shape[0],1)), np.zeros((vertices.shape[0],1))), axis=1)[:,:,0]
    vertices = vertices + np.stack((np.zeros((vertices.shape[0],1)), np.zeros((vertices.shape[0],1)), z*np.ones((vertices.shape[0],1))), axis=1)[:,:,0]

    mesh = ps.register_surface_mesh(name, vertices, faces)
    mesh.set_color(tuple(int(colors[idx_color][i:i + 2], 16) / 255.0 for i in (1, 3, 5)))
    mesh.set_smooth_shade(False)
    mesh.set_transparency(transparency)

    if disp_vectors is not None:
        mesh.add_vector_quantity("displacement vectors", 5*disp_vectors, enabled=True,
                                 color=tuple(int(colors[-1][i:i + 2], 16) / 255.0 for i in (1, 3, 5)), vectortype="ambient")

    if disp_heatmap is not None:
        min_bound, max_bound = disp_heatmap.min(), disp_heatmap.max()  #
        mesh.add_scalar_quantity('relative error heatmap', disp_heatmap, enabled=True, cmap='reds', vminmax=(min_bound, max_bound))

    return mesh

ps.init()
ps.set_up_dir("y_up")
ps.set_ground_plane_mode("shadow_only")
ps.set_ground_plane_height_factor(0)

heatmap = mask
register_surface(name=f'Test', x=0., y=-0., z=0., idx_color=1, mesh=mesh_test,
                         disp_vectors = None, disp_heatmap=heatmap)

ps.show()

