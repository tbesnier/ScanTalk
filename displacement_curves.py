import polyscope as ps
import polyscope.imgui as psim
import trimesh as tri
import numpy as np
import os

ui_int = 0
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

def register_surface(name, mesh, x=0.0, y=0.0, z=0.0, idx_color=0, transparency=1.0, disp_vectors=None, disp_heatmap=None):
    vertices, faces = np.array(mesh.vertices), np.array(mesh.faces)
    vertices = vertices + np.stack((x*np.ones((vertices.shape[0],1)), np.zeros((vertices.shape[0],1)), np.zeros((vertices.shape[0],1))), axis=1)[:,:,0]
    vertices = vertices + np.stack((np.zeros((vertices.shape[0],1)), y*np.ones((vertices.shape[0],1)), np.zeros((vertices.shape[0],1))), axis=1)[:,:,0]
    vertices = vertices + np.stack((np.zeros((vertices.shape[0],1)), np.zeros((vertices.shape[0],1)), z*np.ones((vertices.shape[0],1))), axis=1)[:,:,0]

    mesh = ps.register_surface_mesh(name, vertices, faces, edge_width=0.8)
    mesh.set_color(tuple(int(colors[idx_color][i:i + 2], 16) / 255.0 for i in (1, 3, 5)))
    mesh.set_smooth_shade(False)
    mesh.set_transparency(transparency)

    if disp_vectors is not None:
        mesh.add_vector_quantity("displacement vectors", 5*disp_vectors, enabled=True,
                                 color=tuple(int(colors[-1][i:i + 2], 16) / 255.0 for i in (1, 3, 5)), vectortype="ambient")

    if disp_heatmap is not None:
        min_bound, max_bound = disp_heatmap.min(), disp_heatmap.max() #0, 0.01 #0, 0.0085  # disp_heatmap.min(), disp_heatmap.max()
        mesh.add_scalar_quantity('relative error heatmap', disp_heatmap, enabled=True, cmap='pink-green', vminmax=(min_bound, max_bound))

    return mesh


def register_curve(name, curve_nodes, curve_edges, x=0.0, y=0.0, z=0.0, idx_color=1, transparency=1.0):

    curve_nodes = curve_nodes + np.stack((x*np.ones((curve_nodes.shape[0],1)), np.zeros((curve_nodes.shape[0],1)), np.zeros((curve_nodes.shape[0],1))), axis=1)[:,:,0]
    curve_nodes = curve_nodes + np.stack((np.zeros((curve_nodes.shape[0],1)), y*np.ones((curve_nodes.shape[0],1)), np.zeros((curve_nodes.shape[0],1))), axis=1)[:,:,0]
    curve_nodes = curve_nodes + np.stack((np.zeros((curve_nodes.shape[0],1)), np.zeros((curve_nodes.shape[0],1)), z*np.ones((curve_nodes.shape[0],1))), axis=1)[:,:,0]

    curve = ps.register_curve_network(name, curve_nodes, curve_edges, radius=0.0005)
    curve.set_color(tuple(int(colors[idx_color][i:i + 2], 16) / 255.0 for i in (1, 3, 5)))
    curve.set_transparency(transparency)

    return curve


def callback():

    global ui_int, meshes, meshes_reparam

    # Note that it is a push/pop pair, with the matching pop() below.
    psim.PushItemWidth(150)

    # == Show text in the UI

    psim.TextUnformatted("Sequence of meshes")
    psim.TextUnformatted("Sequence length: {}".format(len(meshes)))
    psim.Separator()

    # Input Int Slider
    changed, ui_int = psim.SliderInt("Frame", ui_int, v_min=0, v_max=len(meshes)-2)
    if changed:
        ps.remove_all_structures()
        register_surface(name=f'Frame {ui_int} Ours', z=ui_int*0.01, mesh=meshes[ui_int])
        indices = [2846, 3531, 1723, 3509]
        for j, idx in enumerate(indices):
            c1_p = np.array([mesh.vertices[idx] + 0.01 * i * np.array([0, 0, 1]) for i, mesh in enumerate(meshes)])
            c1_e = np.array([[i, i + 1] for i in range(len(c1_p) - 1)])

            register_curve(name=f"Curve {idx}", curve_nodes=c1_p, curve_edges=c1_e, idx_color= j+1)

        indices = [1143, 1588, 660, 4648]
        register_surface(name=f'Frame {ui_int} Reparam', x=0.2, z=ui_int * 0.01, mesh=meshes_reparam[ui_int])
        for j, idx in enumerate(indices):
            c1_p = np.array([mesh.vertices[idx] + 0.01 * i * np.array([0, 0, 1]) + 0.2*np.array([1, 0, 0]) for i, mesh in enumerate(meshes_reparam)])
            c1_e = np.array([[i, i + 1] for i in range(len(c1_p) - 1)])

            register_curve(name=f"Curve 2 {idx}", curve_nodes=c1_p, curve_edges=c1_e, idx_color= j+1)


if __name__ == '__main__':
    GT = False

    meshes_dir = '../Data/scantalk_extension/Meshes_infer_original'
    l_mesh_dir = len(os.listdir(meshes_dir))
    meshes = [tri.load(os.path.join(meshes_dir, 'frame_' + str(i+1).zfill(3) + '.ply')) for i in range(0, l_mesh_dir - 1)]
    indices = [2846, 3531, 1723, 3509]
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("tile")
    ps.set_ground_plane_height_factor(0)

    for j, idx in enumerate(indices):
        print(idx)
        c1_p = np.array([mesh.vertices[idx] + 0.01*i*np.array([0, 0, 1]) for i, mesh in enumerate(meshes)])
        c1_e = np.array([[i, i+1] for i in range(len(c1_p) - 1)])

        register_curve(name=f"Curve {idx}", curve_nodes=c1_p, curve_edges=c1_e, idx_color=j + 1)

    register_surface(name=f'Frame {0} Ours', mesh=meshes[ui_int], disp_vectors=None, disp_heatmap=None)

    meshes_dir = '../Data/scantalk_extension/Meshes_infer_reparam'
    l_mesh_dir = len(os.listdir(meshes_dir))
    meshes_reparam = [tri.load(os.path.join(meshes_dir, 'frame_' + str(i + 1).zfill(3) + '.ply')) for i in
              range(0, l_mesh_dir - 1)]
    indices = [1143, 1588, 660, 4648]

    for j, idx in enumerate(indices):
        print(idx)
        c1_p = np.array([mesh.vertices[idx] + 0.01 * i * np.array([0, 0, 1]) + 0.2*np.array([1, 0, 0]) for i, mesh in enumerate(meshes_reparam)])
        c1_e = np.array([[i, i + 1] for i in range(len(c1_p) - 1)])

        register_curve(name=f"Curve 2 {idx}", curve_nodes=c1_p, curve_edges=c1_e, idx_color=j + 1)

    register_surface(name=f'Frame {0} Reparam', mesh=meshes_reparam[ui_int], x=0.2, disp_vectors=None, disp_heatmap=None)

    ps.set_user_callback(callback)
    ps.show()
