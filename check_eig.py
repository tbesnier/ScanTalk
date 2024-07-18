import polyscope as ps
import polyscope.imgui as psim
import trimesh as tri
import numpy as np
import os, time
from sklearn.decomposition import PCA

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

    mesh = ps.register_surface_mesh(name, vertices, faces)
    mesh.set_color(tuple(int(colors[idx_color][i:i + 2], 16) / 255.0 for i in (1, 3, 5)))
    mesh.set_smooth_shade(False)
    mesh.set_transparency(transparency)

    if disp_vectors is not None:
        mesh.add_vector_quantity("displacement vectors", 5*disp_vectors, enabled=True,
                                 color=tuple(int(colors[-1][i:i + 2], 16) / 255.0 for i in (1, 3, 5)), vectortype="ambient")

    if disp_heatmap is not None:
        min_bound, max_bound = disp_heatmap.min(), disp_heatmap.max()  #
        mesh.add_scalar_quantity('relative error heatmap', disp_heatmap, enabled=True, cmap='pink-green', vminmax=(min_bound, max_bound))

    return mesh

# Define our callback function, which Polyscope will repeatedly execute while running the UI.
def callback():

    global ui_int, meshes, meshes_gt, meshes_facediffuser, meshes_faceformer, heatmap

    # == Settings

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
        register_surface(name=f'Frame {ui_int} Ours', x=0.5, mesh=meshes[ui_int], disp_vectors=None)
        register_surface(name=f'Frame {ui_int} CodeTalker', idx_color=0, mesh=meshes_facediffuser[ui_int], disp_vectors=None)
        register_surface(name=f'Frame {ui_int} Faceformer', x=0.25, y=-0., z=0., idx_color=0, mesh=meshes_faceformer[ui_int], disp_vectors=None)
        if meshes_gt is not None:
            register_surface(name=f'Frame GT {ui_int}', x=-0.25, y=-0., z=0., idx_color=1, mesh=meshes_gt[ui_int],
                         disp_vectors=None, disp_heatmap=None)  #error_heatmap[ui_int])


if __name__ == '__main__':
    mesh_path = "../datasets/BIWI/data/templates/F1_original.ply"
    mesh = tri.load(mesh_path)

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0)
    heatmap = np.load("../Data/BIWI/res/eig.npy")
    #pca = PCA()
    #pca.fit(heatmap)
    #heatmap = np.linalg.norm(heatmap, axis=1)

    #print(heatmap.max(), heatmap.min())

    register_surface(name="n=1", x=0, mesh=mesh, disp_vectors=None, disp_heatmap=heatmap[:,0])
    register_surface(name="n=2", x=0.5, mesh=mesh, disp_vectors=None, disp_heatmap=heatmap[:, 1])
    register_surface(name="n=3", x=1.0, mesh=mesh, disp_vectors=None, disp_heatmap=heatmap[:, 2])
    register_surface(name="n=5", x=1.5, mesh=mesh, disp_vectors=None, disp_heatmap=heatmap[:,5])
    register_surface(name="n=50", x=2.0, mesh=mesh, disp_vectors=None, disp_heatmap=heatmap[:,50])
    register_surface(name="n=128", x=2.5, mesh=mesh, disp_vectors=None, disp_heatmap=heatmap[:,-1])

    # register_surface(name="n=1", x=0, mesh=mesh_2, disp_vectors=None, disp_heatmap=heatmap[:, 0])
    # register_surface(name="n=2", x=0.25, mesh=mesh_2, disp_vectors=None, disp_heatmap=heatmap[:, 1])
    # register_surface(name="n=3", x=0.5, mesh=mesh_2, disp_vectors=None, disp_heatmap=heatmap[:, 2])
    # register_surface(name="n=5", x=0.75, mesh=mesh_2, disp_vectors=None, disp_heatmap=heatmap[:, 5])
    # register_surface(name="n=50", x=1.0, mesh=mesh_2, disp_vectors=None, disp_heatmap=heatmap[:, 50])
    # register_surface(name="n=128", x=1.25, mesh=mesh_2, disp_vectors=None, disp_heatmap=heatmap[:, -1])

    ps.show()
