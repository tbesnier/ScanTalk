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

    mesh = ps.register_surface_mesh(name, vertices, faces)
    mesh.set_color(tuple(int(colors[idx_color][i:i + 2], 16) / 255.0 for i in (1, 3, 5)))
    mesh.set_smooth_shade(False)
    mesh.set_transparency(transparency)

    if disp_vectors is not None:
        mesh.add_vector_quantity("displacement vectors", 5*disp_vectors, enabled=True,
                                 color=tuple(int(colors[-1][i:i + 2], 16) / 255.0 for i in (1, 3, 5)), vectortype="ambient")

    if disp_heatmap is not None:
        mesh.add_scalar_quantity('relative error heatmap', disp_heatmap, enabled=True, cmap='reds', vminmax=(0.0, 0.06))

    return mesh

# Define our callback function, which Polyscope will repeatedly execute while running the UI.
def callback():

    global ui_int, meshes, meshes_gt, disp_vectors_gt

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
        register_surface(name=f'Frame {ui_int}', mesh=meshes[ui_int], disp_vectors=disp_vectors[ui_int])
        register_surface(name=f'Frame GT {ui_int}', x=0.25, y=-0.02, z=-0.05, idx_color=1, mesh=meshes_gt[ui_int],
                         disp_vectors = disp_vectors_gt[ui_int], disp_heatmap=error_heatmap[ui_int])


if __name__ == '__main__':
    GT = True

    meshes_dir = '../Data/VOCA/res/Results_Actor/Meshes_Training/150'
    l_mesh_dir = len(os.listdir(meshes_dir))
    meshes = [tri.load(os.path.join(meshes_dir, 'frame_' + str(i).zfill(3) + '.ply')) for i in range(0, l_mesh_dir)]
    disp_vectors = np.array([meshes[i + 1].vertices - meshes[i].vertices for i in range(len(meshes) - 1)])

    if GT:
        meshes_gt_dir = '../datasets/VOCA/FaceTalk_170725_00137_TA/sentence01'
        l_mesh_gt_dir = len(os.listdir(meshes_gt_dir))
        meshes_gt = [tri.load(os.path.join(meshes_gt_dir, 'sentence01.' + str(i).zfill(6) +'.ply')) for i in range(1, l_mesh_gt_dir)]
        disp_vectors_gt = np.array([meshes_gt[i+1].vertices - meshes_gt[i].vertices for i in range(len(meshes_gt) - 1)])
        error_heatmap = np.array([np.linalg.norm(meshes_gt[i+1].vertices - meshes[i].vertices, axis=1) for i in range(len(meshes_gt) - 1)])

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0)
    register_surface(name=f'Frame {0}', mesh=meshes[ui_int], disp_vectors=disp_vectors[0])
    print(error_heatmap[0].shape)
    if GT:
        register_surface(name=f'Frame GT {0}', x=0.25, y=-0.02, z=-0.05, idx_color=1, mesh=meshes_gt[ui_int],
                         disp_vectors = disp_vectors_gt[0], disp_heatmap=error_heatmap[0])
    ps.set_user_callback(callback)
    ps.show()
