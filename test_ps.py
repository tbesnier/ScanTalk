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
        min_bound, max_bound = disp_heatmap.min(), disp_heatmap.max() #0, 0.01 #0, 0.0085  # disp_heatmap.min(), disp_heatmap.max()
        mesh.add_scalar_quantity('relative error heatmap', disp_heatmap, enabled=True, cmap='pink-green', vminmax=(min_bound, max_bound))

    return mesh

# Define our callback function, which Polyscope will repeatedly execute while running the UI.
def callback():

    global ui_int, meshes, meshes_gt, meshes_facediffuser, meshes_faceformer, heatmap, disp_vectors, disp_vectors_facediffusers, disp_vectors_faceformer

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
        #register_surface(name=f'Frame {ui_int} Ours', x=0.5, mesh=meshes[ui_int],
        #                 disp_vectors=disp_vectors[ui_int])
        # register_surface(name=f'Frame {ui_int} FaceDiffuser', idx_color=0, mesh=meshes_facediffuser[ui_int],
        #                  disp_vectors=disp_vectors_facediffuser[ui_int])
        # register_surface(name=f'Frame {ui_int} Faceformer', x=0.25, idx_color=0, mesh=meshes_faceformer[ui_int],
        #                  disp_vectors=disp_vectors_faceformer[ui_int])

        if meshes_gt is not None:
            register_surface(name=f'Disp GT {ui_int}', x=-0., y=-0., z=0., idx_color=1, mesh=meshes_gt[0],
                         disp_vectors=None, disp_heatmap=error_heatmap[ui_int])  #error_heatmap[ui_int])
            # register_surface(name=f'GT {ui_int}', x=-0.25, y=-0., z=0., idx_color=0, mesh=meshes_gt[ui_int],
            #                  disp_vectors=None, disp_heatmap=None)  # error_heatmap[ui_int])


if __name__ == '__main__':
    GT = False

    meshes_dir = '../Data/VOCA/res/Results_Actor/Meshes_infer' #'../datasets/VOCA/FaceTalk_170809_00138_TA/sentence01' #"../../papers/ScanTalk/meshes_results_scans/Meshes_FaceTalk_170731_00024_TA" #'../Data/VOCA/res/meshing_robustness/UPDOWN/Meshes_infer'  #'../Data/VOCA/res/Results_Actor/Meshes_infer'
    l_mesh_dir = len(os.listdir(meshes_dir))
    meshes = [tri.load(os.path.join(meshes_dir, 'frame_' + str(i+1).zfill(3) + '.ply')) for i in range(0, l_mesh_dir - 1)]
    #disp_vectors = np.array([meshes[i + 1].vertices - meshes[i].vertices for i in range(len(meshes) -1)])

    meshes_gt=None
    if GT:
        meshes_gt_dir = "../datasets/VOCA/FaceTalk_170809_00138_TA/sentence01" #'../Data/VOCA/Targets_voca_meshes/FaceTalk_170731_00024_TA_sentence01' #'../datasets/VOCA/FaceTalk_170725_00137_TA/sentence01'
        l_mesh_gt_dir = len(os.listdir(meshes_gt_dir))
        meshes_gt = [tri.load(os.path.join(meshes_gt_dir, 'sentence01.' + str(i+1).zfill(6) + '.ply')) for i in range(0, l_mesh_gt_dir)]
        #disp_vectors_gt = np.array([meshes_gt[i+1].vertices - meshes_gt[i].vertices for i in range(len(meshes_gt) -1)])

    #end = min(l_mesh_dir, l_mesh_gt_dir)
    #error_heatmap = np.array([np.linalg.norm(meshes_gt[i].vertices - meshes_gt[0].vertices, axis=1) for i in range(end)])

    # meshes_dir_facediffuser = "../../papers/ScanTalk/meshes_results_scans/Meshes_FaceTalk_170811_03274_TA"  # '../Data/VOCA/res/meshing_robustness/UPDOWN/Meshes_infer'  #'../Data/VOCA/res/Results_Actor/Meshes_infer'
    # l_mesh_dir_facediffuser = len(os.listdir(meshes_dir_facediffuser))
    # meshes_facediffuser = [tri.load(os.path.join(meshes_dir_facediffuser, 'tst' + str(i + 1).zfill(3) + '.ply')) for i in
    #           range(0, l_mesh_dir_facediffuser - 1)]
    # disp_vectors_facediffuser = np.array([meshes_facediffuser[i + 1].vertices - meshes_facediffuser[i].vertices for i in range(len(meshes_facediffuser) -1)])
    #
    #
    # meshes_dir_faceformer = "../../papers/ScanTalk/meshes_results_scans/Meshes_FaceTalk_170913_03279_TA"  # '../Data/VOCA/res/meshing_robustness/UPDOWN/Meshes_infer'  #'../Data/VOCA/res/Results_Actor/Meshes_infer'
    # l_mesh_dir_faceformer = len(os.listdir(meshes_dir_faceformer))
    # meshes_faceformer = [tri.load(os.path.join(meshes_dir_faceformer, 'tst' + str(i + 1).zfill(3) + '.ply')) for i in
    #           range(0, l_mesh_dir_faceformer - 1)]
    # disp_vectors_faceformer = np.array([meshes_faceformer[i + 1].vertices - meshes_faceformer[i].vertices for i in range(len(meshes_faceformer) -1)])

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_ground_plane_height_factor(0)
    heatmap = np.load("../Data/VOCA/res/Results_Actor/viz_learned_features/actor_vertices_emb.npy")[0]
    #pca = PCA()
    #pca.fit(heatmap)
    heatmap = np.linalg.norm(heatmap, axis=1)
    #print(heatmap.max(), heatmap.min())

    register_surface(name=f'Frame {0} Ours', mesh=meshes[ui_int], disp_vectors=None,
                     disp_heatmap=heatmap)
    # register_surface(name=f'Disp 2', x=0.25, y=-0., z=0., idx_color=2, mesh=meshes[ui_int],
    #                  disp_vectors=None, disp_heatmap=heatmap[:, 1])
    # register_surface(name=f'Disp 3', x=0.5, y=-0., z=0., idx_color=3, mesh=meshes[ui_int],
    #                  disp_vectors=None, disp_heatmap=heatmap[:, 2])
    # register_surface(name=f'Frame {0} FaceDiffuser', idx_color=0, mesh=meshes_facediffuser[ui_int],
    #                  disp_vectors=disp_vectors_facediffuser[ui_int], disp_heatmap=None)
    # register_surface(name=f'Frame {0} FaceFormer', x=0.25, y=-0., z=0., idx_color=0, mesh=meshes_faceformer[ui_int],
    #                  disp_vectors=disp_vectors_faceformer[ui_int], disp_heatmap=None)

    if GT:
        register_surface(name=f'Disp GT {0}', x=0., y=-0., z=0., idx_color=1, mesh=meshes_gt[ui_int],
                         disp_vectors=None, disp_heatmap=heatmap[:, 0]) #error_heatmap[0])

        # register_surface(name=f'GT {ui_int}', x=-0.25, y=-0., z=0., idx_color=0, mesh=meshes_gt[ui_int],
        #                  disp_vectors=None, disp_heatmap=None)  # error_heatmap[ui_int])
    ps.set_user_callback(callback)
    ps.show()
