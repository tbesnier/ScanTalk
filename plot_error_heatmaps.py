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

    mesh = ps.register_surface_mesh(name, vertices, faces, material='normal')
    mesh.set_color(tuple(int(colors[idx_color][i:i + 2], 16) / 255.0 for i in (1, 3, 5)))
    mesh.set_smooth_shade(True)
    mesh.set_transparency(transparency)

    if disp_vectors is not None:
        mesh.add_vector_quantity("displacement vectors", 5*disp_vectors, enabled=True,
                                 color=tuple(int(colors[-1][i:i + 2], 16) / 255.0 for i in (1, 3, 5)), vectortype="ambient")

    if disp_heatmap is not None:
        min_bound, max_bound = 0, 0.009  #disp_heatmap.min(), disp_heatmap.max()
        mesh.add_scalar_quantity('relative error heatmap', disp_heatmap, enabled=True, cmap='jet', vminmax=(min_bound, max_bound))

    return mesh

# Define our callback function, which Polyscope will repeatedly execute while running the UI.
def callback():

    global ui_int, meshes_gt, meshes_codetalker

    # == Settings

    # Note that it is a push/pop pair, with the matching pop() below.
    psim.PushItemWidth(150)

    # == Show text in the UI

    psim.TextUnformatted("Sequence of meshes")
#    psim.TextUnformatted("Sequence length: {}".format(len(meshes_gt)))
    psim.Separator()

    # Input Int Slider
    changed, ui_int = psim.SliderInt("Frame", ui_int, v_min=0, v_max=len(meshes_codetalker)-1)
    if changed:
        ps.remove_all_structures()
        #register_surface(name=f'Frame {ui_int}', mesh=meshes[ui_int], disp_vectors=None)
        register_surface(name=f'Frame GT {ui_int}', x=0., y=-0., z=0., idx_color=1, mesh=meshes_codetalker[ui_int],
                         disp_vectors=None, disp_heatmap=None)  # error_heatmap[ui_int])
        if meshes_gt is not None:
            register_surface(name=f'Frame GT {ui_int}', x=0., y=-0., z=0., idx_color=1, mesh=meshes_codetalker[ui_int],
                         disp_vectors=None, disp_heatmap=None)  #error_heatmap[ui_int])


if __name__ == '__main__':
    GT = False

    # meshes_dir_hubert = '../Data/VOCA/RUN_voca_scantalk_hubert_meshes/FaceTalk_170731_00024_TA_sentence38'  #'../Data/VOCA/res/Results_Actor/Meshes_infer'
    # l_mesh_dir_hubert = len(os.listdir(meshes_dir_hubert))
    # meshes_hubert = [tri.load(os.path.join(meshes_dir_hubert, 'frame_' + str(i+1).zfill(3) + '.ply')) for i in range(0, l_mesh_dir_hubert)]
    #
    # meshes_dir_wav2vec = '../Data/VOCA/RUN_voca_scantalk_wav2vec_meshes/FaceTalk_170731_00024_TA_sentence38'  # '../Data/VOCA/res/Results_Actor/Meshes_infer'
    # l_mesh_dir_wav2vec = len(os.listdir(meshes_dir_wav2vec))
    # meshes_wav2vec = [tri.load(os.path.join(meshes_dir_wav2vec, 'frame_' + str(i + 1).zfill(3) + '.ply')) for i in range(0, l_mesh_dir_wav2vec)]
    #
    # meshes_dir_facediffuser = '../Data/VOCA/RUN_voca_FaceDiffuser_meshes/FaceTalk_170731_00024_TA_sentence38'  # '../Data/VOCA/res/Results_Actor/Meshes_infer'
    # l_mesh_dir_facediffuser = len(os.listdir(meshes_dir_facediffuser))
    # meshes_facediffuser = [tri.load(os.path.join(meshes_dir_facediffuser, 'frame_' + str(i + 1).zfill(3) + '.ply')) for i in
    #                   range(0, l_mesh_dir_facediffuser)]

    meshes_dir_codetalker = '../Data/VOCA/res/Results_Actor/Meshes_infer' #'../results_voca/results_selftalk_voca_meshes/FaceTalk_170809_00138_TA_sentence40'  # '../Data/VOCA/res/Results_Actor/Meshes_infer'
    l_mesh_dir_codetalker = len(os.listdir(meshes_dir_codetalker))
    meshes_codetalker = [tri.load(os.path.join(meshes_dir_codetalker, 'frame_' + str(i + 1).zfill(3) + '.ply')) for i in
                      range(0, l_mesh_dir_codetalker-1)]

    meshes_gt=None
    if GT:
        meshes_gt_dir = '../Data/VOCA/Targets_voca_meshes/FaceTalk_170809_00138_TA_sentence40' #'../datasets/VOCA/FaceTalk_170725_00137_TA/sentence01'
        l_mesh_gt_dir = len(os.listdir(meshes_gt_dir))
        end = min(l_mesh_dir_codetalker, l_mesh_gt_dir)
        meshes_gt = [tri.load(os.path.join(meshes_gt_dir, 'frame_' + str(i+1).zfill(3) + '.ply')) for i in range(0, end)]
        # error_heatmap_hubert = np.array([np.linalg.norm(meshes_gt[i].vertices - meshes_hubert[i].vertices, axis=1) for i in range(len(meshes_gt))])
        # error_heatmap_wav2vec = np.array([np.linalg.norm(meshes_gt[i].vertices - meshes_wav2vec[i].vertices, axis=1) for i in range(len(meshes_gt))])
        # error_heatmap_facediffuser = np.array([np.linalg.norm(meshes_gt[i].vertices - meshes_facediffuser[i].vertices, axis=1) for i in range(len(meshes_gt))])
        error_heatmap_codetalker = np.array([np.linalg.norm(meshes_gt[i].vertices - meshes_codetalker[i].vertices, axis=1) for i in range(len(meshes_gt))])

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_ground_plane_height_factor(0)

    #heatmap = error_heatmap_codetalker
    #print(heatmap[ui_int])

    #print(heatmap.max(), heatmap.min())

    if GT:
        register_surface(name=f'Frame GT {0}', x=0., y=-0., z=0., idx_color=1, mesh=meshes_gt[ui_int],
                         disp_vectors = None, disp_heatmap=heatmap[ui_int]) #error_heatmap[0])

    #register_surface(name=f'Frame GT {0}', x=0., y=-0., z=0., idx_color=1, mesh=meshes_codetalker[ui_int],
    #                                  disp_vectors=None, disp_heatmap=None)  # error_heatmap[0])

    list_frames = [0, 9, 24, 31, 36, 61]
    for frame_num in list_frames:
        print(frame_num)
        register_surface(name=f'Frame {frame_num}', x=0., y=-0., z=0., idx_color=1, mesh=meshes_codetalker[frame_num],
                     disp_vectors=None, disp_heatmap=None, )  # error_heatmap[0])

        ps.screenshot(filename="../../papers/ScanTalk/supplementary/more_exp/thanos/" + str(frame_num) + '.png')
        ps.remove_all_structures()

    ps.set_user_callback(callback)
    ps.show()
