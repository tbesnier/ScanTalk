import argparse
import pickle
import shape_data
from sklearn.metrics.pairwise import euclidean_distances
import torch
from d2d import SpiralAutoencoder
from transformers import Wav2Vec2Processor
import time
import os
import cv2
import tempfile
import numpy as np
from subprocess import call
from psbody.mesh import Mesh
import pyrender
import trimesh
import glob
import librosa
from utils import utils, mesh_sampling

os.environ['PYOPENGL_PLATFORM'] = 'egl'
def get_unit_factor(unit):
    if unit == 'mm':
        return 1000.0
    elif unit == 'cm':
        return 100.0
    elif unit == 'm':
        return 1.0
    else:
        raise ValueError('Unit not supported')

def render_mesh_helper(mesh, t_center, rot=np.zeros(3), tex_img=None, v_colors=None, errors=None, error_unit='m', min_dist_in_mm=0.0, max_dist_in_mm=3.0, z_offset=0):

    background_black = True
    camera_params = {'c': np.array([400, 400]),
                     'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                     'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v - t_center).T).T + t_center

    intensity = 2.0
    rgb_per_v = None

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
        alphaMode='BLEND',
        baseColorFactor=[0.3, 0.3, 0.3, 1.0],
        metallicFactor=0.8,
        roughnessFactor=0.8
    )

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material, smooth=True)

    if background_black:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])  # [0, 0, 0] black,[255, 255, 255] white
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2],
                               bg_color=[255, 255, 255])  # [0, 0, 0] black,[255, 255, 255] white

    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                       fy=camera_params['f'][1],
                                       cx=camera_params['c'][0],
                                       cy=camera_params['c'][1],
                                       znear=frustum['near'],
                                       zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 1.0 - z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3, 3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3, 3] = pos
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def render_sequence_meshes(audio_fname, sequence_vertices, template, out_path , out_fname, fps, uv_template_fname='', texture_img_fname=''):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path)
    if int(cv2.__version__[0]) < 3:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), fps, (800, 800), True)
    else:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (800, 800), True)

    if os.path.exists(uv_template_fname) and os.path.exists(texture_img_fname):
        uv_template = Mesh(filename=uv_template_fname)
        vt, ft = uv_template.vt, uv_template.ft
        tex_img = cv2.imread(texture_img_fname)[:,:,::-1]
    else:
        vt, ft = None, None
        tex_img = None

    num_frames = sequence_vertices.shape[0]
    center = np.mean(sequence_vertices[0], axis=0)
    i = 0
    for i_frame in range(num_frames - 2):
        render_mesh = Mesh(sequence_vertices[i_frame], template.f)
        if vt is not None and ft is not None:
            render_mesh.vt, render_mesh.ft = vt, ft
        img = render_mesh_helper(render_mesh, center, tex_img=tex_img)
        writer.write(img)
        #cv2.imwrite('/home/federico/Scrivania/TH/Meshes_for_Qualitative/FF_Happy/Images/' + str(i_frame).zfill(3) + '.png', img[:, 100:-100])
        i = i + 1
    writer.release()

    video_fname = os.path.join(out_path, out_fname)
    cmd = ('ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p -ar 22050 {2}'.format(
        audio_fname, tmp_video_file.name, video_fname)).split()
    call(cmd)

def generate_mesh_video(out_path, out_fname, meshes_path_fname, fps, audio_fname, template):

    sequence_fnames = sorted(glob.glob(os.path.join(meshes_path_fname, '*.ply*')))

    audio_fname = audio_fname


    uv_template_fname = template
    sequence_vertices = []
    f = None

    for frame_idx, mesh_fname in enumerate(sequence_fnames):
        frame = Mesh(filename=mesh_fname)
        sequence_vertices.append(frame.v)
        if f is None:
            f = frame.f

    template = Mesh(sequence_vertices[0], f)
    sequence_vertices = np.stack(sequence_vertices)
    render_sequence_meshes(audio_fname, sequence_vertices, template, out_path, out_fname, fps, uv_template_fname=uv_template_fname, texture_img_fname='')


def generate_meshes(args):

    device = args.device
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    meshpackage = 'trimesh'

    shapedata = shape_data.ShapeData(nVal=100,
                              reference_mesh_file=args.reference_mesh_file,
                              normalization=False,
                              meshpackage=meshpackage, load_flag=False)

    shapedata.n_vertex = 5023
    shapedata.n_features = 3
    
    reference = trimesh.load(args.reference_mesh_file, process=False)
    template_tri = reference.faces

    # generate/load transform matrices
    transform_fp = '/home/federico/Scrivania/ST/ScanTalk/template/template/transform.pkl'
    template_fp = args.reference_mesh_file
    if not os.path.exists(transform_fp):
        print('Generating transform matrices...')
        mesh = Mesh(filename=template_fp)
        ds_factors = [4, 4, 4, 4]
        _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(
            mesh, ds_factors)
        tmp = {
            'vertices': V,
            'face': F,
            'adj': A,
            'down_transform': D,
            'up_transform': U
        }
        print(tmp)

        with open(transform_fp, 'wb') as fp:
            pickle.dump(tmp, fp)
        print('Done!')
        print('Transform matrices are saved in \'{}\''.format(transform_fp))
    else:
        with open(transform_fp, 'rb') as f:
            tmp = pickle.load(f, encoding='latin1')

    spiral_indices_list = [
        utils.preprocess_spiral(tmp['face'][idx], args.seq_length[idx],
                                tmp['vertices'][idx],
                                args.dilation[idx]).to(device)
        for idx in range(len(tmp['face']) - 1)
    ]
    down_transform_list = [
        utils.to_sparse(down_transform).to(device)
        for down_transform in tmp['down_transform']
    ]
    up_transform_list = [
        utils.to_sparse(up_transform).to(device)
        for up_transform in tmp['up_transform']
    ]
    
    d2d = SpiralAutoencoder(args.in_channels, args.out_channels, args.latent_channels,
           spiral_indices_list, down_transform_list,
           up_transform_list).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    d2d.load_state_dict(checkpoint['autoencoder_state_dict'])


    speech_array, sampling_rate = librosa.load(os.path.join(args.audio), sr=16000)
    audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device) 

    with open(args.template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    actor_vertices = templates[args.actor_name]

    actor = np.reshape(actor_vertices, (1, 5023, 3))
    actor = torch.FloatTensor(actor).to(device=args.device)
    
    d2d.eval()
    with torch.no_grad():
        gen_seq = d2d.predict(audio_feature, actor.float())           
        gen_seq = gen_seq.cpu().detach().numpy()

    for k in range(len(gen_seq)):
        tri_mesh_mixture = trimesh.Trimesh(np.array(gen_seq[k]), np.asarray(template_tri), process=False)
        tri_mesh_mixture.export(os.path.join(args.save_path, 'Meshes', "tst" + str(k).zfill(3) + ".ply"))

    print('Done')

'''
"FaceTalk_170725_00137_TA"
"FaceTalk_170728_03272_TA"
"FaceTalk_170731_00024_TA"
"FaceTalk_170809_00138_TA"
"FaceTalk_170811_03274_TA"
"FaceTalk_170811_03275_TA"
"FaceTalk_170904_00128_TA"
"FaceTalk_170904_03276_TA"
"FaceTalk_170908_03277_TA"
"FaceTalk_170912_03278_TA"
"FaceTalk_170915_00223_TA"
'''

def main():
    parser = argparse.ArgumentParser(description='D2D: Dense to Dense Encoder-Decoder')
    parser.add_argument("--reference_mesh_file", type=str, default='/home/federico/Scrivania/ST/ScanTalk/template/flame_model/FLAME_sample.ply', help='path of the template')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default='/home/federico/Scrivania/ST/Data/Demo_audio_20s')
    parser.add_argument("--audio", type=str, default='/home/federico/Scrivania/ST/Data/Test_Audio/audio_20s.wav')
    parser.add_argument("--template_file", type=str, default="/home/federico/Scrivania/TH/S2L/vocaset/templates.pkl", help='faces to animate')
    parser.add_argument("--actor_name", type=str, default="FaceTalk_170809_00138_TA", help='face to animate')
    parser.add_argument("--model_path", type=str, default='/home/federico/Scrivania/ST/Data/results/d2d_ScanTalk_bigger_lstm_masked_velocity_loss.pth.tar')
    parser.add_argument("--video_name", type=str, default='audio_20s.mp4')     
    parser.add_argument("--fps", type=int, default=30, help='frames per second')

    ##Spiral++ hyperparameters
    parser.add_argument('--out_channels',
                        nargs='+',
                        default=[32, 64, 64, 128],
                        type=int)
    parser.add_argument('--latent_channels', type=int, default=128)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--seq_length', type=int, default=[12, 12, 12, 12], nargs='+')
    parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')

    args = parser.parse_args()
    
    save_path = args.save_path

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, 'Meshes'))

    generate_meshes(args)

    print('Video Generation')
    generate_mesh_video(save_path,
                        args.video_name,
                        os.path.join(save_path, 'Meshes'),
                        args.fps,
                        args.audio,
                        args.reference_mesh_file)
    print('done')

if __name__ == '__main__':
    main()
