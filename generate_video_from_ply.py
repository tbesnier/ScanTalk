import d2d
import spiral_utils as spiral_utils
import shape_data as shape_data
import argparse
import pickle
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import euclidean_distances
import torch

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

os.environ['PYOPENGL_PLATFORM'] = 'egl'

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

def render_sequence_meshes(audio_path, sequence_vertices, template, out_path , out_fname, fps, uv_template_fname='', texture_img_fname=''):
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
        i = i + 1
    writer.release()

    video_fname = os.path.join(out_path, out_fname)
    cmd = ('ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p -ar 22050 {2}'.format(
        tmp_video_file.name, audio_path, video_fname)).split()
    call(cmd)

def generate_mesh_video(audio_path, out_path, out_fname, meshes_path_fname, fps, template):

    sequence_fnames = sorted(glob.glob(os.path.join(meshes_path_fname, '*.ply*')))

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
    render_sequence_meshes(audio_path, sequence_vertices, template, out_path, out_fname, fps, uv_template_fname=uv_template_fname, texture_img_fname='')

def main():
    # In the function render_mesh_helper you can customize your rendering
    parser = argparse.ArgumentParser(description='Python file to render a sequence of meshes into a video')
    #Path where you want to save the video
    parser.add_argument("--save_path", type=str, default='/home/federico/Scrivania/ST/Data/Videos', help='path for video')
    #Path where the .ply of the meshes sequence is. It is important that the meshes are named in order, e.g tst000.ply, tst001.ply, tst002.ply ...
    parser.add_argument("--meshes_path", type=str, default="/home/federico/Scrivania/ST/Data/saves/Meshes_Zero_Init_Masked_Loss_with_Lambda/160", help='path for the meshes sequence')
    #Path of the mesh template, for the FLAME topology, I put it in your folder, like this code
    parser.add_argument("--flame_template", type=str, default="/home/federico/Scrivania/D2D/template/flame_model/FLAME_sample.ply", help='template_path')
    parser.add_argument("--video_name", type=str, default='test_masked_loss_with_lambda_160_epochs.mp4', help='name of the video')
    parser.add_argument("--audio_path", type=str, default='/home/federico/Scrivania/TH/photo.wav', help='audio')
    parser.add_argument("--fps", type=int, default=30, help='frames per second')

    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    print('Video Generation')
    generate_mesh_video(args.audio_path,
                        args.save_path,
                        args.video_name,
                        args.meshes_path,
                        args.fps,
                        args.flame_template)
    print('done')


if __name__ == '__main__':
    main()
