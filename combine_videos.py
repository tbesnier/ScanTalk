import os
from subprocess import call

videos_path = "../videos_ST_geometries/"

paths = [videos_path + elt for elt in os.listdir(videos_path)]
#layout = "xstack=inputs=10:layout='0_0|w0_0|w0+w1_0|w0+w1+w2_0|w0+w1+w2+w3_0|0_h0|w0+h0|w0+w1_h0|w0+w1+w2_h0|w0+w1+w2+w3_h0'" #|0_h0+h1|w0_h0+h1|w0+w1_h0+h1|w0+w1+w2_h0+h1|w0+w1+w2+w3_h0+h1|0_h0+h1+h2|w0_h0+h1+h2|w0+w1_h0+h1+h2|w0+w1+w2_h0+h1+h2|w0+w1+w2+w3_h0+h1+h2'"
layout = "xstack=inputs=3:layout='0_0|w0_0|w0+w1_0'"

videos=""
for i, elt in enumerate(paths):
    videos += f"-i {elt} "

#cmd = "ffmpeg " + videos + " -filter_complex " + layout + " -codec:a copy ../../papers/ST_geometries.mp4"
#cmd = "ffmpeg -i ../videos_ST_geometries/test_head_and_neck.mp4 -i ../videos_ST_geometries/test_full_face_area.mp4 -i ../videos_ST_geometries/test_narrow_face_area.mp4 -filter_complex " + layout + " -codec:a copy ../../papers/ST_geometries.mp4"
#cmd = "ffmpeg -i ../export/videos/ds_scan_rendered_room_2.mp4 -i ../Data/VOCA/res/TH/room_2.wav -map 0:v -map 1:a ../../papers/ex_rendered_ds_scan.mp4"

print(cmd)
call(cmd, shell=True)
