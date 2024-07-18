import os
from subprocess import call

def render_from_png(path, audio_path, out):
    temp = "../../papers/ScanTalk/videos/render_BIWI/temp.mp4"

    cmd = "ffmpeg -r 25 -f image2 -s 1920x1080 -i " + path + "/%04d.png -i " + audio_path + " -map 0:v -map 1:a -vcodec libx264 -pix_fmt yuv420p " + temp
    call(cmd, shell=True)

    crop = 'crop=960:1080:520:0'
    cmd = "ffmpeg -i " + temp + " -vf " + crop + " -c:a copy " + out
    call(cmd, shell=True)

path = "../../papers/ScanTalk/videos/render_BIWI/rendered"
audio_path = "../datasets/BIWI/data/wav/F8_40.wav"
out = "../../papers/ScanTalk/videos/render_BIWI/targets/target.mp4"

def combine_videos(path1, path2, out):
    layout = "xstack=inputs=2:layout='0_0|w0_0'"

    cmd = "ffmpeg -i " + path1 + " -i " + path2 + " -filter_complex " + layout + " -codec:a copy " + out

    call(cmd, shell=True)

path1="../../papers/ScanTalk/videos/render_BIWI/targets/target.mp4"
path2="../../papers/ScanTalk/videos/render_BIWI/preds/pred.mp4"
out_combined = "../../papers/ScanTalk/videos/render_BIWI/gt_vs_pred.mp4"

#render_from_png(path, audio_path, out)
combine_videos(path1=path1, path2=path2, out=out_combined)