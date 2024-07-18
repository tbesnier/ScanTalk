import os
import subprocess


def create_gif(input_dir, output_file, frame_rate=30):
    # Get a list of all PNG files in the input directory
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])

    # Generate ffmpeg command to create the GIF
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it already exists
        '-framerate', str(frame_rate),  # Set frame rate
        '-i', os.path.join(input_dir, '%04d.png'),  # Input file pattern
        '-vf', 'palettegen',  # Generate palette
        '-y',  # Overwrite palette file if it already exists
        os.path.join(input_dir, 'palette.png')  # Output palette file
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    ffmpeg_cmd = [
        'ffmpeg',
        '-framerate', str(frame_rate),  # Set frame rate
        '-i', os.path.join(input_dir, '%04d.png'),  # Input file pattern
        '-i', os.path.join(input_dir, 'palette.png'),  # Input palette file
        '-lavfi', 'paletteuse',  # Use palette
        '-y',  # Overwrite output file if it already exists
        output_file
    ]
    subprocess.run(ffmpeg_cmd, check=True)


def create_video(image_folder, audio_file, output_file='output.mp4'):
    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    image_files.sort()  # Sort files alphabetically

    # Get duration of audio file
    ffprobe_command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        audio_file
    ]
    duration = float(subprocess.check_output(ffprobe_command))

    # Command to create video using ffmpeg
    ffmpeg_command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-thread_queue_size', '512',  # Increase thread queue size
        '-r', '50',  # Frames per second
        '-i', os.path.join(image_folder, '%04d.png'),  # Input images
        '-i', audio_file,  # Input audio
        '-c:v', 'libx264',  # Video codec
        '-c:a', 'aac',  # Audio codec
        '-strict', 'experimental',  # Required for some AAC encoders
        '-shortest',  # End the output when the shortest input ends
        '-vf', 'fps=50',  # Frame rate
        '-t', str(duration),  # Set video duration to match audio duration
        output_file  # Output file
    ]
    print(ffmpeg_command)

    # Run ffmpeg command
    subprocess.run(ffmpeg_command)


def render_video(image_folder, audio_file, output_file='output.mp4'):
    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    image_files.sort()  # Sort files alphabetically

    # Command to create video using ffmpeg
    ffmpeg_command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-framerate', '50',  # Frames per second
        '-i', os.path.join(image_folder, '%04d.png'),  # Input images
        '-i', audio_file,
        '-c:v', 'libx264',  # Video codec
        '-c:a', 'aac',
        '-shortest',
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        output_file  # Output file
    ]

    # Run ffmpeg command
    subprocess.run(ffmpeg_command)


# Example usage:
image_folder = '/home/tbesnier/phd/papers/ScanTalk/render_scan_photo'
output_file = '/home/tbesnier/phd/papers/ScanTalk/render_scan.mp4'
audio_file = '/home/tbesnier/phd/papers/ScanTalk/audios/photo.wav'

render_video(image_folder, audio_file, output_file)


# # Example usage:
# image_folder = '/home/tbesnier/phd/papers/ScanTalk/render_arnold_no_ground/'
# audio_file = '/home/tbesnier/phd/papers/ScanTalk/audios/photo.wav'
# output_file = '/home/tbesnier/phd/papers/ScanTalk/render.mp4'
#
# create_video(image_folder, audio_file, output_file)
