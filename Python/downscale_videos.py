"""
Author: Shohada Sharmin
Date: 2024-06-05
Description: This script processes all MP4 videos in the current directory by:
             1. Reading each MP4 video file and determining its frame rate and dimensions.
             2. Calculating the scaling factor to make all videos a uniform target height and width.
             3. Resizing each frame and converting the video to .mp4 format.
             4. Saving the processed videos in a designated 'DownScaled_Videos' directory.
             5. Processing all videos together in a batch process.

Dependencies:
- OpenCV (`cv2`): Used for video processing.
- os: Used for file and directory operations.
- tqdm: Used for displaying a progress bar.
"""

# Function to downscale a video to the target height and width.
import cv2
import os
from tqdm import tqdm

def downscale_video(video_path, output_path, target_height, target_width):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open file: {video_path}")
        return

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if width == 0 or height == 0:
        print(f"Invalid dimensions for file: {video_path}")
        cap.release()
        return

    if height > width:
        fract = target_height / height
    else:
        fract = target_width / width

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (int(width * fract), int(height * fract)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (int(width * fract), int(height * fract)), interpolation=cv2.INTER_CUBIC)
        out.write(resized_frame)

    cap.release()
    out.release()

# Main function to process all MP4 videos in the current directory, downscale them, and save them in a 'DownScaled_Videos' directory.
def main():
    target_height = 512
    target_width = 512
    current_dir = os.getcwd()
    file_list = os.listdir(current_dir)

    downscaled_dir = os.path.join(current_dir, "DownScaled_Videos")
    if not os.path.exists(downscaled_dir):
        os.mkdir(downscaled_dir)

    for file_name in tqdm(file_list):
        if file_name.endswith(".MP4"):
            video_path = os.path.join(current_dir, file_name)
            base_name = os.path.splitext(file_name)[0]  # Get the base name of the file
            output_file_name = base_name + "_downscaled.mp4"  # Append '_downscaled' and change extension to .mp4
            output_path = os.path.join(downscaled_dir, output_file_name)
            downscale_video(video_path, output_path, target_height, target_width)

if __name__ == "__main__":
    main()
