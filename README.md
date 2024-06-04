video_frame_viewer.py
This script provides functionality to play a video file and display the frame number on each frame. It includes the ability to pause, resume, navigate to the next frame, go back to the previous frame, and jump to a specific frame. The script uses OpenCV for video processing and Tkinter for a simple GUI to input frame numbers.

Features:
- Display the current frame number on the video.
- Pause and resume video playback.
- Navigate to the next and previous frames.
- Jump to a specific frame using a simple input dialog.
- Ensure consistent video processing by utilizing OpenCV and Tkinter libraries.

Usage:
1. Replace 'path_to_video.mp4' with the path to your video file in the play_video_with_frame_numbers function call.
2. Run the script to open the video player window.
3. Use the following key commands to control the video playback:
   - 'q': Quit the video player.
   - 'n': Advance to the next frame.
   - 'p': Go back to the previous frame.
   - 'j': Jump to a specific frame (prompts for frame number).
   - ' ': Toggle pause/resume playback.

Dependencies:
- OpenCV: Install using 'pip install opencv-python'
- Tkinter: Typically included with Python installations.

Example:
To use this script, make sure to specify the correct path to your video file and run the script. The video will open in a new window, and you can control the playback using the specified key commands.

Author: Shohada Sharmin
Date: 2024-06-04
