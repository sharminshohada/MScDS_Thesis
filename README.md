The HandMovement project uses DeepLabCut for analyzing hand movements. Due to the use of Git LFS (Large File Storage), some files cannot be directly viewed on GitHub. To access these files:
Clone the repository: git clone https://github.com/your-username/HandMovement.git
Download the LFS files: git lfs pull
The remaining Python files can be viewed directly on GitHub.

video_frame_viewer.py . This Python script uses OpenCV and Tkinter to play a video file and allow frame-by-frame navigation with the frame numbers displayed on the video. 

Requirements: Python installed, OpenCV library (cv2), Tkinter library (tk).

Usage Instructions:
Replace 'D:/Research/2024/2finalprep/P3_Cheerio_V.mp4' with the path to your video file.
Run the script.
The video will start in a paused state. Use the following keys to control playback:
n: Advance to the next frame.
p: Go back to the previous frame.
j: Jump to a specific frame by entering the frame number in the dialog box.
(spacebar): Toggle between play and pause.
q: Quit the video player.
This script is particularly useful for applications requiring precise video analysis and frame-level inspection, such as behavioral studies, motion analysis, or quality control in video production
