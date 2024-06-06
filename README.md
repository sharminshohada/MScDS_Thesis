The HandMovement project uses DeepLabCut for analyzing hand movements. Due to the use of Git LFS (Large File Storage), some files cannot be directly viewed on GitHub. To access these files:
Clone the repository: git clone https://github.com/your-username/HandMovement.git
Download the LFS files: git lfs pull
The remaining Python files can be viewed directly on GitHub.

video_frame_viewer.py: This script plays a video with frame numbers displayed on each frame by:
1. Opening and processing the video file using OpenCV.
2. Displaying the frame number on each frame of the video.
3. Allowing the user to control the playback with the following features:
- 'q' to quit the video playback.
- 'n' to advance to the next frame.
- 'p' to go back to the previous frame.
- 'j' to jump to a specific frame based on user input.
- Spacebar (' ') to toggle play/pause.
4. Providing a GUI dialog for frame input using Tkinter when jumping to a specific frame.

Requirements: Python installed, OpenCV library (cv2) installed, Tkinter library (tk and simpledialog) installed
