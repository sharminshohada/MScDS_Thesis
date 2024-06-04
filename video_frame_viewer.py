import cv2
import tkinter as tk
from tkinter import simpledialog

def play_video_with_frame_numbers(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_number = 0
    paused = True  # Start paused

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            ret, frame = cap.read()  # This is to ensure the first frame is shown when the video starts
            paused = False  # Prevent loop from freezing at the start

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Display the frame number on the video
        overlay = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, f'Frame: {frame_number}', (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.imshow('Video', frame)

        key = cv2.waitKey(0)  # Wait indefinitely for a key press

        if key == ord('q'):
            break
        elif key == ord('n'):  # Advance to next frame
            paused = False
        elif key == ord('p'):  # Go back to previous frame
            frame_number = max(0, frame_number - 2)  # Subtract 2 because reading next frame increments by 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            paused = True
        elif key == ord('j'):  # Jump to specific frame
            paused, frame_number = jump_to_frame(cap, paused)
        elif key == ord(' '):  # Toggle pause
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

def jump_to_frame(cap, paused):
    root = tk.Tk()
    root.withdraw()
    frame_str = simpledialog.askstring("Frame Input", "Enter the frame number to jump to:", parent=root)

    if frame_str:
        frame_number = int(frame_str)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        return True, frame_number  # Always pause after jumping to a frame

    return paused, int(cap.get(cv2.CAP_PROP_POS_FRAMES))

# Replace 'path_to_video.mp4' with the path to your video file
play_video_with_frame_numbers('D:/Research/2024/2finalprep/P3_Cheerio_V.mp4')
