import cv2
import tkinter as tk
from tkinter import simpledialog

def play_video_with_frame_numbers(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    base_delay = int(1000 / fps)

    frame_number = 0
    paused = False
    slow_motion_start_frame = 0
    slow_motion_end_frame = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Check if current frame is within the slow-motion segment
        if slow_motion_start_frame <= frame_number <= slow_motion_end_frame:
            delay = 0  # Set delay to 0 for manual frame-by-frame progression
        else:
            delay = base_delay

        overlay = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(f'Frame: {frame_number}', font, 1, 2)[0]
        text_x, text_y = 50, 50
        cv2.rectangle(overlay, (text_x, text_y - text_size[1] - 10), (text_x + text_size[0] + 20, text_y + 10), (0, 0, 0), -1)
        cv2.putText(overlay, f'Frame: {frame_number}', (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 1, frame, 0, 0, frame)

        cv2.imshow('Video', frame)

        if not paused:
            key = cv2.waitKey(delay) & 0xFF
        else:
            key = cv2.waitKey() & 0xFF  # Wait indefinitely for a key press when paused

        if key == ord(' '):
            paused = not paused
        elif key == ord('d'):
            frame_number += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        elif key == ord('a'):
            frame_number = max(0, frame_number - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        elif key == ord('j'):
            paused, frame_number = jump_to_frame(cap, paused)
        elif key == ord('s'):
            slow_motion_start_frame, slow_motion_end_frame = set_slow_motion_segment_by_frame()
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def jump_to_frame(cap, paused):
    root = tk.Tk()
    root.withdraw()
    frame_str = simpledialog.askstring("Frame Input", "Enter the frame number to jump to:", parent=root)

    if frame_str:
        frame_number = int(frame_str)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        paused = True
        return paused, frame_number

    return paused, int(cap.get(cv2.CAP_PROP_POS_FRAMES))

def set_slow_motion_segment_by_frame():
    root = tk.Tk()
    root.withdraw()
    start_frame_str = simpledialog.askstring("Slow Motion Start Frame", "Enter start frame for slow motion:", parent=root)
    end_frame_str = simpledialog.askstring("Slow Motion End Frame", "Enter end frame for slow motion:", parent=root)

    if start_frame_str and end_frame_str:
        slow_motion_start_frame = int(start_frame_str)
        slow_motion_end_frame = int(end_frame_str)
        return slow_motion_start_frame, slow_motion_end_frame

    return 0, 0

# Replace 'path_to_video.mp4' with the path to your video file
play_video_with_frame_numbers('D:/Research/2024/2finalprep/P1_Cheerio_NV.mp4')
