# HandMovement project 
This project uses DeepLabCut for analyzing hand movements. Due to the use of Git LFS (Large File Storage), some files cannot be directly viewed on GitHub. To access these files:
Clone the repository: git clone https://github.com/your-username/HandMovement.git
Download the LFS files: git lfs pull
The remaining Python files can be viewed directly on GitHub.

------------------------------------------------------------------------------------------------------------------------------
# video_downscale_processor.py
This Python script processes all MP4 videos in the current directory by reading each file, resizing the frames to a uniform target height and width, and saving the processed videos in a designated directory.

## Requirements
- Python installed
- OpenCV library (`cv2`) installed
- os library installed
- tqdm library installed

## Usage Instructions
1. Place all MP4 videos that you want to process in the same directory as this script.
2. Run the script.
3. The script will:
    - Read each MP4 video file and determine its frame rate and dimensions.
    - Calculate the scaling factor to make all videos a uniform target height and width.
    - Resize each frame and convert the video to .mp4 format.
    - Save the processed videos in a designated 'DownScaled_Videos' directory.
    - Process all videos together in a batch process.

---------------------------------------------------------------------------------------------------------------------------------
# video_frame_viewer.py
This Python script uses OpenCV and Tkinter to play a video file and allow frame-by-frame navigation with the frame numbers displayed on the video.

## Requirements
- Python installed
- OpenCV library (`cv2`) installed
- Tkinter library (`tk` and `simpledialog`) installed

## Usage Instructions
1. Replace `'D:/Research/2024/2finalprep/P3_Cheerio_V.mp4'` with the path to your video file.
2. Run the script.
3. The video will start in a paused state. Use the following keys to control playback:
    - `n`: Advance to the next frame.
    - `p`: Go back to the previous frame.
    - `j`: Jump to a specific frame by entering the frame number in the dialog box.
    - Spacebar (` `): Toggle between play and pause.
    - `q`: Quit the video player.

This script is particularly useful for applications requiring precise video analysis and frame-level inspection, such as behavioral studies, motion analysis, or quality control in video production.

------------------------------------------------------------------------------------------------------------------
# csv_data_processor.py
This Python script processes multiple CSV files by reading each file, adjusting the column names, filtering the data, and saving the processed data to new CSV files.

## Requirements
- Python installed
- pandas library installed

## Usage Instructions
1. Ensure all CSV files that you want to process are listed in the `file_paths` variable within the script.
2. Adjust the file paths in the `file_paths` list to match the location of your files.
3. Run the script.
4. The script will:
    - Read each CSV file with a multi-level header and adjust the column names.
    - Increment the first column's values by 1 and convert the 'Label' column to numeric.
    - Filter the data to include only specific labels (1, 2, 3, or 4) and remove columns ending with '_likelihood'.
    - Save the processed data to new CSV files with '_Filtered' appended to the original filenames.
    - Count and display the occurrences of each label in the 'Label' column for each processed file.
      
---------------------------------------------------------------------------------------------------------------------
  # hand_movement_classification.py
This Python script is designed to classify hand movements (reaching and grasping) from video data that has been processed into CSV files. Each CSV file contains coordinates of body parts over time, and the script extracts features like movement, velocity, acceleration, finger spread, and hand angle. The models are trained and evaluated on this feature set, with results saved for further analysis.

## Requirements
- Python installed
- pandas library installed
- sklearn library installed
- joblib library installed
- numpy library installed
- os library installed
- re library installed

## Usage Instructions
1. Ensure that all CSV files that you want to process are listed in the `all_files` variable within the script.
2. Adjust the file paths in the `all_files` list to match the location of your files.
3. Run the script.
4. The script will:
    - Load and preprocess data from each CSV file.
    - Perform feature engineering, including the calculation of finger spread and hand angle.
    - Train machine learning models (Logistic Regression, Random Forest, Gradient Boosting, SVM) and an ensemble model (Voting Classifier).
    - Evaluate the models using metrics such as accuracy, precision, recall, and F1-score.
    - Save the trained models and the preprocessing transformers (imputer, scaler).
    - Load the trained models and make predictions on test data.
    - Perform hyperparameter tuning using GridSearchCV.
    - Calculate and print average metrics across all scenarios.









