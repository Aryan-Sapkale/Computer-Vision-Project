# Computer Vision Practical Project

## Project Overview
This project demonstrates basic computer vision techniques for image processing, feature extraction, object detection, and motion analysis using Python and OpenCV.

## Tools and Libraries
- Python
- OpenCV
- NumPy
- Matplotlib
- Ultralytics YOLO

## Project Structure
- `dataset/images/` — input images
- `dataset/videos/` — input video
- `outputs/task1/` — image restoration and segmentation results
- `outputs/task2/` — feature extraction and matching results
- `outputs/task3/` — face and object detection results
- `outputs/task4/` — motion detection results
- `src/` — Python source code

## Task 1: Image Restoration and Segmentation
Applied Gaussian and salt-and-pepper noise, image filtering, image-quality comparison, thresholding, morphological operations, edge detection, and watershed segmentation.

## Task 2: Feature Extraction and Matching
Extracted HOG and SIFT features, matched image features, and compared images after rotation and resizing.

## Task 3: Object Recognition
Used Haar Cascade for face detection and pretrained YOLO for object detection. The sample image produced one face detection with Haar Cascade and one person detection with YOLO.

## Task 4: Motion Detection
Applied frame differencing, GMM background subtraction, and Lucas–Kanade optical flow to the recorded video. The video processing produced motion masks and optical-flow images.

## How to Run
Run the scripts from the project root using the project's Python environment:

```powershell
python src/task1_restoration.py
python src/task2_features.py
python src/task3_detection.py
python src/task4_motion.py