import cv2
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "dataset" / "videos" / "WhatsApp Video 2026-11-02 at 4.47.33 PM.mp4"
OUTPUT_DIR = BASE_DIR / "outputs" / "task4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Open video
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

# Read the first frame
ret, previous_frame = cap.read()

if not ret:
    raise RuntimeError("Could not read the first video frame.")

previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
frame_count = 0

while True:
    ret, current_frame = cap.read()

    if not ret:
        break

    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Compare consecutive frames
    difference = cv2.absdiff(previous_gray, current_gray)

    # Convert the difference to a black-and-white motion mask
    _, motion_mask = cv2.threshold(
        difference, 25, 255, cv2.THRESH_BINARY
    )

    # Save the motion mask
    output_path = OUTPUT_DIR / f"frame_difference_{frame_count:04d}.jpg"
    cv2.imwrite(str(output_path), motion_mask)

    previous_gray = current_gray
    frame_count += 1

cap.release()

print(f"Processed {frame_count} frames.")
print(f"Motion masks saved in: {OUTPUT_DIR}")

# ---------------- GMM Background Subtraction ----------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

# Create the Gaussian Mixture-based background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)

gmm_frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Apply background subtraction to detect moving foreground
    foreground_mask = background_subtractor.apply(frame)

    output_path = OUTPUT_DIR / f"gmm_mask_{gmm_frame_count:04d}.jpg"
    cv2.imwrite(str(output_path), foreground_mask)

    gmm_frame_count += 1

cap.release()

print(f"GMM processed {gmm_frame_count} frames.")
print(f"GMM masks saved in: {OUTPUT_DIR}")

# ---------------- Lucas-Kanade Optical Flow ----------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

ret, previous_frame = cap.read()

if not ret:
    raise RuntimeError("Could not read the video for optical flow.")

previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

# Detect feature points to track
previous_points = cv2.goodFeaturesToTrack(
    previous_gray,
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

flow_frame_count = 0

while True:
    ret, current_frame = cap.read()

    if not ret:
        break

    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    if previous_points is not None:
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            previous_points,
            None,
            **lk_params
        )

        if next_points is not None:
            good_new = next_points[status.flatten() == 1]
            good_old = previous_points[status.flatten() == 1]

            # Draw tracked motion points and lines
            for new, old in zip(good_new, good_old):
                x_new, y_new = new.ravel().astype(int)
                x_old, y_old = old.ravel().astype(int)

                cv2.line(
                    current_frame,
                    (x_old, y_old),
                    (x_new, y_new),
                    (0, 255, 0),
                    2
                )
                cv2.circle(
                    current_frame,
                    (x_new, y_new),
                    3,
                    (0, 0, 255),
                    -1
                )

            previous_points = good_new.reshape(-1, 1, 2)

    output_path = OUTPUT_DIR / f"optical_flow_{flow_frame_count:04d}.jpg"
    cv2.imwrite(str(output_path), current_frame)

    previous_gray = current_gray
    flow_frame_count += 1

cap.release()

print(f"Lucas-Kanade processed {flow_frame_count} frames.")
print(f"Optical flow images saved in: {OUTPUT_DIR}")