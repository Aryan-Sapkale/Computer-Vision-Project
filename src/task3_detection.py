import cv2
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "dataset" / "images" / "my_photo.jpg"
OUTPUT_DIR = BASE_DIR / "outputs" / "task3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load image
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load OpenCV's pretrained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Draw bounding boxes and labels
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        image,
        "Face",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

# Save output
output_path = OUTPUT_DIR / "haar_face_detection.jpg"
cv2.imwrite(str(output_path), image)

print(f"Faces detected: {len(faces)}")
print(f"Saved result to: {output_path}")

from ultralytics import YOLO

# Load pretrained YOLO model (downloads weights the first time)
model = YOLO("yolo11n.pt")

# Run detection on your photo
results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.25,
    save=False
)

# Draw detected objects and save the result
for result in results:
    yolo_image = result.plot()
    yolo_output_path = OUTPUT_DIR / "yolo_object_detection.jpg"
    cv2.imwrite(str(yolo_output_path), yolo_image)

    print(f"Objects detected: {len(result.boxes)}")
    print(f"YOLO result saved to: {yolo_output_path}")