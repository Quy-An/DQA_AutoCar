import cv2
from ultralytics import YOLO
import numpy as np

# Load the trained YOLOv8 model
model = YOLO('C:/Users/TANTAI/PycharmProjects/Demo/runs/detect/train44/weights/best.pt')  # Replace with the path to your trained model

# Open video file or capture device (0 for webcam)
video_path = 'C:/Users/TANTAI/PycharmProjects/Demo/dataset1/ketqua/test2.mp4'  # Replace with your video file path or use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create a resizable window and set its size
window_name = 'YOLOv8 Detection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 700, 700)  # Set window size to 1280x720 (adjust as needed)

# Define output video writer
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Process results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Class IDs

        for box, score, cls in zip(boxes, scores, classes):
            if score > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]} {score:.2f}"

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

    # Display the frame (optional, comment out if not needed)
    cv2.imshow('YOLOv8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video processing completed. Output saved as 'output_video.mp4'")