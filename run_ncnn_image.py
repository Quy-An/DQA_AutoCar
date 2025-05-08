# File: run_ncnn_image.py

import cv2
import os
import sys # Needed to potentially add helper directory to path

# Import your NCNN inference helper module
# Ensure ncnn_inference_helper.py is in the same directory as this script,
# or add its directory to sys.path before importing.
# Example if helper is in a subdirectory called 'helpers':
# sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))
try:
    from ncnn_inference_helper import NCNN_YOLO_Processor # Adjust import name if needed
    print("Successfully imported NCNN_YOLO_Processor from ncnn_inference_helper.py")
except ImportError:
    print("-" * 50)
    print("Error: Could not import NCNN_YOLO_Processor from ncnn_inference_helper.py.")
    print("Please ensure:")
    print("1. A file named 'ncnn_inference_helper.py' exists.")
    print("2. It contains the 'NCNN_YOLO_Processor' class definition.")
    print("3. The file is in the same directory as this script, OR")
    print(f"4. The directory containing '{os.path.basename(__file__)}' is added to sys.path if the helper is elsewhere.")
    print("-" * 50)
    exit()
except Exception as e:
     print(f"An unexpected error occurred during import: {e}")
     exit()


# --- Script Configuration ---

# The directory containing your NCNN model files (.param, .bin)
model_dir = "/home/quyan/DQA_AutoCar/weights/best_ncnn_model"
param_file = os.path.join(model_dir, "model.ncnn.param")
bin_file = os.path.join(model_dir, "model.ncnn.bin")

# Path to the image file you want to test on
image_path = "/home/quyan/DQA_AutoCar/anh_103_jpg.rf.9f7e32833b59aa1cb2338440a0021a17.jpg" # <-- CHANGE THIS PATH!

# Configuration (matching your original script)
confidence_threshold = 0.5
image_size = 320 # This is the input size your NCNN model expects

# --- Class Names ---
# You need a list of class names corresponding to the class IDs output by your model.
# The order must match the class IDs (e.g., class ID 0 is class_names[0]).
# Based on your Colab output, class IDs 0 and 1 correspond to "object" and "way".
class_names = [
    "object", "way" # <<< VERIFIED from your Colab output
]
# If your model detects more classes, add them here in the correct order.
# Example for 80 COCO classes:
# class_names = [...] # List of 80 COCO class names


# --- Check file existence ---
if not os.path.exists(param_file):
    print(f"Error: NCNN param file not found at {param_file}")
    exit()
if not os.path.exists(bin_file):
    print(f"Error: NCNN bin file not found at {bin_file}")
    exit()
if not os.path.exists(image_path):
    print(f"Error: Input image file not found at {image_path}")
    exit()


# --- Load NCNN Model Processor ---
print("Initializing NCNN Model Processor...")
try:
    # Create an instance of our NCNN_YOLO_Processor class from the helper file
    ncnn_processor = NCNN_YOLO_Processor(param_file, bin_file, input_size=image_size)
    print("NCNN Processor initialized successfully.")
except Exception as e:
     print(f"Failed to initialize NCNN Processor: {e}")
     print("Please check the error message above and ensure NCNN Python bindings are installed and the helper file is correct.")
     exit()


# --- Load the Input Image ---
print(f"Loading image from {image_path}...")
image_bgr = cv2.imread(image_path) # Read image using OpenCV (in BGR format)

if image_bgr is None:
    print(f"Error: Could not load image from {image_path}. Check the path and file integrity.")
    exit()

original_height, original_width = image_bgr.shape[:2]
print(f"Image loaded: {original_width}x{original_height}")

# --- Run Inference ---
print(f"Running inference on the image with confidence threshold {confidence_threshold}...")
# Call the predict method of our processor from the helper file
# This method handles preprocessing, running NCNN inference, and postprocessing
detections = ncnn_processor.predict(image_bgr, confidence_threshold)

print(f"Inference finished. Found {len(detections)} detections above confidence threshold.")

# --- Draw Detections on the Image ---
print("Drawing detections...")

# Create a resized copy of the original image (320x320) to draw on
# This matches the request to have the output image size at 320x320
drawing_image = cv2.resize(image_bgr, (image_size, image_size))

# Calculate scaling factors from original size to drawing size (image_size x image_size)
# Detections are returned scaled to ORIGINAL size, so we scale them DOWN to drawing size
scale_x_draw = image_size / original_width
scale_y_draw = image_size / original_height


for det in detections:
    # Each detection is expected to be (x1_orig, y1_orig, x2_orig, y2_orig, conf, class_id)
    # where _orig indicates coordinates scaled to the ORIGINAL image size (e.g., 640x640)
    x1_orig, y1_orig, x2_orig, y2_orig, conf, class_id = det

    # Scale coordinates down to the drawing image size (320x320)
    x1_draw = int(x1_orig * scale_x_draw)
    y1_draw = int(y1_orig * scale_y_draw)
    x2_draw = int(x2_orig * scale_x_draw)
    y2_draw = int(y2_orig * scale_y_draw)

    # Ensure coordinates are integers and within bounds of the drawing image (0 to image_size-1)
    x1_draw, y1_draw = max(0, x1_draw), max(0, y1_draw)
    x2_draw, y2_draw = min(image_size - 1, x2_draw), min(image_size - 1, y2_draw)


    # Get class name from the class_names list using the class_id
    # Ensure class_id is a valid index
    class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class {class_id}"

    # Define color for bounding box (e.g., Green in BGR)
    color = (0, 255, 0) # BGR color

    # Draw bounding box on the drawing_image (which is 320x320)
    cv2.rectangle(drawing_image, (x1_draw, y1_draw), (x2_draw, y2_draw), color, 2)

    # Create label text (e.g., "object: 0.91")
    label = f"{class_name}: {conf:.2f}"

    # Calculate text size to position the label
    # Use a smaller font scale appropriate for the 320x320 image
    font_scale = 0.4
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

    # Determine label position (relative to the scaled coordinates)
    label_y_draw = y1_draw - 5 # Position slightly above the box
    if label_y_draw < text_height: # If label is too close to the top edge of the drawing image
        label_y_draw = y2_draw + text_height + baseline # Position below the box

     # Draw background rectangle for text (optional, improves readability)
    text_bg_x1 = x1_draw
    text_bg_y1 = label_y_draw - text_height - baseline
    text_bg_x2 = x1_draw + text_width
    text_bg_y2 = label_y_draw

    # Ensure text background coordinates are within bounds of the drawing image
    text_bg_x1, text_bg_y1 = max(0, text_bg_x1), max(0, text_bg_y1)
    text_bg_x2, text_bg_y2 = min(image_size, text_bg_x2), min(image_size, text_bg_y2)

    # Draw background only if it's a valid rectangle
    if text_bg_x1 < text_bg_x2 and text_bg_y1 < text_bg_y2:
         cv2.rectangle(drawing_image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)

    # Put label text on the drawing_image
    cv2.putText(drawing_image, label, (x1_draw, label_y_draw), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1) # Black text

print("Drawing complete.")

# --- Display the Result ---
# Display the drawing_image (which is 320x320)
print("Displaying result image (320x320). Press any key to close.")
cv2.imshow("NCNN Object Detection Result (320x320)", drawing_image)

# Wait for a key press indefinitely
cv2.waitKey(0)

# --- Cleanup ---
cv2.destroyAllWindows()

print("Program finished.")