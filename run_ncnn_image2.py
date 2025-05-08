# File: run_ncnn_image.py

import cv2
import os
import sys
import numpy as np # Import numpy for mask handling

# Import your NCNN inference helper module
try:
    from ncnn_inference_helper2 import NCNN_YOLO_Processor
    print("Successfully imported NCNN_YOLO_Processor from ncnn_inference_helper.py")
except ImportError:
    print("-" * 50)
    print("Error: Could not import NCNN_YOLO_Processor from ncnn_inference_helper.py.")
    print("Please ensure the file exists and contains the NCNN_YOLO_Processor class.")
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
image_path = "/home/quyan/DQA_AutoCar/test_image.jpg" # <-- CHANGE THIS PATH!

# Configuration
confidence_threshold = 0.5
image_size = 320 # <<< QUAN TRONG: PHAI LA 320, KHONG PHAI 240 >>>

# --- Class Names ---
CLASSES = ["object", "way"] # Your specific class names
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3)) # Random colors for masks


# --- Check file existence ---
if not os.path.exists(param_file): print(f"Error: NCNN param file not found at {param_file}"); exit()
if not os.path.exists(bin_file): print(f"Error: NCNN bin file not found at {bin_file}"); exit(); exit()
if not os.path.exists(image_path): print(f"Error: Input image file not found at {image_path}"); exit(); exit()


# --- Load NCNN Model Processor ---
print("Initializing NCNN Model Processor...")
try:
    # Create an instance of our NCNN_YOLO_Processor class from the helper file
    # Pass the model file paths and input size (320)
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
detections_with_masks = ncnn_processor.predict(image_bgr, confidence_threshold)

print(f"Inference finished. Found {len(detections_with_masks)} detections above confidence threshold.")

# --- Draw Detections on the Image ---
print("Drawing detections...")

# Create a copy of the original image to draw detections and masks on
result_image = image_bgr.copy()
# For mask blending, often helpful to have a version to overlay on
mask_overlay = result_image.copy()
alpha = 0.5 # Transparency factor for mask


for det in detections_with_masks:
    # Each detection is expected to be (x1_orig, y1_orig, x2_orig, y2_orig, conf, class_id, mask_np)
    # where _orig indicates coordinates scaled to the ORIGINAL image size (e.g., 640x640)
    # mask_np is a binary mask scaled to original image size (original_height, original_width)
    if len(det) < 7: # Check for the expected 7 elements
        print(f"Warning: Skipping detection with unexpected format (length < 7): {det}")
        continue

    x1_orig, y1_orig, x2_orig, y2_orig, conf, class_id, mask_np = det

    # Ensure coordinates are integers and within bounds of the original image
    x1_orig, y1_orig = max(0, int(x1_orig)), max(0, int(y1_orig))
    x2_orig, y2_orig = min(original_width-1, int(x2_orig)), min(original_height-1, int(y2_orig))


    # Get class name and color
    try:
        class_id_int = int(class_id) # Ensure class_id is integer
        class_name = CLASSES[class_id_int] if 0 <= class_id_int < len(CLASSES) else f"class {class_id_int}"
        color = COLORS[class_id_int % len(CLASSES)] # Get color based on class ID
    except (ValueError, TypeError, IndexError):
         print(f"Warning: Invalid class_id encountered: {class_id}. Using raw value and default color.")
         class_name = f"class {class_id}" # Fallback to showing raw value
         color = (100, 100, 100) # Default grey color


    # --- VẼ MASK LÊN ẢNH GỐC ---
    # mask_np là mask nhị phân có kích thước (original_height, original_width)
    # Áp dụng mask lên lớp phủ màu
    if mask_np is not None and mask_np.shape == (original_height, original_width) and mask_np.dtype == np.uint8:
        # Tạo một lớp màu cho mask tại vị trí của đối tượng
        colored_mask = np.zeros_like(result_image, dtype=np.uint8)
        colored_mask[mask_np > 0] = color # Chỉ tô màu những pixel trong mask
        
        # Pha trộn lớp màu mask lên ảnh gốc
        cv2.addWeighted(colored_mask, alpha, mask_overlay, 1 - alpha, 0, mask_overlay)
    else:
         print(f"Helper: Warning: Invalid mask data for detection: {mask_np}. Skipping mask drawing.")

    # --- VẼ BOUNDING BOX ---
    cv2.rectangle(result_image, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)

    # --- VẼ NHÃN ---
    label = f"{class_name}: {conf:.2f}"
    font_scale = 0.6
    font_thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

    # Vị trí nhãn (trên ảnh gốc)
    label_x_orig = x1_orig
    label_y_orig = y1_orig - 10
    if label_y_orig < text_height: # Nếu nhãn quá sát mép trên
        label_y_orig = y2_orig + text_height + baseline

     # Vẽ nền cho văn bản (tùy chọn)
    text_bg_x1 = label_x_orig
    text_bg_y1 = label_y_orig - text_height - baseline
    text_bg_x2 = label_x_orig + text_width
    text_bg_y2 = label_y_orig

    # Đảm bảo nền văn bản nằm trong ảnh
    text_bg_x1, text_bg_y1 = max(0, text_bg_x1), max(0, text_bg_y1)
    text_bg_x2, text_bg_y2 = min(original_width, text_bg_x2), min(original_height, text_bg_y2)

    if text_bg_x1 < text_bg_x2 and text_bg_y1 < text_bg_y2:
         cv2.rectangle(result_image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1) # Nền màu

    cv2.putText(result_image, label, (label_x_orig, label_y_orig), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness) # Chữ màu đen

# Pha trộn lớp phủ mask cuối cùng lên ảnh kết quả có box
result_image = cv2.addWeighted(result_image, 1 - alpha, mask_overlay, alpha, 0)


print("Drawing complete.")

# --- Display the Result ---
# Hiển thị ảnh kết quả (kích thước gốc)
print("Displaying result image (original size). Press any key to close.")
cv2.imshow("YOLO11 Segmentation Result", result_image)

# Wait for a key press indefinitely
cv2.waitKey(0)

# --- Cleanup ---
cv2.destroyAllWindows()

print("Program finished.")