# File: ncnn_inference_helper.py

import cv2
import numpy as np
import ncnn # Ensure ncnn Python bindings are installed
import os
import time
# Needed for sigmoid activation
from scipy.special import expit as sigmoid # Using scipy's sigmoid (install scipy: pip install scipy)
# If you don't want to use scipy, you can implement sigmoid manually:
# def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


class NCNN_YOLO_Processor:
    def __init__(self, param_path, bin_path, input_size=320):
        self.net = ncnn.Net()
        self.net.opt.num_threads = 4 # Use multiple threads for inference

        try:
            print(f"Helper: Attempting to load NCNN model from {param_path} and {bin_path}")
            self.net.load_param(param_path)
            self.net.load_model(bin_path)
            print("Helper: NCNN model loaded successfully.")
        except Exception as e:
            print(f"Helper: Error loading NCNN model: {e}")
            print("Helper: Hint: Ensure NCNN Python bindings are correctly installed.")
            raise

        self.input_size = input_size # Model input size (320x320)

        # Verified names from model_ncnn.py
        self.input_name = "in0"
        self.output_names = ["out0", "out1"] # out0: detection/coeffs, out1: prototypes

        # Preprocessing values from Colab output (~1/255.0)
        self.mean_vals = [0.0, 0.0, 0.0]
        self.norm_vals = [0.00392156862745098, 0.00392156862745098, 0.00392156862745098] # Corresponds to 1/255.0

        # Model parameters needed for postprocessing (from YOLOv11-seg.yaml and assumptions)
        # <<< ACTION REQUIRED: VERIFY THESE PARAMETERS BASED ON YOUR SPECIFIC MODEL CONVERSION <<<
        self.num_classes = 2 # Your specific number of classes ("object", "way")
        self.num_mask_coeffs = 32 # From Segment layer args in YAML
        self.num_proto_channels = 256 # From Segment layer args in YAML (number of channels in prototype output)
        self.strides = [8, 16, 32] # Strides for P3, P4, P5 detection heads (needed for decoding)
        # Anchor boxes are CRUCIAL for decoding YOLO outputs. They are NOT in the YAML structure.
        # You MUST find the EXACT anchor box dimensions used during training/conversion for each stride.
        # Example placeholder anchors (replace with actual anchors):
        # This should be a dictionary or list mapping strides to lists/arrays of anchor [width, height].
        # e.g., self.anchor_boxes = {8: np.array([[w1, h1], [w2, h2], [w3, h3]]), 16: ..., 32: ...}
        self.anchor_boxes = {
             8: np.array([[10., 13.], [16., 30.], [33., 23.]]),    # Placeholder anchors for stride 8 (P3)
            16: np.array([[30., 61.], [62., 45.], [59., 119.]]),  # Placeholder anchors for stride 16 (P4)
            32: np.array([[116., 90.], [156., 198.], [373., 326.]]) # Placeholder anchors for stride 32 (P5)
        } # <<< ACTION REQUIRED: REPLACE WITH YOUR MODEL'S ACTUAL ANCHOR BOXES (NumPy array format) <<<
        # You also need to know how the flattened output tensor (38, 2100) maps back to these anchors and strides.

        print(f"Helper: Using input layer name: '{self.input_name}', output layer names: {self.output_names}")
        print(f"Helper: Using preprocessing values: mean={self.mean_vals}, norm={self.norm_vals}")
        print(f"Helper: Postprocessing params: classes={self.num_classes}, mask_coeffs={self.num_mask_coeffs}, proto_channels={self.num_proto_channels}, strides={self.strides}")
        # print(f"Helper: Placeholder anchor boxes (replace): {self.anchor_boxes}")


    def preprocess(self, image_bgr_np):
        """Preprocesses the image (NumPy BGR array) for NCNN input."""
        img_resized = cv2.resize(image_bgr_np, (self.input_size, self.input_size))
        in_mat = ncnn.Mat.from_pixels(img_resized, ncnn.Mat.PixelType.PIXEL_BGR, self.input_size, self.input_size)
        in_mat.substract_mean_normalize(self.mean_vals, self.norm_vals)
        return in_mat

    # --- IMPLEMENT THE POSTPROCESS METHOD BELOW ---
    def postprocess(self, ncnn_outputs, original_width, original_height, conf_threshold, nms_threshold=0.45):
        """
        Postprocesses the raw NCNN output tensors for a YOLOv11 Segmentation model.
        Decodes detection predictions and generates segmentation masks.

        Args:
            ncnn_outputs (list of ncnn.Mat): The raw output tensors from NCNN inference.
                                             Expected: [detection_output_mat, segmentation_output_mat].
                                             - detection_output_mat: Likely 'out0' (shape (38, 2100)).
                                             - segmentation_output_mat: Likely 'out1' (prototype masks).
            original_width (int): Original width of the input image frame.
            original_height (int): Original height of the input image frame.
            conf_threshold (float): Confidence threshold for filtering detections (e.g., 0.5).
            nms_threshold (float): NMS threshold for removing overlapping boxes (e.g., 0.45).

        Returns:
            list: A list of detections, where each detection is a tuple like
                  (x1, y1, x2, y2, conf, class_id, mask_np).
                  Coordinates scaled to original image size.
                  'mask_np' is a NumPy array (H_orig, W_orig) representing the binary mask for that object.
        """
        print("Helper: --- Starting Postprocessing (YOLOv11 Segmentation) ---")
        print("Helper: !!! ACTION REQUIRED: Replace the core decoding/mask generation logic with your actual implementation !!!")
        print(f"Helper: Expected detection output shape (38, 2100). Expected segmentation output shape (256, H_proto, W_proto).")

        if not ncnn_outputs or len(ncnn_outputs) < 2:
            print("Helper: Expected 2 output tensors (detection, segmentation). Received:", len(ncnn_outputs))
            return []

        # --- Step 1: Identify and Convert Output Tensor(s) to NumPy ---
        # Assuming outputs are in order based on model_ncnn.py: [detection_output, segmentation_output]
        detection_output_mat = ncnn_outputs[0] # Likely 'out0' (shape (38, 2100))
        segmentation_output_mat = ncnn_outputs[1] # Likely 'out1' (Prototype masks)

        detection_output_np = np.array(detection_output_mat) # Shape (38, 2100)
        segmentation_output_np = np.array(segmentation_output_mat) # Shape (num_prototypes, H_proto, W_proto) or similar

        print("Helper: Converted outputs to NumPy.")
        print("Helper: Detection output shape (likely out0):", detection_output_np.shape)
        print("Helper: Segmentation output shape (likely out1):", segmentation_output_np.shape)


        # --- Step 2: Implement Detection Decoding Logic (ACTION REQUIRED - YOLOv11 Specific) ---
        # Decode the `detection_output_np` tensor (shape (38, 2100))
        # into a list of potential detections: (x1_320, y1_320, x2_320, y2_320, conf, class_id, mask_coeffs).
        # This is the core mathematical operation to convert raw network output into box/confidence/class/mask_coeff predictions.
        # The exact implementation depends heavily on how the (38, 2100) tensor is structured by the NCNN converter.

        potential_detections_scaled_320 = [] # Store (x1_320, y1_320, x2_320, y2_320, conf, class_id, mask_coeffs_np)

        # --- >>> YOUR YOLOv11 SEGMENTATION DETECTION DECODING IMPLEMENTATION GOES HERE <<< ---
        # This is the most complex part. You need to implement the logic that iterates through
        # detection_output_np (shape (38, 2100)) and correctly extracts/calculates
        # the box coordinates, confidence, class scores, and mask coefficients for each prediction.

        # Key information needed for this step (find in your conversion script/examples):
        # 1. How the (38, 2100) tensor is flattened/structured. It's likely a concatenation of outputs from P3, P4, P5.
        #    The 2100 matches the total grid cells (1600 + 400 + 100). The 38 must be related to the per-prediction attributes.
        #    Maybe (38, 2100) is a transposed/reshaped version of something like (total_predictions, attributes) or (attributes, total_predictions).
        #    If attributes = 39, and total predictions = 2100 * num_anchors (e.g., 3), then total predictions = 6300.
        #    Possible structure might be (39, 6300) or (6300, 39). The shape (38, 2100) doesn't directly fit.
        #    You need to understand how YOUR CONVERTER produced this (38, 2100) shape.

        # 2. The specific formulas to convert raw box outputs (tx, ty, tw, th) to (x,y,w,h) using anchor boxes and grid cell coordinates for each stride (8, 16, 32).
        #    Example formulas often involve `sigmoid` and multiplication/addition with anchors and grid offsets.

        # 3. How to access the raw tx, ty, tw, th, objectness, class scores, and mask coefficients (32 values) for each prediction within the (38, 2100) tensor.

        # Look for YOLOv11 Segmentation NCNN decoding examples (C++ or Python) or your conversion script for the precise loop structure and calculations.

        # --- Example Placeholder (Does NOT decode correctly - REPLACE THIS ENTIRE BLOCK) ---
        print("Helper: Running placeholder for detection decoding. This does NOT decode the (38, 2100) tensor.")
        print("Helper: You MUST replace this entire 'Step 2' section with your actual YOLOv11 decoding logic.")

        # This placeholder simulates receiving decoded boxes (scaled to 320x320) directly based on previous Colab output.
        # Replace with code that iterates through the (38, 2100) tensor and applies decoding formulas.

        # Dummy data simulating output from decoding (replace with real decoded data):
        # Format: (x1_320, y1_320, x2_320, y2_320, conf, class_id, mask_coeffs)
        num_dummy_coeffs = self.num_mask_coeffs # 32
        # Create a few dummy mask coefficients arrays
        dummy_mask_coeffs_list = [np.random.rand(num_dummy_coeffs).astype(np.float32) for _ in range(5)]

        # Example dummy potential detections (replace with real decoded detections):
        # These coordinates are scaled to 320x320 input size.
        # Coordinates are derived from scaling the Colab output coordinates (640x640) down to 320x320.
        scale_orig_to_320 = self.input_size / original_width # 320 / 640 = 0.5
        potential_detections_scaled_320 = [
            (int(226.95 * scale_orig_to_320), int(291.87 * scale_orig_to_320), int(305.34 * scale_orig_to_320), int(408.27 * scale_orig_to_320), 0.9054, 0, dummy_mask_coeffs_list[0]), # Scaled Colab object 1
            (int(204.73 * scale_orig_to_320), int(486.27 * scale_orig_to_320), int(297.50 * scale_orig_to_320), int(637.08 * scale_orig_to_320), 0.7926, 1, dummy_mask_coeffs_list[1]), # Scaled Colab way 1
            (int(182.67 * scale_orig_to_320), int(229.48 * scale_orig_to_320), int(207.00 * scale_320, 0.5832, 0, dummy_mask_coeffs_list[2]), # Scaled Colab object 2
             # Add more real decoded detections from your implementation here
        ]
        # Filter by confidence threshold *after* decoding
        potential_detections_scaled_320 = [det for det in potential_detections_scaled_320 if det[4] > conf_threshold]

        # --- END OF PLACEHOLDER FOR DETECTION DECODING ---
        print(f"Helper: After decoding (or placeholder), {len(potential_detections_scaled_320)} potential boxes above confidence threshold.")


        # --- Step 3: Apply Non-Maximum Suppression (NMS) (ACTION REQUIRED) ---
        # Apply NMS to the potential_detections_scaled_320 based on bounding box overlap and confidence.

        detections_after_nms_scaled_320 = [] # Store detections after NMS
        # --- >>> YOUR NMS IMPLEMENTATION GOES HERE <<< ---
        # Use cv2.dnn.NMSBoxes or a similar implementation on the bounding boxes (elements 0-3).
        # Pass confidences (element 4). Use the conf_threshold and the nms_threshold.
        # Make sure to keep ALL data for detections that survive NMS (x1,y1,x2,y2, conf, class_id, mask_coeffs).

        # Example using OpenCV NMS (Requires correct input format for boxes and confidences):
        confidences = [det[4] for det in potential_detections_scaled_320]
        box_coords_xyxy = [det[:4] for det in potential_detections_scaled_320] # x1, y1, x2, y2 relative to 320x320

        # Check if there are boxes to process before calling NMS
        if box_coords_xyxy:
            # NMSBoxes returns indices of boxes to keep
            # Result format can vary (tuple of arrays, numpy array, etc.)
            indices_to_keep = cv2.dnn.NMSBoxes(box_coords_xyxy, confidences, conf_threshold, nms_threshold)

            # Ensure indices to keep is a flat list of integers
            # Handle potential empty result or different return types from NMSBoxes
            if isinstance(indices_to_keep, tuple):
                 indices_to_keep = indices_to_keep[0].flatten().tolist() if indices_to_keep and isinstance(indices_to_keep[0], np.ndarray) else []
            elif isinstance(indices_to_keep, np.ndarray):
                 indices_to_keep = indices_to_keep.flatten().tolist()
            else: # Fallback for other potential formats, assuming it might be a list of lists or similar
                 flat_indices = []
                 if indices_to_keep:
                     for item in indices_to_keep:
                         if isinstance(item, (list, tuple, np.ndarray)):
                             flat_indices.extend(item)
                         else:
                             flat_indices.append(item)
                 indices_to_keep = [int(i) for i in flat_indices] # Ensure they are integers


            detections_after_nms_scaled_320 = [potential_detections_scaled_320[i] for i in indices_to_keep]
        else:
            detections_after_nms_scaled_320 = [] # No boxes to process NMS

        # --- END OF NMS IMPLEMENTATION ---
        print(f"Helper: After NMS, {len(detections_after_nms_scaled_320)} detections remaining.")


        # --- Step 4: Implement Mask Generation and Scale Results (ACTION REQUIRED) ---
        # For each detection remaining after NMS:
        # 1. Generate the object's segmentation mask using its mask_coeffs and the prototype masks (`segmentation_output_np`).
        # 2. Scale the bounding box coordinates and the generated mask to the original image size (640x640).

        final_detections_with_masks = []
        # --- >>> YOUR MASK GENERATION AND FINAL SCALING IMPLEMENTATION GOES HERE <<< ---
        # Iterate through detections_after_nms_scaled_320
        # For each detection: (x1_320, y1_320, x2_320, y2_320, conf, class_id, mask_coeffs)

        scale_x_orig = original_width / self.input_size # 640 / 320 = 2
        scale_y_orig = original_height / self.input_size # 640 / 320 = 2
        # Get prototype dimensions assuming shape is (num_prototypes, H_proto, W_proto)
        # You need to verify the exact shape of segmentation_output_np
        proto_height, proto_width = 160, 160 # <<< ACTION REQUIRED: VERIFY H_proto, W_proto >>>
        if segmentation_output_np.ndim >= 3:
             proto_height = segmentation_output_np.shape[1]
             proto_width = segmentation_output_np.shape[2]
        else:
             print("Helper: Warning: Segmentation output shape is not as expected for mask generation (ndim < 3).")
             # Fallback to placeholder dimensions, but mask generation will likely fail
             # proto_height, proto_width = 160, 160


        # Reshape prototype masks for matrix multiplication (num_prototypes, H_proto * W_proto) -> (H_proto * W_proto, num_prototypes)
        # Check if segmentation_output_np has the expected number of prototype channels
        if segmentation_output_np.shape[0] != self.num_proto_channels:
             print(f"Helper: Warning: Segmentation output has unexpected number of prototype channels ({segmentation_output_np.shape[0]}). Expected ({self.num_proto_channels}).")
             prototypes_reshaped = None # Cannot proceed safely
        elif segmentation_output_np.ndim < 3:
              print("Helper: Warning: Segmentation output has less than 3 dimensions. Cannot reshape for mask generation.")
              prototypes_reshaped = None
        else:
             try:
                prototypes_reshaped = segmentation_output_np.reshape(self.num_proto_channels, proto_height * proto_width).T # Shape (H_proto*W_proto, num_prototypes=256)
             except ValueError as e:
                 print(f"Helper: Error reshaping prototype masks: {e}. Check segmentation output shape.")
                 prototypes_reshaped = None # Cannot proceed with mask generation


        for det_320 in detections_after_nms_scaled_320:
            # Assuming detection format is (x1_320, y1_320, x2_320, y2_320, conf, class_id, mask_coeffs)
            if len(det_320) < 7 or not isinstance(det_320[6], np.ndarray) or det_320[6].ndim != 1 or det_320[6].shape[0] != self.num_mask_coeffs:
                 print(f"Helper: Warning: Detection format missing or invalid mask coefficients. Skipping mask generation for {det_320}.")
                 # Add a blank mask and scaled box, but warn
                 x1_320, y1_320, x2_320, y2_320, conf, class_id = det_320[:6]
                 x1_orig = int(x1_320 * scale_x_orig); y1_orig = int(y1_320 * scale_y_orig)
                 x2_orig = int(x2_320 * scale_x_orig); y2_orig = int(y2_320 * scale_y_orig)
                 x1_orig, y1_orig = max(0, x1_orig), max(0, y1_orig)
                 x2_orig, y2_orig = min(original_width-1, x2_orig), min(original_height-1, y2_orig)
                 binary_mask_np = np.zeros((original_height, original_width), dtype=np.uint8) # Blank mask
                 final_detections_with_masks.append((x1_orig, y1_orig, x2_orig, y2_orig, conf, class_id, binary_mask_np))
                 continue

            x1_320, y1_320, x2_320, y2_320, conf, class_id, mask_coeffs = det_320 # mask_coeffs is np.ndarray (32,)

            # 1. Generate Raw Mask (ACTION REQUIRED)
            # Combine mask_coeffs (shape (num_mask_coeffs,)) with prototype masks (prototypes_reshaped shape (H_proto*W_proto, num_prototypes))
            # This is typically matrix multiplication: np.dot(prototypes_reshaped, mask_coeffs)
            # The result shape before reshaping should be (H_proto * W_proto,).
            # Reshape the result back to (H_proto, W_proto).

            raw_mask = None # Initialize raw mask

            if prototypes_reshaped is None:
                 print("Helper: Warning: Prototype masks not correctly prepared. Cannot generate mask.")
                 # raw_mask will remain None
            else:
                 try:
                     # >>> YOUR MATRIX MULTIPLICATION LOGIC HERE <<<
                     # Example: raw_mask_flattened = np.dot(prototypes_reshaped, mask_coeffs) # Shape (H_proto*W_proto,)
                     # Reshape to (H_proto, W_proto): raw_mask = raw_mask_flattened.reshape(proto_height, proto_width)

                     # Placeholder raw mask (replace with real calculation):
                     print("Helper: Using placeholder raw mask (replace with real calculation).")
                     raw_mask = np.random.rand(proto_height, proto_width).astype(np.float32) # Replace this


                 except Exception as e:
                     print(f"Helper: Error during raw mask calculation: {e}. Skipping mask generation.")
                     raw_mask = None


            # 2. Apply Sigmoid Activation (if raw_mask was generated)
            if raw_mask is not None:
                 activated_mask = sigmoid(raw_mask) # Values between 0 and 1
            else:
                 activated_mask = np.zeros((proto_height, proto_width), dtype=np.float32) # Blank mask if calculation failed


            # 3. Resize Mask to Original Image Size
            # Interpolation=cv2.INTER_LINEAR is common for masks
            mask_orig_size = cv2.resize(activated_mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

            # 4. Threshold Mask to get Binary Mask
            # Threshold value is often 0.5, but can be optimized.
            binary_mask_np = (mask_orig_size > 0.5).astype(np.uint8) * 255 # Binary mask (0 or 255)


            # 5. Scale Box Coordinates back to Original Image Size
            x1_orig = int(x1_320 * scale_x_orig)
            y1_orig = int(y1_320 * scale_y_orig)
            x2_orig = int(x2_320 * scale_x_orig)
            y2_orig = int(y2_320 * scale_y_orig)

            # Clamp coordinates to original image boundaries
            x1_orig, y1_orig = max(0, x1_orig), max(0, y1_orig)
            x2_orig, y2_orig = min(original_width-1, x2_orig), min(original_height-1, y2_orig)

            # Store as (x1, y1, x2, y2, conf, class_id, mask_np) scaled to original image size
            final_detections_with_masks.append((x1_orig, y1_orig, x2_orig, y2_orig, conf, class_id, binary_mask_np))

        # --- END OF MASK GENERATION AND FINAL SCALING IMPLEMENTATION ---


        print(f"Helper: Postprocessing finished. Returning {len(final_detections_with_masks)} final detections (scaled to original size, with generated masks).")

        # Return the list of detections including masks (scaled to original image size)
        return final_detections_with_masks

    # --- (predict method remains the same) ---
    def predict(self, image_bgr_np, conf_threshold):
        """
        Runs inference on a single BGR NumPy image using the loaded NCNN model.
        Returns detections including masks, scaled to original image size.
        """
        original_height, original_width = image_bgr_np.shape[:2]
        in_mat = self.preprocess(image_bgr_np)
        ex = self.net.create_extractor()
        ex.set_num_threads(4)

        ret_in = ex.input(self.input_name, in_mat)
        if ret_in != 0: print(f"Helper: Error setting input '{self.input_name}': {ret_in}"); return []

        output_mats = []
        # Extract 'out0'
        ret_out0, out0_mat = ex.extract(self.output_names[0])
        if ret_out0 == 0: output_mats.append(out0_mat)
        else: print(f"Helper: Error extracting '{self.output_names[0]}': {ret_out0}")

        # Extract 'out1'
        ret_out1, out1_mat = ex.extract(self.output_names[1])
        if ret_out1 == 0: output_mats.append(out1_mat)
        else: print(f"Helper: Error extracting '{self.output_names[1]}', return code: {ret_out1}")

        if not output_mats: print("Helper: No output tensors extracted."); return []

        # Pass list of output Mats to postprocess
        # Pass conf_threshold and nms_threshold (default value used if not passed)
        detections_with_masks = self.postprocess(output_mats, original_width, original_height, conf_threshold)

        return detections_with_masks