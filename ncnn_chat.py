import cv2
import numpy as np
import time
import ncnn

# Load mô hình
net = ncnn.Net()
# Có thể thêm opt settings ở đây nếu cần, ví dụ:
# net.opt.use_vulkan_compute = False # Nếu không dùng Vulkan
# net.opt.num_threads = 4 # Số luồng cho Raspberry Pi

# Tải mô hình NCNN
try:
    net.load_param("/home/quyan/DQA_AutoCar/weights/best_ncnn_model/model.ncnn.param")
    net.load_model("/home/quyan/DQA_AutoCar/weights/best_ncnn_model/model.ncnn.bin")
    print("Đã tải mô hình thành công.")
except Exception as e:
    print(f"Không thể tải mô hình: {e}")
    exit(1)


# Đọc ảnh từ đường dẫn
image_path = "/home/quyan/DQA_AutoCar/test_image.jpg"
frame = cv2.imread(image_path)

if frame is None:
    print(f"Không thể đọc ảnh từ {image_path}. Kiểm tra đường dẫn.")
    exit(1)

# Kích thước input theo model (đảm bảo match với model của bạn)
input_w, input_h = 320, 240

# Tên layer (đảm bảo match với file .param của model)
input_layer = "in0" # Kiểm tra lại tên input layer trong file .param
output_layer = "out0" # Kiểm tra lại tên output layer trong file .param

orig_h, orig_w = frame.shape[:2]

# Resize và tiền xử lý ảnh
resized = cv2.resize(frame, (input_w, input_h))
# Chuyển BGR sang RGB nếu model được train với RGB, ngược lại giữ nguyên BGR
# img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) # Uncomment nếu cần RGB
img = resized.astype(np.float32) / 255.0 # Chuẩn hóa về [0, 1]

# Chuyển từ HWC sang CHW, copy để đảm bảo liên tục trong bộ nhớ (cần cho ncnn.Mat)
img = img.transpose((2, 0, 1)).copy()  # CHW (Channels, Height, Width)

# --- DÒNG BỊ LỖI ĐÃ SỬA ---
# Tạo ncnn.Mat từ mảng NumPy CHW
# Cách 1 (Phổ biến và có khả năng cao là đúng):
in_mat = ncnn.Mat(img)

# Cách 2 (Nếu cách 1 không hoạt động, thử cách này - ít phổ biến hơn cho CHW numpy):
# in_mat = ncnn.Mat(img.shape[2], img.shape[1], img.shape[0], data=img.data)
# ---------------------------


# Inference và đo thời gian
start = time.time()

ex = net.create_extractor()
# Đặt các tùy chọn cho extractor (nếu có)
# ex.set_num_threads(4)
# ex.set_light_mode(True) # Chế độ nhẹ hơn, có thể nhanh hơn

print(f"Setting input layer: {input_layer}")
ex.input(input_layer, in_mat)

print(f"Extracting output layer: {output_layer}")
ret, out = ex.extract(output_layer)
if not ret:
    print(f"Extract thất bại từ layer '{output_layer}'. Kiểm tra lại tên layer hoặc model.")
    exit(1)
print(f"Extract thành công. Output Mat shape: {out.shape()}") # In shape ncnn.Mat

inference_time = time.time() - start

# Xử lý output
# Chuyển ncnn.Mat sang mảng NumPy. to_numpy() thường trả về HWC hoặc CHW tùy thuộc binding
out_np = out.to_numpy() # Shape: (H, W, C) hoặc (C, H, W)
print(f"Output numpy shape after to_numpy(): {out_np.shape}") # In shape NumPy

# Dựa trên code xử lý mask của bạn (argmax theo axis 0) và cách bạn dùng
# out_np.shape[0] = C, có vẻ như out_np đang ở định dạng (C, H, W).
# Xác nhận lại shape thực tế bạn in ra ở trên.

# Giả định out_np có shape (C, H, W)
C_out, H_out, W_out = out_np.shape

# Tạo mask phân đoạn: argmax theo trục lớp (axis=0 nếu là CHW)
# Nếu out_np có shape (H, W, C), thì axis=2
try:
    if C_out not in out_np.shape:
         print(f"Warning: Assuming output shape is (C, H, W), but {C_out} (C) is not in shape {out_np.shape}. Check axis for argmax.")
    pred_mask = np.argmax(out_np, axis=0).astype(np.uint8)
    confidence_map = np.max(out_np, axis=0)
    avg_confidence = float(np.mean(confidence_map))
    print(f"pred_mask shape: {pred_mask.shape}")
    print(f"confidence_map shape: {confidence_map.shape}")
except Exception as e:
    print(f"Lỗi khi xử lý argmax/max trên output numpy: {e}. Có thể shape output không như mong đợi.")
    exit(1)


# Resize mask về kích thước gốc và hiển thị màu
# Resize pred_mask (shape H_out, W_out) về kích thước gốc (orig_w, orig_h)
resized_pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

# Áp dụng colormap. cv2.applyColorMap mong đợi ảnh 1 kênh (H, W) uint8
# rescale pred_mask để map màu (ví dụ: từ 0 đến C-1 -> 0 đến 255)
# Sử dụng kích thước C_out (số lớp) để scale màu
mask_scaled_for_color = (resized_pred_mask * 255 // (C_out-1) if C_out > 1 else resized_pred_mask).astype(np.uint8)

mask_color = cv2.applyColorMap(mask_scaled_for_color, cv2.COLORMAP_JET)

# Blend ảnh gốc và mask màu
blended = cv2.addWeighted(frame, 0.5, mask_color, 0.5, 0)

# Hiển thị overlay text
fps = 1.0 / inference_time if inference_time > 0 else 0.0
cv2.putText(blended, f"FPS: {fps:.2f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(blended, f"Confidence: {avg_confidence:.2f}", (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Hiển thị ảnh kết quả
cv2.imshow("YOLOv11 Segmentation", blended)
print("Nhấn phím bất kỳ để thoát.")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Giải phóng net (NCNN) - Python binding sẽ tự xử lý khi object bị garbage collected,
# nhưng gọi tường minh cũng không hại gì nếu bạn muốn chắc chắn.
del net

# del ex # Có thể cần nếu extractor được tạo trong vòng lặp và gây rò rỉ