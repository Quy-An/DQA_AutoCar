import ncnn
import cv2
import numpy as np
import time

# Định nghĩa các lớp (dựa trên tập dữ liệu của bạn)
CLASSES = ["object", "way"]  # Danh sách lớp của bạn
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))  # Màu ngẫu nhiên cho mask

# Hàm vẽ mask và bounding box
def draw_segmentation(image, boxes, masks, class_ids, confidences):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        confidence = confidences[i]
        mask = masks[i]

        # Vẽ bounding box
        color = COLORS[class_id]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        label = f"{CLASSES[class_id]}: {confidence:.2f}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Áp dụng mask
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.uint8) * 255
        mask_colored = np.zeros_like(image[y:y+h, x:x+w])
        mask_colored[:, :, :] = color
        mask_colored = cv2.bitwise_and(mask_colored, mask_colored, mask=mask)
        image[y:y+h, x:x+w] = cv2.addWeighted(image[y:y+h, x:x+w], 0.5, mask_colored, 0.5, 0)

    return image

def main():
    # Đường dẫn đến ảnh và mô hình
    image_path = "/home/quyan/DQA_AutoCar/test_image.jpg"
    param_path = "/home/quyan/DQA_AutoCar/weights/best_ncnn_model/model.ncnn.param"
    bin_path = "/home/quyan/DQA_AutoCar/weights/best_ncnn_model/model.ncnn.bin"
    output_path = "output.jpg"

    # Khởi tạo NCNN
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False  # Không dùng Vulkan trên Raspberry Pi
    net.opt.num_threads = 4  # Sử dụng 4 luồng cho Raspberry Pi 4

    # Tải mô hình NCNN
    try:
        net.load_param(param_path)
        net.load_model(bin_path)
    except Exception as e:
        print(f"Không thể tải mô hình: {e}")
        return

    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return

    img_h, img_w = image.shape[:2]

    # Chuẩn bị dữ liệu đầu vào với kích thước 320x240
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (320, 240), (0, 0, 0), swapRB=True)

    # Đo thời gian suy luận
    start_time = time.time()

    # Tạo extractor và quản lý tài nguyên
    extractor = net.create_extractor()
    try:
        # Thiết lập đầu vào
        input_mat = ncnn.Mat(blob.squeeze(0))
        extractor.input("in0", input_mat)

        # Suy luận
        output = None
        mask_output = None
        ret, output = extractor.extract("out0")  # Đầu ra detections
        if not ret:
            print("Không thể trích xuất đầu ra 'out0'")
            return
        ret, mask_output = extractor.extract("out1")  # Đầu ra mask proto
        if not ret:
            print("Không thể trích xuất đầu ra 'out1'")
            return

        # Tính thời gian xử lý
        inference_time = time.time() - start_time
        fps = 1 / inference_time

        # Xử lý kết quả
        boxes = []
        class_ids = []
        confidences = []
        masks = []

        # Giả sử output có định dạng: [num_detections, (x, y, w, h, conf, class_scores..., mask_proto)]
        try:
            output = np.array(output)
            if output.size == 0:
                print("Đầu ra detections rỗng")
                return
            # Giả sử 32 kênh mask proto, điều chỉnh nếu cần
            output = output.reshape(-1, 5 + len(CLASSES) + 32)
        except Exception as e:
            print(f"Lỗi khi xử lý đầu ra detections: {e}")
            return

        try:
            mask_output = np.array(mask_output)
            if mask_output.size == 0:
                print("Đầu ra mask proto rỗng")
                return
        except Exception as e:
            print(f"Lỗi khi xử lý đầu ra mask proto: {e}")
            return

        for det in output:
            conf = det[4]
            if conf < 0.5:  # Ngưỡng tin cậy
                continue

            scores = det[5:5+len(CLASSES)]
            class_id = np.argmax(scores)
            if scores[class_id] < 0.5:
                continue

            # Tọa độ bounding box, tỷ lệ với kích thước 320x240
            x = int(det[0] * img_w / 320)
            y = int(det[1] * img_h / 240)
            w = int(det[2] * img_w / 320)
            h = int(det[3] * img_h / 240)
            x = max(0, x - w // 2)
            y = max(0, y - h // 2)

            # Xử lý mask
            mask_proto = det[5+len(CLASSES):]
            try:
                mask = np.dot(mask_output, mask_proto)  # Tính mask từ proto
                mask = 1 / (1 + np.exp(-mask))  # Sigmoid
                mask = cv2.resize(mask, (320, 240), interpolation=cv2.INTER_NEAREST)
            except Exception as e:
                print(f"Lỗi khi xử lý mask: {e}")
                continue

            boxes.append([x, y, w, h])
            class_ids.append(class_id)
            confidences.append(float(conf))
            masks.append(mask)

        # Vẽ kết quả
        result_image = draw_segmentation(image.copy(), boxes, masks, class_ids, confidences)

        # Hiển thị thông tin FPS và thời gian xử lý
        print(f"Thời gian suy luận: {inference_time:.4f} giây")
        print(f"FPS: {fps:.2f}")

        # Thêm FPS vào ảnh
        cv2.putText(result_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Lưu ảnh đầu ra
        cv2.imwrite(output_path, result_image)
        print(f"Đã lưu ảnh đầu ra tại {output_path}")

        # Hiển thị ảnh
        cv2.imshow("YOLO11 Segmentation", result_image)
        cv2.waitKey(0)

    except Exception as e:
        print(f"Lỗi trong quá trình xử lý: {e}")
    finally:
        # Đảm bảo giải phóng tài nguyên
        cv2.destroyAllWindows()

    # Giải phóng net (NCNN)
    del net

if __name__ == "__main__":
    main()