import ncnn
import cv2
import numpy as np
import time

# Định nghĩa các lớp (dựa trên COCO hoặc tập dữ liệu của bạn)
CLASSES = ["person", "car", "dog", "cat"]  # Thay bằng danh sách lớp của bạn
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
    # Khởi tạo NCNN
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False  # Không dùng Vulkan trên Raspberry Pi
    net.opt.num_threads = 4  # Sử dụng 4 luồng cho Raspberry Pi 4

    # Tải mô hình NCNN
    net.load_param("yolo11n-seg.param")
    net.load_model("yolo11n-seg.bin")

    # Khởi tạo camera
    cap = cv2.VideoCapture(0)  # Sử dụng camera mặc định
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Không thể mở camera")
        return

    while True:
        start_time = time.time()

        # Đọc frame từ camera
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame")
            break

        # Chuẩn bị dữ liệu đầu vào
        img_h, img_w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), (0, 0, 0), swapRB=True)

        # Tạo extractor
        extractor = ncnn.Extractor(net, net.create_extractor())
        extractor.input("images", ncnn.Mat(blob.squeeze(0)))

        # Suy luận
        _, output = extractor.extract("output")  # Tên đầu ra phụ thuộc vào mô hình
        _, mask_output = extractor.extract("mask_output")  # Đầu ra mask

        # Xử lý kết quả
        boxes = []
        class_ids = []
        confidences = []
        masks = []

        # Giả sử output có định dạng: [num_detections, (x, y, w, h, conf, class_scores...)]
        output = np.array(output).reshape(-1, 5 + len(CLASSES) + 32)  # 32 là số kênh mask proto
        for det in output:
            conf = det[4]
            if conf < 0.5:  # Ngưỡng tin cậy
                continue

            scores = det[5:5+len(CLASSES)]
            class_id = np.argmax(scores)
            if scores[class_id] < 0.5:
                continue

            # Tọa độ bounding box
            x = int(det[0] * img_w / 640)
            y = int(det[1] * img_h / 640)
            w = int(det[2] * img_w / 640)
            h = int(det[3] * img_h / 640)
            x = max(0, x - w // 2)
            y = max(0, y - h // 2)

            # Xử lý mask
            mask_proto = det[5+len(CLASSES):]
            mask = np.dot(mask_output, mask_proto)  # Tính mask từ proto
            mask = 1 / (1 + np.exp(-mask))  # Sigmoid
            mask = cv2.resize(mask, (640, 640), interpolation=cv2.INTER_NEAREST)

            boxes.append([x, y, w, h])
            class_ids.append(class_id)
            confidences.append(float(conf))
            masks.append(mask)

        # Vẽ kết quả
        frame = draw_segmentation(frame, boxes, masks, class_ids, confidences)

        # Tính FPS
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị frame
        cv2.imshow("YOLO11 Segmentation", frame)

        # Thoát nếu nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()