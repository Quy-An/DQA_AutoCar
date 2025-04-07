from pynput import keyboard
import time
from L9110 import Driver
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import threading

# Khởi tạo GStreamer
Gst.init(None)

# Hàm chạy pipeline GStreamer trong luồng riêng
def start_gstreamer():
    # Tạo pipeline
    pipeline = Gst.Pipeline()
    
    # Các phần tử trong pipeline
    source = Gst.ElementFactory.make("libcamerasrc", "source")
    capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
    convert = Gst.ElementFactory.make("videoconvert", "convert")
    encoder = Gst.ElementFactory.make("x264enc", "encoder")
    payloader = Gst.ElementFactory.make("rtph264pay", "payloader")
    sink = Gst.ElementFactory.make("udpsink", "sink")

    if not all([pipeline, source, capsfilter, convert, encoder, payloader, sink]):
        print("Không thể tạo pipeline GStreamer")
        return

    # Thêm phần tử vào pipeline
    pipeline.add(source)
    pipeline.add(capsfilter)
    pipeline.add(convert)
    pipeline.add(encoder)
    pipeline.add(payloader)
    pipeline.add(sink)

    # Liên kết các phần tử
    source.link(capsfilter)
    capsfilter.link(convert)
    convert.link(encoder)
    encoder.link(payloader)
    payloader.link(sink)

    # Cấu hình pipeline
    caps = Gst.Caps.from_string("video/x-raw,width=640,height=480,framerate=30/1")
    capsfilter.set_property("caps", caps)
    encoder.set_property("tune", "zerolatency")
    encoder.set_property("bitrate", 500)
    sink.set_property("host", "IP_CUA_MAY_TINH")  # Thay bằng IP của máy tính
    sink.set_property("port", 5000)

    # Xử lý thông điệp bus
    def on_bus_message(bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("Kết thúc stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Lỗi GStreamer: {err}, Debug: {debug}")
            loop.quit()
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    loop = GLib.MainLoop()
    bus.connect("message", on_bus_message, loop)

    # Chạy pipeline
    pipeline.set_state(Gst.State.PLAYING)
    print("Bắt đầu truyền video qua GStreamer...")
    try:
        loop.run()
    except Exception as e:
        print(f"Lỗi trong GStreamer: {e}")
    finally:
        pipeline.set_state(Gst.State.NULL)

# Khởi tạo driver L9110
l9110 = Driver()

# Biến toàn cục
w_pressed = False
s_pressed = False
a_pressed = False
d_pressed = False
i_pressed = False
k_pressed = False
j_pressed = False
l_pressed = False

angle_S1 = 90
angle_S2 = 90

# Xử lý sự kiện nhấn phím
def on_press(key):
    global w_pressed, s_pressed, a_pressed, d_pressed, i_pressed, k_pressed, j_pressed, l_pressed
    try:
        if key.char == 'w':
            w_pressed = True
        elif key.char == 's':
            s_pressed = True
        elif key.char == 'a':
            a_pressed = True
        elif key.char == 'd':
            d_pressed = True
        elif key.char == 'i':
            i_pressed = True
        elif key.char == 'k':
            k_pressed = True
        elif key.char == 'j':
            j_pressed = True
        elif key.char == 'l':
            l_pressed = True
    except AttributeError:
        pass

# Xử lý sự kiện nhả phím
def on_release(key):
    global w_pressed, s_pressed, a_pressed, d_pressed, i_pressed, k_pressed, j_pressed, l_pressed
    try:
        if key.char == 'w':
            w_pressed = False
        elif key.char == 's':
            s_pressed = False
        elif key.char == 'a':
            a_pressed = False
        elif key.char == 'd':
            d_pressed = False
        elif key.char == 'i':
            i_pressed = False
        elif key.char == 'k':
            k_pressed = False
        elif key.char == 'j':
            j_pressed = False
        elif key.char == 'l':
            l_pressed = False
    except AttributeError:
        pass
    if key == keyboard.Key.esc:
        print("Thoát chương trình")
        return False

# Hàm điều khiển xe
def stop():
    l9110.control_dc(l9110.MA, 0, l9110.CW)
    l9110.control_dc(l9110.MB, 0, l9110.CW)

def run_forward():
    l9110.control_dc(l9110.MA, 100, l9110.CCW)
    l9110.control_dc(l9110.MB, 100, l9110.CCW)

def run_backward():
    l9110.control_dc(l9110.MA, 100, l9110.CW)
    l9110.control_dc(l9110.MB, 100, l9110.CW)

def turn_left():
    l9110.control_dc(l9110.MA, 100, l9110.CW)
    l9110.control_dc(l9110.MB, 100, l9110.CCW)

def turn_right():
    l9110.control_dc(l9110.MA, 100, l9110.CCW)
    l9110.control_dc(l9110.MB, 100, l9110.CW)

def control_servo(servo, angle):
    l9110.control_rc(servo, angle)

# Khởi tạo góc servo ban đầu
control_servo(l9110.S1, angle_S1)
control_servo(l9110.S2, angle_S2)

# Chạy GStreamer trong luồng riêng
gstreamer_thread = threading.Thread(target=start_gstreamer)
gstreamer_thread.daemon = True  # Luồng sẽ dừng khi chương trình chính dừng
gstreamer_thread.start()

# Khởi tạo listener bàn phím
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# In hướng dẫn
print("Khởi động AutoCar")
print("Nhấn W để tiến, S để lùi, A để trái, D để phải")
print("Nhấn J hoặc L để điều khiển servo S1")
print("Nhấn I hoặc K để điều khiển servo S2")
print("Nhấn ESC để thoát")

# Vòng lặp chính
while listener.is_alive():
    if w_pressed:
        run_forward()
    elif s_pressed:
        run_backward()
    elif a_pressed:
        turn_left()
    elif d_pressed:
        turn_right()
    elif i_pressed:
        angle_S2 -= 1
        control_servo(l9110.S2, angle_S2)
    elif k_pressed:
        angle_S2 += 1
        control_servo(l9110.S2, angle_S2)
    elif j_pressed:
        angle_S1 += 1
        control_servo(l9110.S1, angle_S1)
    elif l_pressed:
        angle_S1 -= 1
        control_servo(l9110.S1, angle_S1)
    else:
        stop()
    time.sleep(0.1)

# Dừng listener
listener.join()