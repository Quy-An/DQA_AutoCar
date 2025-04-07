from pynput import keyboard
import time
from L9110 import Driver
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

# Initialize L9110 driver
l9110 = Driver()

# Global variables
w_pressed = False  # to run forward
s_pressed = False  # to run backward
a_pressed = False  # to turn left
d_pressed = False  # to turn right

i_pressed = False  # to control servo S2
k_pressed = False  # to control servo S2
j_pressed = False  # to control servo S1
l_pressed = False  # to control servo S1

angle_S1 = 90  # initial angle of servo S1
angle_S2 = 90  # initial angle of servo S2

# Key listener
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

# Key listener
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

    # press ESC to exit
    if key == keyboard.Key.esc:
        print("exit")
        return False

# stop
def stop():
    l9110.control_dc(l9110.MA, 0, l9110.CW)
    l9110.control_dc(l9110.MB, 0, l9110.CW)

# run forward
def run_forward():
    l9110.control_dc(l9110.MA, 100, l9110.CCW)
    l9110.control_dc(l9110.MB, 100, l9110.CCW)

# run backward
def run_backward():
    l9110.control_dc(l9110.MA, 100, l9110.CW)
    l9110.control_dc(l9110.MB, 100, l9110.CW)

# turn left
def turn_left():
    l9110.control_dc(l9110.MA, 100, l9110.CW)
    l9110.control_dc(l9110.MB, 100, l9110.CCW)

# turn right
def turn_right():
    l9110.control_dc(l9110.MA, 100, l9110.CCW)
    l9110.control_dc(l9110.MB, 100, l9110.CW)

# control servo
def control_servo(servo, angle):
    l9110.control_rc(servo, angle)

control_servo(l9110.S1, angle_S1)
control_servo(l9110.S2, angle_S2)

# Gstreamer setup
Gst.init(None)

def on_bus_message(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End of stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, Debug: {debug}")
        loop.quit()
    return True

pipeline_str = (
    "libcamerasrc ! "
    "video/x-raw, width=640, height=480, framerate=30/1 ! "
    "videoconvert ! "
    "videoscale ! "
    "videoflip method=vertical-flip ! "
    "clockoverlay time-format=\"%D %H:%M:%S\" ! "
    "autovideosink"
)

pipeline = Gst.parse_launch(pipeline_str)

bus = pipeline.get_bus()
bus.add_signal_watch()
loop = GLib.MainLoop()

bus.connect("message", on_bus_message, loop)

pipeline.set_state(Gst.State.PLAYING)

# Start listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Print instructions
print("start")
print("press W to run forward, S to run backward, A to turn left, D to turn right")
print("press J or L to control servo S1")
print("press I or K to control servo S2")
print("press ESC to exit")



# Main loop (combined)
try:
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

        # check if gstreamer loop is running
        if not loop.is_running():
            break

    loop.run()

except Exception as e:
    print(f"Error: {e}")
finally:
    pipeline.set_state(Gst.State.NULL)
    listener.join()