from pynput import keyboard
import time
from L9110 import Driver

# initialize L9110 driver
l9110 = Driver()

# global variables
w_pressed = False           # to run forward
s_pressed = False           # to run backward
a_pressed = False           # to turn left
d_pressed = False           # to turn right

i_pressed = False           # to control servo S2 
k_pressed = False           # to control servo S2
j_pressed = False           # to control servo S1
l_pressed = False           # to control servo S1

angle_S1 = 90               # initial angle of servo S1
angle_S2 = 90               # initial angle of servo S2

# key listener
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
    
# key listener
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
    l9110.control_dc(l9110.MA, 100, l9110.CW)
    l9110.control_dc(l9110.MB, 100, l9110.CW)
    
# run backward
def run_backward():
    l9110.control_dc(l9110.MA, 100, l9110.CCW)
    l9110.control_dc(l9110.MB, 100, l9110.CCW)
    
# turn left
def turn_left():
    l9110.control_dc(l9110.MA, 100, l9110.CCW)
    l9110.control_dc(l9110.MB, 100, l9110.CW)

# turn right    
def turn_right():
    l9110.control_dc(l9110.MA, 100, l9110.CW)    
    l9110.control_dc(l9110.MB, 100, l9110.CCW)  
    
# control servo 
def control_servo(servo, angle):
    l9110.control_rc(servo, angle)

# start listener 
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# print instructions
print("start")
print("press W to run forward, S to run backward, A to turn left, D to turn right")
print("press J or L to control servo S1")
print("press I or K to control servo S2")
print("press ESC to exit")

# main loop
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
        angle_S2 += 1
        control_servo(l9110.S2, angle_S2)
    elif k_pressed:
        angle_S2 -= 1
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

# stop listener
listener.join()