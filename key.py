from pynput import keyboard
import time

# global variables
w_pressed = False           # to run forward
s_pressed = False           # to run backward
a_pressed = False           # to turn left
d_pressed = False           # to turn right

i_pressed = False           # to control servo S2 
k_pressed = False           # to control servo S2
j_pressed = False           # to control servo S1
l_pressed = False           # to control servo S1

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
        print("forward")
    elif s_pressed:
        print("backward")
    elif a_pressed:
        print("turn left")
    elif d_pressed:
        print("turn right")
    elif i_pressed:
        print("S2 up")
    elif k_pressed:
        print("S2 down")
    elif j_pressed:
        print("S1 turn left")
    elif l_pressed:
        print("S1 turn right")
    time.sleep(0.1)

# stop listener
listener.join()