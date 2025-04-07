from pynput import keyboard
import time
from L9110 import Driver

class AutoCarController:
    def __init__(self):
        """
        Initialize an AutoCarController object.

        The object is initialized with a Driver object, and several state variables
        are set to False. The angle of the two servo motors are set to 90 degrees
        and the keyboard listener is set to None.

        """
        self.l9110 = Driver()
        self.w_pressed = False
        self.s_pressed = False
        self.a_pressed = False
        self.d_pressed = False
        self.i_pressed = False
        self.k_pressed = False
        self.j_pressed = False
        self.l_pressed = False
        self.angle_S1 = 90
        self.angle_S2 = 90
        self.listener = None

    def on_press(self, key):
        """
        Set the corresponding state variable to True when a key is pressed.

        :param key: The key that was pressed
        :type key: keyboard.Key or str
        """
        
        try:
            if key.char == 'w':
                self.w_pressed = True
            elif key.char == 's':
                self.s_pressed = True
            elif key.char == 'a':
                self.a_pressed = True
            elif key.char == 'd':
                self.d_pressed = True
            elif key.char == 'i':
                self.i_pressed = True
            elif key.char == 'k':
                self.k_pressed = True
            elif key.char == 'j':
                self.j_pressed = True
            elif key.char == 'l':
                self.l_pressed = True
        except AttributeError:
            pass

    def on_release(self, key):
        """
        Set the corresponding state variable to False when a key is released.

        :param key: The key that was released
        :type key: keyboard.Key or str
        """
        
        try:
            if key.char == 'w':
                self.w_pressed = False
            elif key.char == 's':
                self.s_pressed = False
            elif key.char == 'a':
                self.a_pressed = False
            elif key.char == 'd':
                self.d_pressed = False
            elif key.char == 'i':
                self.i_pressed = False
            elif key.char == 'k':
                self.k_pressed = False
            elif key.char == 'j':
                self.j_pressed = False
            elif key.char == 'l':
                self.l_pressed = False
        except AttributeError:
            pass

        if key == keyboard.Key.esc:
            print("exit")
            return False

    def start(self):
        self.l9110.control_rc(self.l9110.S1, self.angle_S1)
        self.l9110.control_rc(self.l9110.S2, self.angle_S2)
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        print("start")
        print("press W to run forward, S to run backward, A to turn left, D to turn right")
        print("press J or L to control servo S1")
        print("press I or K to control servo S2")
        print("press ESC to exit")

        try:
            while self.listener.is_alive():
                if self.w_pressed:
                    self.l9110.run_forward()
                elif self.s_pressed:
                    self.l9110.run_backward()
                elif self.a_pressed:
                    self.l9110.turn_left()
                elif self.d_pressed:
                    self.l9110.turn_right()
                elif self.i_pressed:
                    self.angle_S2 -= 1
                    self.l9110.control_rc(self.l9110.S2, self.angle_S2)
                elif self.k_pressed:
                    self.angle_S2 += 1
                    self.l9110.control_rc(self.l9110.S2, self.angle_S2)
                elif self.j_pressed:
                    self.angle_S1 += 1
                    self.l9110.control_rc(self.l9110.S1, self.angle_S1)
                elif self.l_pressed:
                    self.angle_S1 -= 1
                    self.l9110.control_rc(self.l9110.S1, self.angle_S1)
                else:
                    self.l9110.stop()
                time.sleep(0.1)
        finally:
            if self.listener:
                self.listener.join()

if __name__ == "__main__":
    controller = AutoCarController()
    controller.start()