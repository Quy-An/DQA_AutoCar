from pynput import keyboard
import time

class KeyController:
    def __init__(self):
        self.w_pressed = False
        self.s_pressed = False
        self.a_pressed = False
        self.d_pressed = False
        self.i_pressed = False
        self.k_pressed = False
        self.j_pressed = False
        self.l_pressed = False
        self.listener = None

    def on_press(self, key):
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

    def start_listener(self):
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        print("start")
        print("press W to run forward, S to run backward, A to turn left, D to turn right")
        print("press J or L to control servo S1")
        print("press I or K to control servo S2")
        print("press ESC to exit")

    def run(self):
        """
        Starts the keyboard listener and runs an infinite loop to monitor keyboard input.

        When a key is pressed, the corresponding action is printed to the console.

        The loop is terminated when the user presses the ESC key.

        :return:
        """
        self.start_listener()
        try:
            while self.listener.is_alive():
                if self.w_pressed:
                    print("forward")
                elif self.s_pressed:
                    print("backward")
                elif self.a_pressed:
                    print("turn left")
                elif self.d_pressed:
                    print("turn right")
                elif self.i_pressed:
                    print("S2 up")
                elif self.k_pressed:
                    print("S2 down")
                elif self.j_pressed:
                    print("S1 turn left")
                elif self.l_pressed:
                    print("S1 turn right")
                time.sleep(0.1)
        finally:
            if self.listener:
                self.listener.join()

if __name__ == "__main__":
    controller = KeyController()
    controller.run()