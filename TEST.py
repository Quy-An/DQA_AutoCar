from CAMERA2 import CameraStreamer
from CONTROLLER import AutoCarController
import threading
from INA219 import INA219  # Import INA219 class
import time

# Flag to signal program termination
stop_event = threading.Event()

# Initialize camera, controller, and INA219
camera = CameraStreamer()
controller = AutoCarController()
ina219 = INA219(addr=0x41)  # Initialize INA219 with address 0x41

# Function to run the camera in a separate thread
def run_camera():
    try:
        camera.start_stream()
    except Exception as e:
        print(f"Error in camera thread: {e}")
    finally:
        if stop_event.is_set():
            camera.stop_stream()

# Function to run the controller in a separate thread
def run_controller():
    # Override on_release to set stop_event when ESC is pressed
    original_on_release = controller.on_release

    def new_on_release(key):
        result = original_on_release(key)
        if result is False:  # ESC is pressed
            stop_event.set()  # Set the stop flag
        return result

    controller.on_release = new_on_release

    try:
        controller.start()
    except Exception as e:
        print(f"Error in controller thread: {e}")
    finally:
        stop_event.set()  # Ensure the stop flag is set when the controller stops

# Function to read INA219 data every 30 seconds in a separate thread
def read_ina219():
    while not stop_event.is_set():
        try:
            bus_voltage = ina219.getBusVoltage_V()  # Voltage on V- (load side)
            shunt_voltage = ina219.getShuntVoltage_mV() / 1000  # Voltage between V+ and V- across the shunt
            power = ina219.getPower_W()  # Power in W
            p = (bus_voltage - 9) / 3.6 * 100  # Calculate percentage
            if p > 100:
                p = 100
            if p < 0:
                p = 0

            # Print the INA219 readings
            print("=== INA219 Readings ===")
            print("Load Voltage:  {:6.3f} V".format(bus_voltage))
            print("Power:         {:6.3f} W".format(power))
            print("Percent:       {:3.1f}%".format(p))
            print("======================")

            # Wait for 30 seconds before the next reading
            for _ in range(30):
                if stop_event.is_set():
                    break
                time.sleep(1)
        except Exception as e:
            print(f"Error in INA219 thread: {e}")
            time.sleep(1)  # Wait briefly before retrying

# Create threads for camera, controller, and INA219
camera_thread = threading.Thread(target=run_camera)
controller_thread = threading.Thread(target=run_controller)
ina219_thread = threading.Thread(target=read_ina219)

# Start the threads
camera_thread.start()
controller_thread.start()
ina219_thread.start()

# Wait for the threads to complete
try:
    controller_thread.join()  # Wait for the controller to stop (when ESC is pressed)
    stop_event.wait()  # Wait for the stop flag to be set
    camera.stop_stream()  # Stop the camera
    camera_thread.join()  # Wait for the camera thread to stop
    ina219_thread.join()  # Wait for the INA219 thread to stop
except KeyboardInterrupt:
    print("Program stopped by user (Ctrl+C)")
    stop_event.set()  # Set the stop flag
    camera.stop_stream()  # Stop the camera
finally:
    camera.stop_stream()  # Ensure the camera stops in all cases