from smbus2 import SMBus

class Driver:
    
    MODE_RC = 0
    MODE_DC = 1
    MODE_SET_ADDR = 2
    
    S1, S2 = 1, 2           # Servo motors
    MA, MB = 0, 1           # DC motors
    CW, CCW = 0, 1          # Clockwise, Counterclockwise
    
    def __init__(self, address=0x40, bus_number=1):
        """
        Initialize a L9110 object with an I2C address and bus number.

        Args:
            address: The I2C address of the device (default: 0x40).
            bus_number: The I2C bus number (default: 1 on Raspberry Pi).
        """
        self._address = address
        self._bus = SMBus(bus_number)  # Initialize the bus once

    def _map_range(self, x, in_min, in_max, out_min, out_max):
        """
        Map a value x from a range (in_min, in_max) to a new range (out_min, out_max).

        Args:
            x: The value to map.
            in_min: The minimum of the original range.
            in_max: The maximum of the original range.
            out_min: The minimum of the new range.
            out_max: The maximum of the new range.

        Returns:
            The mapped value.
        """
        return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min
    
    def send_i2c_command(self, command_data):
        """
        Send I2C command to the L9110 device.

        Args:
            command_data: The I2C command data (address + data).
        """
        try:
            self._bus.write_i2c_block_data(command_data[0], command_data[0], command_data[1:])
        except IOError as e:
            raise IOError(f"Failed to send I2C command: {e}") from e

    def control_rc(self, servo, angle):
        """
        Send a control command to the servo via I2C.

        Args:
            servo: Servo motor (1 for S1, 2 for S2).
            angle: Servo angle (0-180 degrees).
        """
        if servo not in (self.S1, self.S2):
            raise ValueError("Servo must be 1 (S1) or 2 (S2)")
        if not 0 <= angle <= 180:
            raise ValueError("Angle must be between 0 and 180 degrees")

        pulse_width = self._map_range(angle, 0, 180, 500, 2500)
        command_data = [
            self._address, 
            self.MODE_RC, 
            servo, 
            (pulse_width >> 8) & 0xFF, 
            pulse_width & 0xFF, 
            0
        ]
        command_data[5] = sum(command_data) & 0xFF  # 8-bit checksum
        self.send_i2c_command(command_data)

    def control_dc(self, motor, speed_percent, rotation_direction):
        """
        Send a control command to the DC motor via I2C.

        Args:
            motor: Motor identifier (0 for MA, 1 for MB).
            speed_percent: Motor speed (0-100%).
            rotation_direction: Rotation direction (0 for CW, 1 for CCW).
        """
        if motor not in (self.MA, self.MB):
            raise ValueError("Motor must be 0 (MA) or 1 (MB)")
        if not 0 <= speed_percent <= 100:
            raise ValueError("Speed percent must be between 0 and 100")
        if rotation_direction not in (self.CW, self.CCW):
            raise ValueError("Rotation direction must be 0 (CW) or 1 (CCW)")

        speed_value = self._map_range(speed_percent, 0, 100, 0, 255)
        command_data = [
            self._address, 
            self.MODE_DC, 
            motor, 
            speed_value, 
            rotation_direction, 
            0
        ]
        command_data[5] = sum(command_data) & 0xFF  # 8-bit checksum

        self.send_i2c_command(command_data)

    def set_address(self, new_address):
        """
        Change the I2C address of the L9110 device.

        Args:
            new_address: The new I2C address.
        """
        if not 0x03 <= new_address <= 0x77:
            raise ValueError("New address must be in the range 0x03-0x77")

        try:
            self._bus.write_i2c_block_data(self._address, self.MODE_SET_ADDR, [new_address])
            self._address = new_address  # Update the address
        except IOError as e:
            raise IOError(f"Failed to change address: {e}") from e

    def __enter__(self):
        """Support using the class with a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the I2C bus when exiting the context."""
        self._bus.close()


# Example usage
if __name__ == "__main__":
    with Driver() as l9110:
        # control servo S1 to 90 degrees
        l9110.control_rc(l9110.S1, 90)

        # control DC motor MA to 50% speed clockwise
        l9110.control_dc(l9110.MA, 50,l9110.CW)

        # control DC motor MB to 50% speed counter-clockwise
        l9110.control_dc(l9110.MB, 50, l9110.CCW)