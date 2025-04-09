import time
from pyftdi.i2c import I2cController, I2cNackError

# --- Configuration ---
STM32_I2C_ADDRESS = 0x42  # Must match the address set in STM32 CubeMX
FTDI_URL = 'ftdi://ftdi:232h/1' # Or ftdi://ftdi:ft232h/1 - check your device description
I2C_FREQUENCY = 100_000  # 100 kHz (Start slow)
MESSAGE_TO_SEND = "Hello STM32!"
# --- End Configuration ---

print("Initializing FTDI device...")
i2c = I2cController()

try:
    # Configure the first interface (index 0) of the FTDI device as an I2C master
    i2c.configure(FTDI_URL, frequency=I2C_FREQUENCY)
    print(f"FTDI I2C Master configured at {I2C_FREQUENCY/1000:.1f} kHz")

    # Get a port to the specified I2C slave address
    slave = i2c.get_port(STM32_I2C_ADDRESS)
    print(f"Got I2C port for slave address {hex(STM32_I2C_ADDRESS)}")

    # Convert the string message to bytes
    data_bytes = MESSAGE_TO_SEND.encode('utf-8')

    print(f"Attempting to send: '{MESSAGE_TO_SEND}' ({len(data_bytes)} bytes)")

    # Write the data bytes to the slave
    # The write() function handles Start, Address+W, Data, ACKs, Stop
    slave.write(data_bytes)

    print("Data sent successfully!")

except I2cNackError:
    print(f"ERROR: Slave at address {hex(STM32_I2C_ADDRESS)} did not ACK.")
    print("Check: Wiring, Pull-ups, STM32 I2C initialization, STM32 I2C Address setting.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Is the FT232H connected? Correct URL? libusb drivers installed?")
finally:
    # Close the FTDI device connection
    if i2c.configured:
        i2c.close()
        print("FTDI connection closed.")