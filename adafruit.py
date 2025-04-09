import time
from pyftdi.ftdi import Ftdi
from pyftdi.i2c import I2cController, I2cIOError

# --- Configuration ---
FTDI_DEVICE_URL = 'ftdi://ftdi:232h/1' # Common URL, adjust if you have multiple FTDI devices
SLAVE_ADDRESS = 0x42                 # Your STM32's 7-bit I2C address
STRING_TO_SEND = "hello world"
USE_LENGTH_PREFIX = True             # <<< SET TO TRUE/FALSE based on STM32 receiver code
LENGTH_BYTES_COUNT = 2               # Use 2 bytes for the length prefix
LENGTH_BYTE_ORDER = 'big'            # Use big-endian (network byte order) for length

def main():
    """Connects to FT232H and sends a string over I2C."""

    i2c = I2cController()
    configured_ok = False # <<< ADD A FLAG

    try:
        print(f"Attempting to configure FTDI device at: {FTDI_DEVICE_URL}")
        # Configure the first interface (index 0) of the FTDI device as an I2C master
        # You might need to adjust clock speed (frequency) if default (100k) is wrong
        i2c.configure(FTDI_DEVICE_URL, frequency=100000) # 100 kHz clock
        print("FTDI device configured successfully for I2C.")

        # Get the I2C port object for the specific slave address
        slave = i2c.get_port(SLAVE_ADDRESS)
        print(f"Got I2C port for slave address 0x{SLAVE_ADDRESS:02X}")

        # --- Prepare Data ---
        data_bytes = STRING_TO_SEND.encode('utf-8')
        print(f"String to send: '{STRING_TO_SEND}' ({len(data_bytes)} bytes)")

        # --- Perform I2C Write ---
        if USE_LENGTH_PREFIX:
            if len(data_bytes) >= (1 << (LENGTH_BYTES_COUNT * 8)):
                print(f"Error: String length ({len(data_bytes)}) exceeds capacity of {LENGTH_BYTES_COUNT}-byte length field.")
                return

            length_bytes = len(data_bytes).to_bytes(LENGTH_BYTES_COUNT, LENGTH_BYTE_ORDER)
            print(f"Sending length prefix: {length_bytes.hex()} ({len(data_bytes)})")

            try:
                # Transaction 1: Send length
                slave.write(length_bytes)
                print("Length prefix sent successfully.")
                # Optional small delay can sometimes help slave process
                time.sleep(0.01)

                # Transaction 2: Send data
                print(f"Sending data bytes: {data_bytes.hex()}")
                slave.write(data_bytes)
                print("Data bytes sent successfully.")

            except I2cIOError as e:
                print(f"\nI2C Error during transmission: {e}")
                print("Check wiring (SCL, SDA, GND, PULL-UPS!), slave address, and if slave is running/listening.")

        else: # Send raw string without length prefix
            print("Sending raw data bytes (no length prefix)...")
            try:
                slave.write(data_bytes)
                print("Data bytes sent successfully.")
            except I2cIOError as e:
                print(f"\nI2C Error during transmission: {e}")
                print("Check wiring (SCL, SDA, GND, PULL-UPS!), slave address, and if slave is running/listening.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Could not configure or communicate with the FTDI device.")
        print("Ensure the device is connected, drivers/libusb are installed,")
        print(f"and the URL '{FTDI_DEVICE_URL}' is correct.")
        print("On macOS, you might need to unload the default Apple FTDI driver if not using libusb:")
        print("  sudo kextunload -b com.apple.driver.AppleUSBFTDI")
        print("(You may need to run `sudo kextload -b com.apple.driver.AppleUSBFTDI` later to restore it)")

    finally:
        # Close the connection to the FTDI device
        if i2c.configured:
            print("Closing FTDI device connection.")
            i2c.close()

if __name__ == '__main__':
    main()