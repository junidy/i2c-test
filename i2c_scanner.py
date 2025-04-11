#!/usr/bin/env python3

import sys
import time
import argparse # For command-line arguments
from pyftdi.ftdi import Ftdi
from pyftdi.i2c import I2cController, I2cNackError

def scan_i2c_bus(ftdi_url='ftdi://ftdi:232h/1', frequency=100000):
    """Scans the I2C bus for responding slave devices at a specified frequency."""

    print(f"Attempting to connect to FTDI device at: {ftdi_url}")
    try:
        i2c = I2cController()
        # --- Configure with specified frequency ---
        print(f"Configuring I2C frequency to approximately {frequency} Hz...")
        i2c.configure(ftdi_url, frequency=frequency)
        # Note: pyftdi might adjust the frequency slightly based on FTDI clock divisors
        actual_frequency = i2c.frequency
        print(f"FTDI device configured. Actual I2C frequency: {actual_frequency:.2f} Hz")
        # --- End configuration change ---
    except Exception as e:
        print(f"Error: Could not configure FTDI device: {e}", file=sys.stderr)
        print("Ensure the FT232H is connected and the URL is correct.", file=sys.stderr)
        print("Common URLs: 'ftdi://ftdi:232h/1', 'ftdi://::/1'", file=sys.stderr)
        sys.exit(1)

    print("Scanning I2C bus (7-bit addresses)...")
    found_devices = []

    # Standard I2C addresses are 7-bit (0x08 to 0x77)
    for address in range(0x08, 0x78):
        try:
            port = i2c.get_port(address)
            # Try a minimal write to check for ACK
            port.write(b'') # Send address + Write bit, expect ACK

            # If write didn't raise I2cNackError, the device ACKed
            print(f"  ACK received from address: 0x{address:02X}")
            found_devices.append(address)
            # time.sleep(0.01)

        except I2cNackError:
            # Expected NACK for empty addresses
            pass
        except Exception as e:
            print(f"  Error accessing address 0x{address:02X}: {e}", file=sys.stderr)

        # Increased delay might be needed at very low frequencies
        time.sleep(0.05) # Slightly longer delay between addresses

    # Clean up the I2C controller connection
    try:
        i2c.terminate()
        print("FTDI connection terminated.")
    except Exception as e:
        print(f"Warning: Error during termination: {e}", file=sys.stderr)

    if found_devices:
        print("\nFound devices at the following 7-bit addresses:")
        for addr in found_devices:
            print(f"  - 0x{addr:02X}")
    else:
        print("\nNo I2C devices found responding on the bus.")
        print("Check wiring, pull-up resistors, device power, STM32 I2C init/address, and FTDI connection.")
        print(f"Also ensure the STM32 I2C timing parameters match the target frequency (~{frequency} Hz).")

if __name__ == "__main__":
    # --- Add command line argument parsing ---
    parser = argparse.ArgumentParser(description='Scan I2C bus using FT232H at a specified frequency.')
    parser.add_argument('-u', '--url', default='ftdi://::/1',
                        help='FTDI device URL (default: ftdi://::/1)')
    parser.add_argument('-f', '--frequency', type=int, default=100000, # Default to 100kHz
                        help='I2C clock frequency in Hz (default: 100000)')
    args = parser.parse_args()
    # --- End argument parsing ---

    if args.frequency <= 0:
        print("Error: Frequency must be a positive value.", file=sys.stderr)
        sys.exit(1)

    # Pass URL and frequency to the scanner function
    scan_i2c_bus(ftdi_url=args.url, frequency=args.frequency)